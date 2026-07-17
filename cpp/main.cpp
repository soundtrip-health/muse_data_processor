// museproc — streaming Muse EEG biomarker processor.
//
// Reads a Muse JSONL session (or stdin) and emits a real-time text time-series:
//   t_seconds  entropy  theta_alpha  alpha_symmetry  quality
// where entropy is the multiscale-entropy scalar, theta_alpha the quality-
// weighted band-power ratio, alpha_symmetry = ln(alpha_AF8) - ln(alpha_AF7),
// and quality a blended 0-1 data-quality index. See README.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "jsonl_reader.h"
#include "metrics.h"

namespace {

struct Options {
    std::string path;
    int hop = 128;          // samples between emits (~2 Hz at 256 Hz)
    bool csv = false;
    bool realtime = false;
    bool notch = true;
    double line_hz = 60.0;
    double fs = EEG_FS_DEFAULT;
};

void usage() {
    std::fprintf(stderr,
        "usage: museproc [options] <session.jsonl | ->\n"
        "  --hop N        samples between output rows (default 128, ~2 Hz)\n"
        "  --csv          comma-separated output (default space)\n"
        "  --realtime     pace output to wall-clock time (emulate live stream)\n"
        "  --no-notch     disable the causal 60 Hz notch filter\n"
        "  --line-hz F    mains frequency for the notch (default 60)\n"
        "  --fs F         EEG sample rate (default 256)\n");
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_d = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
        if (a == "--hop") { if (i + 1 < argc) o.hop = std::atoi(argv[++i]); }
        else if (a == "--csv") o.csv = true;
        else if (a == "--realtime") o.realtime = true;
        else if (a == "--no-notch") o.notch = false;
        else if (a == "--line-hz") next_d(o.line_hz);
        else if (a == "--fs") next_d(o.fs);
        else if (a == "-h" || a == "--help") { usage(); return false; }
        else if (!a.empty() && a[0] == '-' && a != "-") {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        }
        else o.path = a;
    }
    if (o.path.empty()) { usage(); return false; }
    if (o.hop < 1) o.hop = 1;
    return true;
}

std::string fmt(double v) {
    if (std::isnan(v)) return "nan";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 1;

    EegStream stream(MSE_WIN, o.fs);
    stream.set_notch(o.notch, o.line_hz);
    MorletKernel k_theta = make_morlet(6.0, 4.0, o.fs);
    MorletKernel k_alpha = make_morlet(10.0, 6.0, o.fs);

    std::istream* in = &std::cin;
    std::ifstream file;
    if (o.path != "-") {
        file.open(o.path);
        if (!file) { std::fprintf(stderr, "cannot open %s\n", o.path.c_str()); return 1; }
        in = &file;
    }

    const char sep = o.csv ? ',' : ' ';
    std::printf("# t%centropy%ctheta_alpha%calpha_symmetry%cquality\n", sep, sep, sep, sep);
    std::fflush(stdout);

    std::vector<std::vector<double>> ch256(NCH, std::vector<double>(EEG_WIN));
    std::vector<std::vector<double>> ch2048(NCH, std::vector<double>(MSE_WIN));
    std::vector<double> mse_sig(MSE_WIN);

    auto t_start = std::chrono::steady_clock::now();

    auto emit = [&]() {
        long L = stream.grid_len();
        double t = stream.t_offset() + static_cast<double>(L) / o.fs;

        // ---- last-256 windows: per-channel quality + band power ----
        Quality q[NCH];
        double rms[NCH], theta_ch[NCH], alpha_ch[NCH];
        for (int c = 0; c < NCH; ++c) {
            stream.copy_window(c, EEG_WIN, ch256[c].data());
            rms[c] = rms_window(ch256[c].data(), EEG_WIN);
            q[c] = classify_quality(rms[c]);
            theta_ch[c] = wavelet_power(ch256[c].data(), EEG_WIN, k_theta);
            alpha_ch[c] = wavelet_power(ch256[c].data(), EEG_WIN, k_alpha);
        }
        Weights wt = quality_weights(q);

        // theta/alpha ratio: quality-weighted raw band powers.
        double theta_alpha = std::nan("");
        {
            double num = 0.0, den = 0.0;
            bool ok = wt.total > 0.0;
            for (int c = 0; c < NCH && ok; ++c) {
                if (wt.w[c] <= 0.0) continue;
                if (std::isnan(theta_ch[c]) || std::isnan(alpha_ch[c])) { ok = false; break; }
                num += wt.w[c] * theta_ch[c];
                den += wt.w[c] * alpha_ch[c];
            }
            if (ok && den > 0.0) theta_alpha = num / den;
        }

        // Frontal alpha symmetry: ln(alpha_AF8) - ln(alpha_AF7)  (ch2 - ch1).
        double alpha_symmetry = std::nan("");
        if (!std::isnan(alpha_ch[1]) && !std::isnan(alpha_ch[2])) {
            const double eps = 1e-12;
            alpha_symmetry = std::log(alpha_ch[2] + eps) - std::log(alpha_ch[1] + eps);
        }

        // ---- last-2048 window: multiscale-entropy complexity ----
        double entropy = std::nan("");
        if (stream.available() >= static_cast<std::size_t>(MSE_WIN) && wt.total > 0.0) {
            for (int c = 0; c < NCH; ++c) stream.copy_window(c, MSE_WIN, ch2048[c].data());
            for (int i = 0; i < MSE_WIN; ++i) {
                double v = 0.0;
                bool bad = false;
                for (int c = 0; c < NCH; ++c) {
                    if (wt.w[c] <= 0.0) continue;
                    double x = ch2048[c][i];
                    if (std::isnan(x)) { bad = true; break; }
                    v += wt.w[c] * x;
                }
                mse_sig[i] = bad ? std::nan("") : v;
            }
            entropy = mse_complexity(mse_sig, 13);
        }

        // ---- blended data-quality index ----
        int n_good = 0;
        double best_rms = std::nan("");
        for (int c = 0; c < NCH; ++c) {
            if (q[c] == Quality::Good) ++n_good;
            if (!std::isnan(rms[c]) && (std::isnan(best_rms) || rms[c] < best_rms)) best_rms = rms[c];
        }
        double cleanliness = std::isnan(best_rms) ? 0.0 : 1.0 - std::min(best_rms, 100.0) / 100.0;
        double quality = 0.5 * (n_good / 4.0) + 0.5 * cleanliness;

        if (o.realtime) {
            auto target = t_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                        std::chrono::duration<double>(static_cast<double>(L) / o.fs));
            std::this_thread::sleep_until(target);
        }

        std::printf("%s%c%s%c%s%c%s%c%s\n", fmt(t).c_str(), sep, fmt(entropy).c_str(), sep,
                    fmt(theta_alpha).c_str(), sep, fmt(alpha_symmetry).c_str(), sep,
                    fmt(quality).c_str());
        std::fflush(stdout);
    };

    long next_emit = EEG_WIN;
    std::string line;
    while (std::getline(*in, line)) {
        long added = stream.feed(line);
        if (added <= 0) continue;
        if (stream.grid_len() >= next_emit) {
            emit();
            next_emit = ((stream.grid_len() / o.hop) + 1) * o.hop;
        }
    }
    return 0;
}
