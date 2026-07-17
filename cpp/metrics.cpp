#include "metrics.h"

#include <algorithm>
#include <cmath>

MorletKernel make_morlet(double f, double tau, double fs) {
    MorletKernel k;
    double sigma = tau / (2.0 * M_PI * f);
    double amp = 1.0 / std::sqrt(sigma * std::sqrt(M_PI));
    k.half = static_cast<int>(std::ceil(3.0 * sigma * fs));
    int n = 2 * k.half + 1;
    k.re.resize(n);
    k.im.resize(n);
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i - k.half) / fs;
        double g = amp * std::exp(-0.5 * (t / sigma) * (t / sigma));
        k.re[i] = g * std::cos(2.0 * M_PI * f * t);
        k.im[i] = g * std::sin(2.0 * M_PI * f * t);
    }
    return k;
}

double wavelet_power(const double* sig, int n, const MorletKernel& k) {
    for (int i = 0; i < n; ++i)
        if (std::isnan(sig[i])) return std::nan("");
    int half = k.half;
    int start = half;
    int end = n - 1 - half;
    if (start > end) return 0.0;
    int klen = static_cast<int>(k.re.size());
    double total = 0.0;
    int count = 0;
    for (int i = start; i <= end; ++i) {
        double r = 0.0, m = 0.0;
        const double* w = sig + (i - half);
        for (int j = 0; j < klen; ++j) {
            r += k.re[j] * w[j];
            m += k.im[j] * w[j];
        }
        total += r * r + m * m;
        ++count;
    }
    return total / count;
}

double rms_window(const double* sig, int n) {
    double sum = 0.0;
    int cnt = 0;
    for (int i = 0; i < n; ++i)
        if (!std::isnan(sig[i])) { sum += sig[i]; ++cnt; }
    if (cnt < 10) return std::nan("");
    double mean = sum / cnt;
    double ss = 0.0;
    for (int i = 0; i < n; ++i)
        if (!std::isnan(sig[i])) { double d = sig[i] - mean; ss += d * d; }
    return std::sqrt(ss / cnt);
}

Quality classify_quality(double rms) {
    if (!std::isfinite(rms)) return Quality::Poor;
    if (rms < SQ_LOW) return Quality::Good;
    if (rms < SQ_HIGH) return Quality::Marginal;
    return Quality::Poor;
}

Weights quality_weights(const Quality q[4]) {
    auto qw = [](Quality x) { return x == Quality::Good ? 1.0 : x == Quality::Marginal ? 0.5 : 0.0; };

    // Poor channels are drop candidates; drop up to the 2 worst.
    std::vector<int> poor;
    for (int c = 0; c < 4; ++c)
        if (q[c] == Quality::Poor) poor.push_back(c);
    bool dropped[4] = {false, false, false, false};
    for (std::size_t i = 0; i < poor.size() && i < 2; ++i) dropped[poor[i]] = true;

    Weights out;
    double total = 0.0;
    for (int c = 0; c < 4; ++c) {
        out.w[c] = dropped[c] ? 0.0 : qw(q[c]);
        total += out.w[c];
    }
    out.total = total;
    if (total > 0.0) {
        for (int c = 0; c < 4; ++c) out.w[c] /= total;
    } else {
        int keep = 0;
        for (int c = 0; c < 4; ++c) if (!dropped[c]) ++keep;
        if (keep > 0)
            for (int c = 0; c < 4; ++c) out.w[c] = dropped[c] ? 0.0 : 1.0 / keep;
    }
    return out;
}

double sample_entropy(const std::vector<double>& sig, int m, double r) {
    int n = static_cast<int>(sig.size());
    if (n < m + 2) return 0.0;
    int L = n - m;
    long A = 0, B = 0;
    for (int i = 0; i < L; ++i) {
        for (int j = i + 1; j < L; ++j) {
            bool match = true;
            for (int k = 0; k < m; ++k) {
                if (std::fabs(sig[i + k] - sig[j + k]) > r) { match = false; break; }
            }
            if (match) {
                ++B;
                if (std::fabs(sig[i + m] - sig[j + m]) <= r) ++A;
            }
        }
    }
    if (A == 0 || B == 0) return 0.0;
    return -std::log(static_cast<double>(A) / static_cast<double>(B));
}

namespace {

std::vector<double> coarse_grain(const std::vector<double>& x, int tau) {
    if (tau <= 1) return x;
    int n = (static_cast<int>(x.size()) / tau) * tau;
    std::vector<double> out(n / tau);
    for (int j = 0; j < n / tau; ++j) {
        double s = 0.0;
        for (int k = 0; k < tau; ++k) s += x[j * tau + k];
        out[j] = s / tau;
    }
    return out;
}

// Linearly interpolate NaN runs of length <= max_gap; leave longer runs and
// boundary runs as NaN. Mirrors utils.interpolate_short_gaps.
void interpolate_short_gaps(std::vector<double>& x, int max_gap) {
    int n = static_cast<int>(x.size());
    int i = 0;
    while (i < n) {
        if (!std::isnan(x[i])) { ++i; continue; }
        int s = i;
        while (i < n && std::isnan(x[i])) ++i;
        int e = i;  // [s, e) is the NaN run
        int len = e - s;
        if (len <= max_gap && s > 0 && e < n) {
            double a = x[s - 1], b = x[e];
            for (int j = s; j < e; ++j) {
                double frac = static_cast<double>(j - (s - 1)) / (e - (s - 1));
                x[j] = a + frac * (b - a);
            }
        }
    }
}

}  // namespace

double mse_complexity(std::vector<double> sig, int max_gap) {
    interpolate_short_gaps(sig, max_gap);
    for (double v : sig)
        if (std::isnan(v)) return std::nan("");

    double sum = 0.0;
    for (double v : sig) sum += v;
    double mean = sum / sig.size();
    double ss = 0.0;
    for (double v : sig) { double d = v - mean; ss += d * d; }
    double sigma = std::sqrt(ss / sig.size());
    if (sigma < 1e-6) return 0.0;

    double r = MSE_R_COEF * sigma;
    static const int scales[5] = {1, 3, 5, 7, 9};
    double acc = 0.0;
    for (int s : scales) {
        double se = sample_entropy(coarse_grain(sig, s), MSE_M, r);
        if (!std::isfinite(se) || se < 0.0) se = 0.0;
        acc += se;
    }
    return acc / 5.0;
}
