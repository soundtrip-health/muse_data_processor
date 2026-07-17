#include "jsonl_reader.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

const double EegStream::kNaN = std::numeric_limits<double>::quiet_NaN();

namespace {

// Find `key` (including trailing colon, e.g. "\"index\":") in `s` and set
// `pos` to the first character after the colon. Returns false if not found.
bool find_key(const std::string& s, const char* key, std::size_t& pos) {
    std::size_t k = s.find(key);
    if (k == std::string::npos) return false;
    pos = k + std::strlen(key);
    return true;
}

}  // namespace

bool parse_eeg_line(const std::string& line, EegRecord& out) {
    // Fast reject: must be an eeg record. Match the exact key/value so the
    // "eeg" inside meta's sampleRates ({"eeg":256}) does not false-positive.
    if (line.find("\"type\":\"eeg\"") == std::string::npos) return false;

    std::size_t pos;
    if (!find_key(line, "\"electrode\":", pos)) return false;
    out.electrode = static_cast<int>(std::strtol(line.c_str() + pos, nullptr, 10));
    if (out.electrode < 0 || out.electrode >= NCH) return false;

    if (!find_key(line, "\"index\":", pos)) return false;
    out.index = std::strtol(line.c_str() + pos, nullptr, 10);

    out.has_t = false;
    if (find_key(line, "\"t\":", pos)) {
        out.has_t = true;
        out.t_ms = std::strtod(line.c_str() + pos, nullptr);
    }

    if (!find_key(line, "\"samples\":[", pos)) return false;
    const char* p = line.c_str() + pos;
    int n = 0;
    while (n < EEG_SAMPLES_PER_PACKET) {
        char* end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p) break;  // no more numbers
        out.samples[n++] = v;
        p = end;
        while (*p == ' ' || *p == ',') ++p;
        if (*p == ']' || *p == '\0') break;
    }
    out.n = n;
    return n > 0;
}

EegStream::EegStream(std::size_t cap, double fs) : fs_(fs), cap_(cap) {
    rings_.reserve(NCH);
    for (int i = 0; i < NCH; ++i) rings_.emplace_back(cap);
}

void EegStream::set_notch(bool enabled, double line_hz) {
    notch_ = enabled;
    z1_.fill(0.0);
    z2_.fill(0.0);
    if (!enabled) return;
    const double Q = 30.0;
    double w0 = 2.0 * M_PI * line_hz / fs_;
    double cosw = std::cos(w0);
    double alpha = std::sin(w0) / (2.0 * Q);
    double a0 = 1.0 + alpha;
    b0_ = 1.0 / a0;
    b1_ = (-2.0 * cosw) / a0;
    b2_ = 1.0 / a0;
    a1_ = (-2.0 * cosw) / a0;
    a2_ = (1.0 - alpha) / a0;
}

void EegStream::append_gap(long n) {
    if (n <= 0) return;
    // A dropout ends a contiguous segment: reset the causal filter so noise
    // isn't smeared across the gap (mirrors notch_eeg's per-segment filtfilt).
    z1_.fill(0.0);
    z2_.fill(0.0);
    long fill = n;
    if (fill > static_cast<long>(cap_)) fill = static_cast<long>(cap_);  // buffer is all-NaN anyway
    for (long i = 0; i < fill; ++i)
        for (int ch = 0; ch < NCH; ++ch) rings_[ch].push(kNaN);
    grid_len_ += n;
}

void EegStream::write_packet(long idx, const std::array<std::vector<double>, NCH>& block,
                             bool has_t, double t_ms) {
    if (!have_first_) {
        have_first_ = true;
        first_idx_ = idx;
        t_offset_ = has_t ? t_ms / 1000.0 : 0.0;
        grid_len_ = 0;
    }
    long s = (idx - first_idx_) * EEG_SAMPLES_PER_PACKET;
    if (s < grid_len_) return;          // overlap: an earlier packet already won
    if (s > grid_len_) append_gap(s - grid_len_);

    int len = static_cast<int>(block[0].size());
    for (int i = 0; i < len; ++i) {
        for (int ch = 0; ch < NCH; ++ch) {
            double x = block[ch][i];
            double y = x;
            if (notch_ && !std::isnan(x)) {
                y = b0_ * x + z1_[ch];
                z1_[ch] = b1_ * x - a1_ * y + z2_[ch];
                z2_[ch] = b2_ * x - a2_ * y;
            }
            rings_[ch].push(y);
        }
        ++grid_len_;
    }
}

long EegStream::feed(const std::string& line) {
    EegRecord rec;
    if (!parse_eeg_line(line, rec)) return 0;

    int e = rec.electrode;
    long raw = rec.index;
    if (prev_raw_[e] >= 0 && (prev_raw_[e] - raw) > 32768) wraps_[e] += 1;
    prev_raw_[e] = raw;
    long unwrapped = raw + wraps_[e] * 65536;

    Partial& part = partials_[unwrapped];
    part.ch[e].assign(rec.samples, rec.samples + rec.n);
    part.have |= (1 << e);
    bool part_has_t = rec.has_t;
    double part_t = rec.t_ms;

    if (part.have != 0b1111) return 0;

    long before = grid_len_;
    write_packet(unwrapped, part.ch, part_has_t, part_t);
    partials_.erase(unwrapped);

    // Evict stale incomplete packets (a dropped electrode notification): once a
    // later packet has completed, anything more than 2 behind will never fill.
    for (auto it = partials_.begin(); it != partials_.end();) {
        if (it->first < unwrapped - 2)
            it = partials_.erase(it);
        else
            ++it;
    }
    return grid_len_ - before;
}
