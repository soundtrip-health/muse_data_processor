#ifndef MUSEPROC_JSONL_READER_H
#define MUSEPROC_JSONL_READER_H

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ring_buffer.h"

// Number of EEG channels and their fixed Muse ordering (electrode index).
//   0 = TP9, 1 = AF7, 2 = AF8, 3 = TP10
static constexpr int NCH = 4;
static constexpr int EEG_SAMPLES_PER_PACKET = 12;

// A single parsed `"type":"eeg"` record.
struct EegRecord {
    long index = 0;          // raw 16-bit packet counter (shared across 4 electrodes)
    int electrode = -1;      // 0..3
    int n = 0;               // sample count (normally 12)
    double samples[EEG_SAMPLES_PER_PACKET] = {0};
    bool has_t = false;      // whether a `t` (ms since capture epoch) field was present
    double t_ms = 0.0;
};

// Parse one JSONL line. Returns true and fills `out` iff the line is an eeg
// record. All other record types (ppg/accel/gyro/bands/hr/mse/entrain/meta)
// return false. `timestamp` is intentionally ignored (float32-quantized).
bool parse_eeg_line(const std::string& line, EegRecord& out);

// Streaming ingest: turns a sequence of eeg records into per-channel sample
// buffers laid out on a regular fs-Hz grid. Handles 16-bit counter unwrap,
// grouping the 4 electrodes of one packet, gap NaN-fill, an optional causal
// 60 Hz notch, and rolling buffers sized for the longest metric window.
class EegStream {
public:
    // `cap` is the per-channel rolling buffer capacity (>= largest window).
    EegStream(std::size_t cap, double fs);

    void set_notch(bool enabled, double line_hz);

    // Feed one raw line. If it completes a packet, its 12 samples/channel are
    // appended to the grid (with any preceding gap filled by NaN). Non-eeg or
    // incomplete lines are no-ops. Returns the number of grid samples appended.
    long feed(const std::string& line);

    long grid_len() const { return grid_len_; }         // total samples on the grid
    double t_offset() const { return t_offset_; }        // seconds anchor of sample 0
    double fs() const { return fs_; }

    // Copy the last `n` grid samples of channel `ch` (chronological, NaN pad).
    void copy_window(int ch, std::size_t n, double* out) const {
        rings_[ch].copy_last(n, out, kNaN);
    }
    std::size_t available() const { return rings_[0].size(); }

private:
    struct Partial {
        std::array<std::vector<double>, NCH> ch;
        int have = 0;  // bitmask of electrodes present
    };

    void write_packet(long unwrapped_idx, const std::array<std::vector<double>, NCH>& block,
                      bool has_t, double t_ms);
    void append_gap(long n_samples);
    void append_sample(int ch, double v, bool is_gap);

    static const double kNaN;

    double fs_;
    std::size_t cap_;
    std::vector<RingBuffer> rings_;

    // 16-bit counter unwrap state, keyed per electrode.
    std::array<long, NCH> prev_raw_{{-1, -1, -1, -1}};
    std::array<long, NCH> wraps_{{0, 0, 0, 0}};

    std::unordered_map<long, Partial> partials_;

    bool have_first_ = false;
    long first_idx_ = 0;
    long grid_len_ = 0;       // number of samples appended so far (incl. gaps)
    double t_offset_ = 0.0;

    // Causal biquad notch state (Direct Form II transposed), per channel.
    bool notch_ = false;
    double b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
    std::array<double, NCH> z1_{{0, 0, 0, 0}};
    std::array<double, NCH> z2_{{0, 0, 0, 0}};
};

#endif  // MUSEPROC_JSONL_READER_H
