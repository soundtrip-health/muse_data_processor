#ifndef MUSEPROC_RING_BUFFER_H
#define MUSEPROC_RING_BUFFER_H

#include <cstddef>
#include <vector>

// Fixed-capacity rolling buffer of doubles. Stores the most recent `cap`
// samples pushed; older samples fall off the front. Gap samples are stored as
// NaN by the caller. `copy_last(n, out)` copies the last `n` samples in
// chronological (oldest-first) order into `out`, left-padding with NaN when
// fewer than `n` samples have been seen.
class RingBuffer {
public:
    explicit RingBuffer(std::size_t cap) : cap_(cap), buf_(cap, 0.0) {}

    void push(double v) {
        buf_[head_] = v;
        head_ = (head_ + 1) % cap_;
        if (count_ < cap_) ++count_;
    }

    std::size_t size() const { return count_; }

    // Copy the last `n` samples (chronological order) into out[0..n).
    // If fewer than `n` samples exist, the leading slots are filled with `pad`.
    void copy_last(std::size_t n, double* out, double pad) const {
        for (std::size_t i = 0; i < n; ++i) {
            // position from newest: newest is offset 0
            std::size_t from_end = n - 1 - i;  // out[i] is `from_end` back from newest
            if (from_end >= count_) {
                out[i] = pad;
            } else {
                // index into ring: newest sample sits at (head_-1)
                std::size_t idx = (head_ + cap_ - 1 - from_end) % cap_;
                out[i] = buf_[idx];
            }
        }
    }

private:
    std::size_t cap_;
    std::size_t head_ = 0;   // next write position
    std::size_t count_ = 0;  // number of valid samples
    std::vector<double> buf_;
};

#endif  // MUSEPROC_RING_BUFFER_H
