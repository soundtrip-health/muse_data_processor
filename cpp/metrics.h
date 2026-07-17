#ifndef MUSEPROC_METRICS_H
#define MUSEPROC_METRICS_H

#include <vector>

// ---- Constants (verbatim from analysis/utils.py) ----
static constexpr double EEG_FS_DEFAULT = 256.0;
static constexpr int EEG_WIN = 256;    // 1 s band-power / quality window
static constexpr int MSE_WIN = 2048;   // 8 s multiscale-entropy window
static constexpr double SQ_LOW = 50.0;   // µV: good < 50
static constexpr double SQ_HIGH = 100.0; // µV: marginal 50–100, poor > 100
static constexpr int MSE_M = 2;
static constexpr double MSE_R_COEF = 0.15;

enum class Quality { Good, Marginal, Poor };

// Precomputed Morlet wavelet kernel (BOSC convention).
struct MorletKernel {
    std::vector<double> re, im;
    int half = 0;
};
MorletKernel make_morlet(double f, double tau, double fs);

// Mean |W|^2 over the edge-free centre of `sig` (length n). Returns NaN if any
// sample in `sig` is NaN, or if the window is too short for the kernel.
double wavelet_power(const double* sig, int n, const MorletKernel& k);

// Mean-subtracted RMS over the finite samples of sig[0..n). NaN if < 10 finite.
double rms_window(const double* sig, int n);
Quality classify_quality(double rms);  // NaN rms -> Poor

// Drop up to the 2 worst Poor channels; weights good=1.0/marginal=0.5/poor=0,
// normalized to sum 1. Returns per-channel weights and their pre-norm total.
struct Weights {
    double w[4] = {0, 0, 0, 0};
    double total = 0.0;  // sum of pre-normalization weights (0 => no usable channel)
};
Weights quality_weights(const Quality q[4]);

// Sample entropy (Richman & Moorman): m, tolerance r, Chebyshev distance,
// single-counted i<j pairs. Returns 0 on degenerate input.
double sample_entropy(const std::vector<double>& sig, int m, double r);

// Multiscale-entropy scalar: mean SampEn over scales {1,3,5,7,9} of the
// quality-weighted average signal `sig` (length MSE_WIN, may contain NaN).
// Linearly interpolates NaN runs <= max_gap; returns NaN if longer gaps remain.
double mse_complexity(std::vector<double> sig, int max_gap);

#endif  // MUSEPROC_METRICS_H
