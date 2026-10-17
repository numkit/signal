// toolboxes/signal/include/nk/signal/spectral_analysis/spectral_metrics.hpp
//
// Spectral measurement functions.

#pragma once

#include <limits>
#include <memory_resource>
#include <string>
#include <tuple>
#include <numkit/value/value.hpp>

namespace numkit::signal {

/// @addtogroup group_signal
/// @{


/// @file
/// @ingroup group_signal
/// @brief Spectral measurement functions.
///
/// **Conventions:**
/// - `fs` defaults to `Value::Empty`. With `fs.isEmpty()` frequencies
///   span `[0, π]` rad/sample; with `fs > 0` they span `[0, fs/2]` Hz.
/// - Functions compute the PSD internally via a single-segment
///   periodogram.

/// @brief Power in a frequency band (`p = bandpower(x, fs, freqrange)`).
///
/// - `fs.isEmpty()`: returns `mean(x^2)` (Parseval-equivalent total
///   power, ignoring `freqrange`).
/// - `fs > 0` and `freqrange.isEmpty()`: integrates the full PSD.
/// - `fs > 0` and `freqrange = [f1 f2]`: integrates the PSD over
///   `[f1, f2]`.
///
/// @param x          Real 1-D signal.
/// @param fs         Sample rate in Hz, or `Value::Empty`.
/// @param freqrange  Optional `[fLo, fHi]` band edges in Hz.
/// @param mr         Memory resource (nullptr → process default).
/// @return           Scalar DOUBLE power.
Value bandpower(const Value &x,
                const Value &fs = Value::Empty,
                const Value &freqrange = Value::Empty,
                std::pmr::memory_resource *mr = nullptr);

Value bandpowerPsd(const Value &pxx,
                   const Value &f,
                   const Value &freqrange = Value::Empty,
                   std::pmr::memory_resource *mr = nullptr);

/// @brief Power-weighted mean frequency.
///
/// @f$ \dfrac{\sum f \cdot P}{\sum P} @f$.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty` for rad/sample.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar mean frequency.
/// @see medfreq, powerbw
Value meanfreq(const Value &x, const Value &fs = Value::Empty,
               std::pmr::memory_resource *mr = nullptr);

/// @brief Median frequency.
///
/// `F` such that the cumulative PSD below `F` equals half the total.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar median frequency.
/// @see meanfreq
Value medfreq(const Value &x, const Value &fs = Value::Empty,
              std::pmr::memory_resource *mr = nullptr);

/// @brief Equivalent noise bandwidth of a window
/// (`b = enbw(window, fs)`).
///
/// @f$ \mathrm{ENBW} = f_s \cdot \sum w_n^2 \big/ (\sum w_n)^2 @f$.
/// Dimensionless when `fs.isEmpty()`.
///
/// @param window  Window vector.
/// @param fs      Sample rate, or `Value::Empty`.
/// @param mr      Memory resource (nullptr → process default).
/// @return        Scalar ENBW.
Value enbw(const Value &window, const Value &fs = Value::Empty,
           std::pmr::memory_resource *mr = nullptr);

/// @brief Occupied bandwidth at fraction `p` of total power
/// (`[bw, flo, fhi, power] = obw(x, fs, p)`).
///
/// Bandwidth (Hz or rad/s) of the contiguous frequency band that holds
/// fraction `p` (default 0.99) of the total power, centred so that the
/// `(1-p)/2` tail lies on each side. Matches MATLAB `obw`: the PSD is a
/// rectangular-windowed periodogram with `nfft = numel(x)` (no zero-pad),
/// the cumulative power is the rectangle rule `Σ P·df`, and a band edge is
/// the frequency (at the bin's upper edge `F + df/2`) where the cumulative
/// reaches `(1∓p)/2` of the total.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate or `Value::Empty`.
/// @param p   Power fraction (default 0.99).
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(bw, flo, fhi, power)`: bandwidth, lower/upper band
///            edges, and the in-band power (`p · total`).
std::tuple<Value, Value, Value, Value>
obw(const Value &x, const Value &fs = Value::Empty,
    double p = 0.99,
    std::pmr::memory_resource *mr = nullptr);

/// @brief 3-dB power bandwidth around the dominant peak.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar bandwidth in Hz / rad/s.
/// @see obw, meanfreq
Value powerbw(const Value &x, const Value &fs = Value::Empty,
              std::pmr::memory_resource *mr = nullptr);

/// @brief Spectral crest: `max(P) / mean(P)` (peakedness).
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar crest.
/// @see spectralflatness
Value spectralcrest(const Value &x, const Value &fs = Value::Empty,
                    std::pmr::memory_resource *mr = nullptr);

/// @brief Spectral flatness: `geomean(P) / mean(P)`.
///
/// `1` = white noise, `0` = tonal.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar flatness in `[0, 1]`.
/// @see spectralcrest
Value spectralflatness(const Value &x, const Value &fs = Value::Empty,
                       std::pmr::memory_resource *mr = nullptr);

/// @brief Shannon entropy of the normalised PSD.
///
/// Treats the PSD as a probability distribution over frequency bins.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar entropy.
Value spectralentropy(const Value &x, const Value &fs = Value::Empty,
                      std::pmr::memory_resource *mr = nullptr);

/// @brief Kurtosis of the PSD treated as a frequency distribution.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar spectral kurtosis.
/// @see spectralskewness
Value spectralkurtosis(const Value &x, const Value &fs = Value::Empty,
                       std::pmr::memory_resource *mr = nullptr);

/// @brief Skewness of the PSD treated as a frequency distribution.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate (Hz) or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar spectral skewness.
/// @see spectralkurtosis
Value spectralskewness(const Value &x, const Value &fs = Value::Empty,
                       std::pmr::memory_resource *mr = nullptr);

// ─────────────────────────────────────────────────────────────────────
// Harmonic / SNR family
//
// All compute fundamental power and noise / harmonic power on the
// periodogram, then return the ratio in dB. Default behaviour treats
// the dominant peak (excluding DC) as the fundamental and uses N = 6
// harmonics.
// ─────────────────────────────────────────────────────────────────────

/// @brief Signal-to-noise ratio in dB.
///
/// Power of the fundamental over the noise floor (everything outside
/// the fundamental and the first 6 harmonic skirts).
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar SNR in dB.
/// @see sinad, thd, sfdr
Value snr(const Value &x, const Value &fs = Value::Empty,
          std::pmr::memory_resource *mr = nullptr);

/// @brief Signal-to-noise-and-distortion in dB.
///
/// Like @ref snr but treats harmonic distortion as noise.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar SINAD in dB.
/// @see snr, thd
Value sinad(const Value &x, const Value &fs = Value::Empty,
            std::pmr::memory_resource *mr = nullptr);

/// @brief Total harmonic distortion in dB.
///
/// Power sum of the first 6 harmonics over the fundamental.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar THD in dB.
/// @see snr, sinad
Value thd(const Value &x, const Value &fs = Value::Empty,
          std::pmr::memory_resource *mr = nullptr);

/// @brief Spurious-free dynamic range in dB.
///
/// Distance from the fundamental peak to the next-largest spur.
///
/// @param x   Real 1-D signal.
/// @param fs  Sample rate or `Value::Empty`.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Scalar SFDR in dB.
/// @see snr
Value sfdr(const Value &x, const Value &fs = Value::Empty,
           std::pmr::memory_resource *mr = nullptr);

// ─────────────────────────────────────────────────────────────────────
// Power spectrum / spectrogram and time-frequency moment metrics
// ─────────────────────────────────────────────────────────────────────

struct PspectrumOptions {
    std::string type = "power";   // "power" or "spectrogram"
    double leakage = 0.5;         // in [0, 1]
    double overlapPercent = -1.0; // -1 = automatic from leakage; else in [0, 100)
    double timeResolution = -1.0; // -1 = automatic; else > 0 in seconds
    double fmin = std::numeric_limits<double>::quiet_NaN();
    double fmax = std::numeric_limits<double>::quiet_NaN();
    bool twoSided = false;
    bool hasTwoSided = false;
};

struct PspectrumResult {
    Value p;
    Value f;
    Value t;
};

/// @brief Power spectrum or spectrogram via Kaiser-windowed zoom CZT.
PspectrumResult pspectrum(const Value &x,
                          const Value &fsOrT = Value::Empty,
                          const PspectrumOptions &opts = {},
                          std::pmr::memory_resource *mr = nullptr);

/// @brief Instantaneous frequency via first conditional spectral moment
/// over the `pspectrum` spectrogram.
Value instfreq(const Value &x, const Value &fs = Value::Empty,
               std::pmr::memory_resource *mr = nullptr);

/// @brief Instantaneous frequency and time vector (`[ifreq, t] = instfreq(...)`).
std::tuple<Value, Value>
instfreqWithTime(const Value &x, const Value &fs = Value::Empty,
                 const PspectrumOptions &opts = {},
                 std::pmr::memory_resource *mr = nullptr);

/// @brief Instantaneous frequency via Hilbert analytic-signal phase derivative
/// (`instfreq(..., 'Method', 'hilbert')`).
std::tuple<Value, Value>
instfreqHilbert(const Value &x, const Value &fs = Value::Empty,
                std::pmr::memory_resource *mr = nullptr);

/// @brief Instantaneous bandwidth via second central conditional spectral moment
/// over the `pspectrum` spectrogram.
Value instbw(const Value &x, const Value &fs = Value::Empty,
             std::pmr::memory_resource *mr = nullptr);

/// @brief Instantaneous bandwidth and time vector (`[ibw, t] = instbw(...)`).
std::tuple<Value, Value>
instbwWithTime(const Value &x, const Value &fs = Value::Empty,
               const PspectrumOptions &opts = {},
               std::pmr::memory_resource *mr = nullptr);


/// @}
} // namespace numkit::signal
