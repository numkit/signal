/// @file analog_filters.hpp
/// @ingroup group_signal
// toolboxes/signal/include/numkit/signal/filter_design/analog_filters.hpp
//
// Analog filter prototypes + lowpass-to-X frequency transformations +
// bilinear z-transform + analog freq response. Together these form the
// machinery underneath cheby1 / cheby2 / ellip / besself; the
// top-level filters compose prototype → lp2X → bilinear.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>

#include <tuple>

namespace numkit::signal {

/// @addtogroup group_signal
/// @{


// ─────────────────────────────────────────────────────────────────────
// Analog lowpass prototypes (cutoff Ω = 1 rad/s)
//
// Each returns (z, p, k) — zeros, poles, gain. `z` may be empty
// (all-pole filters); `p` is always a COMPLEX column vector; `k` is a
// real DOUBLE scalar.
// ─────────────────────────────────────────────────────────────────────

/// Butterworth analog lowpass prototype.
///
/// All poles lie on the unit circle in the s-plane. Maximally flat
/// magnitude response. No finite zeros (all-pole).
///
/// @param N   Filter order, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z, p, k)` — empty zeros, N poles, scalar gain.
///
/// @see cheb1ap, cheb2ap, besselap, ellipap
std::tuple<Value, Value, Value>
buttap(keel::i32 N, keel::Allocator *mr = nullptr);

/// Chebyshev type-I analog lowpass prototype.
///
/// Equiripple passband at Rp dB; no finite zeros (all-pole).
///
/// @param N   Filter order, ≥ 1.
/// @param Rp  Passband ripple in dB, > 0.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z, p, k)`.
std::tuple<Value, Value, Value>
cheb1ap(keel::i32 N, keel::f64 Rp, keel::Allocator *mr = nullptr);

/// Chebyshev type-II analog lowpass prototype.
///
/// Monotonic passband, equiripple stopband at Rs dB. Has finite zeros
/// on the imaginary axis.
///
/// @param N   Filter order, ≥ 1.
/// @param Rs  Stopband attenuation in dB, > 0.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z, p, k)`.
std::tuple<Value, Value, Value>
cheb2ap(keel::i32 N, keel::f64 Rs, keel::Allocator *mr = nullptr);

/// Bessel-Thompson analog lowpass prototype.
///
/// Poles of a reverse Bessel polynomial. Maximally flat group delay
/// (linear phase) in the passband. `besselap` normalises so
/// |H(jΩ)| = 1/√2 at Ω = 1 rad/s — group-delay normalisation, not
/// magnitude — and this function matches.
///
/// @param N   Filter order, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z, p, k)` — empty zeros, N poles, scalar gain.
std::tuple<Value, Value, Value>
besselap(keel::i32 N, keel::Allocator *mr = nullptr);

/// Elliptic (Cauer) analog lowpass prototype.
///
/// Equiripple in both bands. Algorithm: Sophocleous / Orfanidis formulas
/// built on Jacobi elliptic functions and the degree equation
/// `K(k')/K(k) = (1/N) · K(k1')/K(k1)`. Has finite zeros on the
/// imaginary axis (transmission zeros).
///
/// @param N   Filter order, ≥ 1.
/// @param Rp  Passband ripple in dB, > 0.
/// @param Rs  Stopband attenuation in dB, > Rp.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z, p, k)`.
std::tuple<Value, Value, Value>
ellipap(keel::i32 N, keel::f64 Rp, keel::f64 Rs, keel::Allocator *mr = nullptr);

// ─────────────────────────────────────────────────────────────────────
// Lowpass → X frequency transformations on (z, p, k)
// ─────────────────────────────────────────────────────────────────────

/// Scale an analog lowpass prototype to cutoff `Wo`.
///
/// Applies the substitution `s → s / Wo`. The result is an analog
/// lowpass filter with cutoff Wo rad/s.
///
/// @param z   Prototype zeros (may be empty).
/// @param p   Prototype poles.
/// @param k   Prototype gain.
/// @param Wo  Target cutoff frequency in rad/s.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z', p', k')` of the transformed filter.
std::tuple<Value, Value, Value>
lp2lp(const Value &                z,
      const Value &                p,
      keel::f64                       k,
      keel::f64                       Wo,
      keel::Allocator *  mr = nullptr);

/// Lowpass → highpass frequency transformation.
///
/// Applies the substitution `s → Wo / s`. Result is an analog
/// highpass filter with cutoff Wo rad/s.
///
/// @param z   Prototype zeros.
/// @param p   Prototype poles.
/// @param k   Prototype gain.
/// @param Wo  Target cutoff frequency in rad/s.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z', p', k')`.
std::tuple<Value, Value, Value>
lp2hp(const Value &                z,
      const Value &                p,
      keel::f64                       k,
      keel::f64                       Wo,
      keel::Allocator *  mr = nullptr);

/// Lowpass → bandpass frequency transformation.
///
/// Applies the substitution `s → (s² + Wo²) / (Bw · s)`. Order doubles.
///
/// @param z   Prototype zeros.
/// @param p   Prototype poles.
/// @param k   Prototype gain.
/// @param Wo  Centre frequency in rad/s.
/// @param Bw  Bandwidth in rad/s (Wo_high − Wo_low).
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z', p', k')`.
std::tuple<Value, Value, Value>
lp2bp(const Value &                z,
      const Value &                p,
      keel::f64                       k,
      keel::f64                       Wo,
      keel::f64                       Bw,
      keel::Allocator *  mr = nullptr);

/// @brief Lowpass → bandstop frequency transformation
/// (`[z', p', k'] = lp2bs(z, p, k, Wo, Bw)`).
///
/// Applies the substitution `s → (Bw · s) / (s² + Wo²)`. Order doubles.
///
/// @param z   Prototype zeros.
/// @param p   Prototype poles.
/// @param k   Prototype gain.
/// @param Wo  Centre frequency in rad/s.
/// @param Bw  Bandwidth in rad/s.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(z', p', k')` of the bandstop filter.
/// @see lp2bp
std::tuple<Value, Value, Value>
lp2bs(const Value &z, const Value &p, keel::f64 k,
      keel::f64 Wo, keel::f64 Bw,
      keel::Allocator *mr = nullptr);

// ─────────────────────────────────────────────────────────────────────
// Analog → digital
// ─────────────────────────────────────────────────────────────────────

/// Bilinear (Tustin) transform of an analog filter (b, a) to digital.
///
/// Maps `s → (2/T)·(z-1)/(z+1)` with `T = 1/fs`. The optional pre-warp
/// frequency `fp` adjusts the effective sample rate so that the
/// digital response matches the analog response exactly at `fp`.
///
/// @param b   Analog numerator polynomial.
/// @param a   Analog denominator polynomial.
/// @param fs  Sample rate (Hz).
/// @param fp  Pre-warp frequency (Hz). `0.0` (default) → no pre-warp.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Tuple `(bd, ad)` — digital numerator / denominator.
///
/// @see impinvar
std::tuple<Value, Value>
bilinear(const Value &                b,
         const Value &                a,
         keel::f64                       fs,
         keel::f64                       fp = 0.0,
         keel::Allocator *  mr = nullptr);

/// Impulse-invariance design: digital filter with the same impulse-response
/// samples as the analog filter sampled at rate `fs`.
///
/// Partial-fraction-based; preserves transient response of the analog
/// filter. Aliasing may be significant if the analog filter has
/// significant energy above fs/2.
///
/// @param b    Analog numerator polynomial.
/// @param a    Analog denominator polynomial.
/// @param fs   Sample rate (Hz).
/// @param tol  Tolerance for pole multiplicity detection. Default 1e-3.
/// @param mr   Memory resource (nullptr → process default).
/// @return     Tuple `(bd, ad)` — digital coefficients.
///
/// @see bilinear
std::tuple<Value, Value>
impinvar(const Value &                b,
         const Value &                a,
         keel::f64                       fs,
         keel::f64                       tol = 1e-3,
         keel::Allocator *  mr  = nullptr);

/// Frequency response of an analog filter b(s)/a(s).
///
/// Evaluates `H(jω) = B(jω) / A(jω)` at the angular frequencies in `w`.
///
/// @param b   Analog numerator polynomial.
/// @param a   Analog denominator polynomial.
/// @param w   Angular frequencies in rad/s.
/// @param mr  Memory resource (nullptr → process default).
/// @return    COMPLEX vector of `H(jω)` values, same shape as `w`.
///
/// @see freqz
Value freqs(const Value &                b,
            const Value &                a,
            const Value &                w,
            keel::Allocator *  mr = nullptr);


/// @}
} // namespace numkit::signal
