/// @file frequency_response.hpp
/// @ingroup group_signal
// toolboxes/signal/include/numkit/signal/filter_analysis/frequency_response.hpp
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>

#include <keel/tuple.h>

namespace numkit::signal {

/// @addtogroup group_signal
/// @{


/// Frequency response of a digital filter.
///
/// Computes
/// \f$ H(e^{j\omega}) = \frac{B(e^{j\omega})}{A(e^{j\omega})} \f$
/// on a uniform grid \f$ \omega \in [0, \pi] \f$.
///
/// @param b     Numerator polynomial (real row / column vector).
/// @param a     Denominator polynomial. For FIR, pass `{1.0}`.
///              `a[0]` must be non-zero.
/// @param npts  Number of frequency points. Default 512.
/// @param mr    Memory resource (nullptr → process default).
/// @return      Tuple `(H, W)`:
///                - `H` — `npts × 1` COMPLEX vector of response values.
///                - `W` — `npts × 1` DOUBLE vector of frequencies in rad/sample.
///
/// @code
/// auto [b, a] = butter(4, 0.3);
/// auto [H, w] = freqz(b, a, 1024);
/// // magnitude in dB:
/// // mag_dB = 20*log10(abs(H))
/// @endcode
///
/// @see phasez, grpdelay, freqs
///
/// `whole` selects the full unit circle: the grid spans `[0, 2π)` with
/// `w = 2π·(0:n-1)/n` (MATLAB `freqz(..., 'whole')`). Default is the
/// half circle `[0, π)`, `w = π·(0:n-1)/n`.
///
/// `fs` > 0 selects the sample-rate form `freqz(b, a, n, fs)`: the returned
/// frequency vector is then in Hz over `[0, fs/2)` (or `[0, fs)` with
/// `whole`). The response `H` is unchanged — only the frequency axis is
/// rescaled by `fs/(2π)`.
keel::Tuple<Value, Value>
freqz(const Value &                b,
      const Value &                a,
      keel::usize                       npts = 512,
      bool                         whole = false,
      keel::f64                       fs    = 0.0,
      keel::Allocator *  mr   = nullptr);

/// freqz(b, a, w): response at the GIVEN normalised radian frequencies
/// (the vector-w form).
keel::Tuple<Value, Value>
freqz(const Value &                b,
      const Value &                a,
      const Value &                wvec,
      keel::Allocator *  mr   = nullptr);

/// Unwrapped phase response of a digital filter.
///
/// Equivalent to `unwrap(angle(freqz(b, a, npts)))`.
///
/// @param b     Numerator polynomial.
/// @param a     Denominator polynomial.
/// @param npts  Number of frequency points. Default 512.
/// @param mr    Memory resource (nullptr → process default).
/// @return      Tuple `(phi, W)` — phase in radians and frequencies.
///
/// @see freqz, grpdelay
///
/// `fs` > 0 (the `phasez(b,a,n,fs)` form) returns the frequency vector in
/// Hz over `[0, fs/2)` instead of rad/sample; the phase values are
/// unchanged.
keel::Tuple<Value, Value>
phasez(const Value &                b,
       const Value &                a,
       keel::usize                       npts = 512,
       keel::f64                       fs   = 0.0,
       keel::Allocator *  mr   = nullptr);

/// Group delay of a digital filter.
///
/// Computes the negative derivative of the unwrapped phase response
/// with respect to ω, i.e. \f$ \tau_g(\omega) = -\frac{d\phi}{d\omega} \f$.
/// Implemented via finite-differences of `unwrap(angle(freqz(…)))` on
/// the uniform grid; endpoint values use one-sided differences.
///
/// @param b     Numerator polynomial.
/// @param a     Denominator polynomial.
/// @param npts  Number of frequency points. Default 512.
/// @param mr    Memory resource (nullptr → process default).
/// @return      Tuple `(gd, W)` — group delay in samples per radian
///              and frequency grid in rad/sample.
///
/// @see freqz, phasez, phasedelay
///
/// `fs` > 0 (the `grpdelay(b,a,n,fs)` form) returns the frequency vector in
/// Hz over `[0, fs/2)` instead of rad/sample; the group delay (in samples)
/// is unchanged.
keel::Tuple<Value, Value>
grpdelay(const Value &                b,
         const Value &                a,
         keel::usize                       npts = 512,
         keel::f64                       fs   = 0.0,
         keel::Allocator *  mr   = nullptr);


/// @}
} // namespace numkit::signal
