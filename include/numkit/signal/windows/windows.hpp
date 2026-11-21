/// @file windows.hpp
/// @ingroup group_signal
// toolboxes/signal/include/numkit/signal/windows/windows.hpp
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>
#include <utility>

namespace numkit::signal {

/// @addtogroup group_signal
/// @{


// ─────────────────────────────────────────────────────────────────────
// Window generators — every function returns an N×1 DOUBLE column.
//
// Shapes and values are bit-identical for the documented forms.
// ─────────────────────────────────────────────────────────────────────

/// Hamming window of length N.
///
/// Formula: \f$ w(n) = 0.54 - 0.46 \cos\!\left(\frac{2\pi n}{N-1}\right) \f$
/// for n = 0..N-1. Symmetric (endpoints equal).
///
/// @param N   Window length, ≥ 1. `N == 1` returns the scalar 1.0.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
///
/// @code  Value w = hamming(64);  @endcode
Value hamming(keel::usize N, keel::Allocator *mr = nullptr);

/// Hann (Hanning) window of length N.
///
/// Formula: \f$ w(n) = 0.5\,(1 - \cos(2\pi n/(N-1))) \f$ for n = 0..N-1.
/// Symmetric. Endpoints are exactly zero.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value hann(keel::usize N, keel::Allocator *mr = nullptr);

/// Blackman window of length N — classic 3-term cosine sum.
///
/// Formula: \f$ w(n) = 0.42 - 0.5\cos(2\pi n/(N-1)) + 0.08\cos(4\pi n/(N-1)) \f$.
/// Symmetric. Sidelobe attenuation ≈ -58 dB.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value blackman(keel::usize N, keel::Allocator *mr = nullptr);

/// Kaiser window of length N with shape parameter β.
///
/// Formula: \f$ w(n) = I_0\!\left(\beta\sqrt{1-(2n/(N-1)-1)^2}\right) / I_0(\beta) \f$
/// where \f$ I_0 \f$ is the zeroth-order modified Bessel function.
///
/// β controls the tradeoff between mainlobe width and sidelobe height:
/// β=0 ≡ rectangular, β=5 ≈ Hamming sidelobes, β=8 ≈ Blackman sidelobes,
/// β=14 ≈ -120 dB sidelobes.
///
/// @param N     Window length, ≥ 1.
/// @param beta  Shape parameter, β ≥ 0.
/// @param mr    Memory resource (nullptr → process default).
/// @return      N×1 DOUBLE column vector.
///
/// @code  Value w = kaiser(128, 8.0);  // Blackman-like sidelobes  @endcode
Value kaiser(keel::usize N, keel::f64 beta, keel::Allocator *mr = nullptr);

/// Rectangular window of length N — all ones.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector of 1.0.
Value rectwin(keel::usize N, keel::Allocator *mr = nullptr);

/// Bartlett window of length N — triangular with zero endpoints.
///
/// Formula: \f$ w(n) = 1 - |2n - (N-1)| / (N-1) \f$ for n = 0..N-1.
/// Endpoints are exactly zero (unlike `triang`).
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value bartlett(keel::usize N, keel::Allocator *mr = nullptr);

/// Triangular window of length N — non-zero endpoints variant of Bartlett.
///
/// For odd N: \f$ w(n) = 1 - |2n - (N-1)| / N \f$.
/// For even N: \f$ w(n) = 1 - |2n - (N-1)| / (N-1) \f$.
///
/// Differs from `bartlett` only in that endpoints are non-zero (≈ 1/N).
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value triang(keel::usize N, keel::Allocator *mr = nullptr);

/// Tukey (cosine-tapered) window of length N.
///
/// First and last `r*(N-1)/2` samples are a cosine taper; the middle is
/// flat 1.0. r=0 → rectangular, r=1 → Hann.
///
/// @param N   Window length, ≥ 1.
/// @param r   Cosine-fraction in [0, 1]. Default 0.5.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value tukeywin(keel::usize N, keel::f64 r = 0.5,
               keel::Allocator *mr = nullptr);

/// Flat-top window of length N — 5-term cosine sum, optimised for
/// amplitude accuracy (not frequency resolution).
///
/// Used in calibration / amplitude-measurement applications: the
/// flat passband near DC ensures accurate peak-amplitude readout even
/// when a tone falls between bins.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value flattopwin(keel::usize N, keel::Allocator *mr = nullptr);

/// Gaussian window of length N.
///
/// Formula: \f$ w(n) = \exp\!\left(-\frac{1}{2}\left(\alpha \frac{n - (N-1)/2}{(N-1)/2}\right)^2\right) \f$.
///
/// α controls the width: smaller α → wider mainlobe / smaller sidelobes.
///
/// @param N      Window length, ≥ 1.
/// @param alpha  Shape parameter > 0. Default 2.5.
/// @param mr     Memory resource (nullptr → process default).
/// @return       N×1 DOUBLE column vector.
/// @throws       numkit::Error if α ≤ 0.
Value gausswin(keel::usize N, keel::f64 alpha = 2.5,
               keel::Allocator *mr = nullptr);

/// Dolph-Chebyshev window of length N — equiripple sidelobes at given
/// attenuation.
///
/// All sidelobes have height exactly `-at` dB below mainlobe peak.
/// Minimises mainlobe width for the given sidelobe ripple.
///
/// @param N   Window length, ≥ 1.
/// @param at  Sidelobe attenuation in dB (positive number; 100 → -100 dB).
///            Default 100.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
/// @throws    numkit::Error if `at` ≤ 0.
Value chebwin(keel::usize N, keel::f64 at = 100.0,
              keel::Allocator *mr = nullptr);

/// Parzen (de la Vallée Poussin) window of length N — piecewise cubic.
///
/// Smooth (continuous second derivative) → low sidelobes, wider mainlobe
/// than Bartlett.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value parzenwin(keel::usize N, keel::Allocator *mr = nullptr);

/// Nuttall window of length N — 4-term, continuous-first-derivative.
///
/// Coefficients chosen to minimise sidelobe height under a
/// continuous-derivative constraint. Sidelobe attenuation ≈ -98 dB.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value nuttallwin(keel::usize N, keel::Allocator *mr = nullptr);

/// Taylor window of length N — radar / antenna design.
///
/// Distributes sidelobe energy so the first `nbar` sidelobes are roughly
/// equal in height (at `sll` dB), with the rest decaying. Used in
/// pattern synthesis where the inner sidelobes matter most.
///
/// @param N     Window length, ≥ 1.
/// @param nbar  Number of nearly-equal-height sidelobes, ≥ 2. Default 4.
/// @param sll   Peak sidelobe level in dB, must be < 0. Default -30.
/// @param mr    Memory resource (nullptr → process default).
/// @return      N×1 DOUBLE column vector.
/// @throws      numkit::Error if `sll ≥ 0` or `nbar < 2`.
Value taylorwin(keel::usize N, keel::i32 nbar = 4, keel::f64 sll = -30.0,
                keel::Allocator *mr = nullptr);

/// Blackman-Harris window of length N — 4-term, minimum-sidelobe.
///
/// Like Nuttall but optimised purely for minimum sidelobe height (no
/// derivative constraint). Sidelobe attenuation ≈ -92 dB.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value blackmanharris(keel::usize N, keel::Allocator *mr = nullptr);

/// Bohman window of length N — convolution of two half-cosine pulses.
///
/// Smoother than Bartlett, lower sidelobes than Hann.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value bohmanwin(keel::usize N, keel::Allocator *mr = nullptr);

/// Modified Bartlett-Hann window of length N.
///
/// Linear combination of Bartlett and Hann: better sidelobes than Bartlett
/// while keeping a triangular envelope.
///
/// @param N   Window length, ≥ 1.
/// @param mr  Memory resource (nullptr → process default).
/// @return    N×1 DOUBLE column vector.
Value barthannwin(keel::usize N, keel::Allocator *mr = nullptr);

std::pair<Value, Value> dpss(keel::usize N, keel::f64 nw, keel::usize k,
                             keel::Allocator *mr = nullptr);

/// @}
} // namespace numkit::signal
