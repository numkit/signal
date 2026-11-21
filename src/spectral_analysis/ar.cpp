// toolboxes/signal/src/spectral_analysis/ar.cpp
//
// Parametric AR-model PSD estimators: pyulear and pburg. Both fit
// an AR(p) model to the input and evaluate
//
//   Pxx(f) = σ² / |1 + Σ a_k · e^{−jωk}|²
//
// on a one-sided ω ∈ [0, π] grid. They differ only in how the AR
// coefficients are estimated:
//
//   pyulear : Yule-Walker autocorrelation → Levinson-Durbin recursion
//   pburg   : minimise Σ (forward² + backward²) prediction error,
//             building a, σ², and the lattice reflection coefficients
//             jointly per Burg's update rule.
//
// Both give similar PSDs on stationary input; pburg is preferred on
// short signals because it doesn't bias σ² downward.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace {

void ar_argcheck(const Value &x, int p, const char *fn) {
    if (p < 1)
        throw Error(keel::format("{}: order p must be ≥ 1", fn),
                    0, 0, fn, "", "numkit:ar:order");
    if (keel::as<keel::usize>(p) >= x.numel())
        throw Error(keel::format("{}: order p must be < length(x)", fn),
                    0, 0, fn, "", "numkit:ar:order");
}

// Evaluate Pxx(f) = σ² / |A(e^{jω})|² on a one-sided grid of nOut bins.
// `a` holds AR coefficients (a_1 … a_p), and `sigma2` the prediction-
// error variance. Returns (Pxx, F) Values sized nOut × 1.
std::tuple<Value, Value>
arSpectrum(keel::ArrayRef<keel::f64> a, keel::f64 sigma2, keel::usize nfft, keel::Allocator *mr)
{
    const keel::usize nOut = nfft / 2 + 1;
    Value Pxx = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    keel::f64 *pd = Pxx.doubleDataMut();
    keel::f64 *fd = F.doubleDataMut();
    for (keel::usize k = 0; k < nOut; ++k) {
        const keel::f64 w = M_PI * keel::as<keel::f64>(k) /
                                   keel::as<keel::f64>(nOut - 1);
        keel::c64 A(1.0, 0.0);
        for (keel::usize i = 0; i < a.size(); ++i) {
            const keel::c64 z =
                keel::polar(1.0, -w * keel::as<keel::f64>(i + 1));
            A += a[i] * z;
        }
        const keel::f64 mag2 = keel::norm(A);
        pd[k] = (mag2 > 0.0) ? sigma2 / mag2 : 0.0;
        fd[k] = w;
    }
    return std::make_tuple(std::move(Pxx), std::move(F));
}

} // anonymous

// Yule-Walker fit shared by pyulear / aryule / lpc.
// Returns AR coefficients a[0..p-1] (representing
// 1 + a_1·z^-1 + … + a_p·z^-p) and the prediction-error variance σ²
// via Levinson-Durbin recursion on the biased autocorrelation.
namespace {
struct YWFit { keel::Array<keel::f64> a; keel::f64 sigma2; };

YWFit yuleWalkerFit(const Value &x, keel::usize pp) {
    const keel::usize N = x.numel();
    keel::Array<keel::f64> r(pp + 1, 0.0);
    for (keel::usize k = 0; k <= pp; ++k) {
        keel::f64 s = 0.0;
        for (keel::usize n = 0; n + k < N; ++n)
            s += x.elemAsDouble(n) * x.elemAsDouble(n + k);
        r[k] = s / keel::as<keel::f64>(N);
    }
    keel::Array<keel::f64> a(pp, 0.0);
    keel::f64 sigma2 = r[0];
    if (pp >= 1) {
        keel::Array<keel::f64> aPrev(pp, 0.0);
        const keel::f64 k1 = -r[1] / r[0];
        a[0] = k1;
        sigma2 = r[0] * (1.0 - k1 * k1);
        for (keel::usize m = 1; m < pp; ++m) {
            keel::f64 num = r[m + 1];
            for (keel::usize k = 0; k < m; ++k) num += a[k] * r[m - k];
            const keel::f64 kappa = -num / sigma2;
            for (keel::usize k = 0; k < m; ++k) aPrev[k] = a[k];
            for (keel::usize k = 0; k < m; ++k)
                a[k] = aPrev[k] + kappa * aPrev[m - 1 - k];
            a[m] = kappa;
            sigma2 *= (1.0 - kappa * kappa);
        }
    }
    return {std::move(a), sigma2};
}

} // anonymous

std::tuple<Value, Value>
pyulear(const Value &x, int p, keel::usize nfft, keel::Allocator *mr)
{
    ar_argcheck(x, p, "pyulear");
    if (nfft == 0) nfft = 256;
    auto fit = yuleWalkerFit(x, keel::as<keel::usize>(p));
    return arSpectrum(fit.a, fit.sigma2, nfft, mr);
}

// `aryule` / `lpc` live in toolboxes/signal/src/spectral_analysis/
// signal_modeling.cpp — predates this TU. We re-use the
// yuleWalkerFit helper above for pyulear; the existing aryule / lpc
// implementations have been around since before the parametric-AR
// stack was even on the parity map.

std::tuple<Value, Value>
pburg(const Value &x, int p, keel::usize nfft, keel::Allocator *mr)
{
    ar_argcheck(x, p, "pburg");
    const keel::usize N = x.numel();
    const keel::usize pp = keel::as<keel::usize>(p);
    if (nfft == 0) nfft = (N >= 256) ? 256 : 256;

    // Forward / backward prediction error sequences (start = signal).
    keel::Array<keel::f64> f(N, mr), b(N, mr);
    for (keel::usize n = 0; n < N; ++n) {
        const keel::f64 v = x.elemAsDouble(n);
        f[n] = v;
        b[n] = v;
    }

    keel::Array<keel::f64> a(pp, 0.0, mr);
    keel::Array<keel::f64> aPrev(pp, 0.0, mr);

    // Initial prediction-error variance σ² = (1/N) Σ x²
    keel::f64 sigma2 = 0.0;
    for (keel::usize n = 0; n < N; ++n) {
        const keel::f64 v = x.elemAsDouble(n);
        sigma2 += v * v;
    }
    sigma2 /= keel::as<keel::f64>(N);

    for (keel::usize m = 0; m < pp; ++m) {
        // kappa = -2 · Σ f[n+1] · b[n]   /   Σ (f[n+1]² + b[n]²)
        // n ranges over 0..(N - m - 2).
        keel::f64 num = 0.0, den = 0.0;
        for (keel::usize n = 0; n + 1 < N - m; ++n) {
            num += f[n + 1] * b[n];
            den += f[n + 1] * f[n + 1] + b[n] * b[n];
        }
        const keel::f64 kappa = (den > 0.0) ? (-2.0 * num / den) : 0.0;

        // AR coefficient update: a' [k] = a[k] + kappa · a[m-1-k];
        // a' [m] = kappa.
        for (keel::usize k = 0; k < m; ++k) aPrev[k] = a[k];
        for (keel::usize k = 0; k < m; ++k)
            a[k] = aPrev[k] + kappa * aPrev[m - 1 - k];
        a[m] = kappa;
        sigma2 *= (1.0 - kappa * kappa);

        // Update forward / backward prediction errors in place.
        // f_new[n] = f[n+1] + kappa · b[n]
        // b_new[n] = b[n]   + kappa · f[n+1]
        // for n = 0..(N - m - 2). After this the active length is
        // N - m - 1.
        const keel::usize L = N - m - 1;
        keel::Array<keel::f64> fNew(L, mr), bNew(L, mr);
        for (keel::usize n = 0; n < L; ++n) {
            fNew[n] = f[n + 1] + kappa * b[n];
            bNew[n] = b[n]     + kappa * f[n + 1];
        }
        f = std::move(fNew);
        b = std::move(bNew);
    }
    return arSpectrum(a, sigma2, nfft, mr);
}

} // namespace numkit::signal
