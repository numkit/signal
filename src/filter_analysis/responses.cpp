// toolboxes/signal/src/filter_analysis/responses.cpp
//
// impz / impzlength / stepz / phasedelay / zerophase.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/filter_analysis/responses.hpp>

#include <numkit/signal/internal/shims/polyfun.hpp>     // roots()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <tuple>
#include <numkit/signal/digital_filtering/filter.hpp>   // filter()
#include <numkit/signal/filter_analysis/frequency_response.hpp>  // freqz, phasez

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace {

// Build a length-n column vector of doubles from a generator.
template <typename Gen>
Value makeColVector(keel::usize n, Gen &&g, keel::Allocator *mr)
{
    auto v = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *dst = v.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) dst[i] = g(i);
    return v;
}

// 0-based time-index column vector [0, 1, ..., n-1] as doubles.
Value sampleIndex(keel::usize n, keel::Allocator *mr)
{
    return makeColVector(n, [](keel::usize i) { return static_cast<double>(i); }, mr);
}

// True if `a` is "the trivial denominator" — scalar 1 or a vector
// whose only non-zero element is a(0) == 1 (within tolerance).
bool isTrivialA(const Value &a)
{
    if (a.isEmpty()) return true;
    const keel::usize n = a.numel();
    if (n == 0) return true;
    if (keel::abs(a.elemAsDouble(0) - 1.0) > 1e-12) return false;
    for (keel::usize i = 1; i < n; ++i)
        if (keel::abs(a.elemAsDouble(i)) > 1e-12) return false;
    return true;
}

// Largest-magnitude root of a polynomial. Empty / scalar → 0.
double maxRootRadius(const Value &p, keel::Allocator *mr)
{
    if (p.numel() < 2) return 0.0;
    auto r = numkit::builtin::roots(p, mr);
    double mx = 0.0;
    if (r.isComplex()) {
        const keel::c64 *src = r.complexData();
        for (keel::usize i = 0; i < r.numel(); ++i)
            mx = std::max(mx, keel::abs(src[i]));
    } else {
        const double *src = r.doubleData();
        for (keel::usize i = 0; i < r.numel(); ++i)
            mx = std::max(mx, keel::abs(src[i]));
    }
    return mx;
}

void responses_argcheck(const Value &b, const Value &a)
{
    (void)b;
    (void)a;
}

} // namespace

// ── impzlength ────────────────────────────────────────────────────────
//
// Reference convention: returns the number of samples needed for the
// impulse response to decay to 0.00005 (= 5e-5) of its peak. Formula:
//   N = floor(log(5e-5) / log(rho))
// where rho is the largest pole magnitude.
keel::usize impzlength(const Value &b, const Value &a, keel::Allocator *mr)
{
    responses_argcheck(b, a);
    if (isTrivialA(a)) {
        // FIR: response length is degree + 1 = numel(b).
        return std::max<keel::usize>(b.numel(), 1);
    }
    const double rho = maxRootRadius(a, mr);
    if (!(rho > 0.0)) {
        // Trivial / no IIR contribution -> fall back to FIR length.
        return std::max<keel::usize>(b.numel(), 1);
    }
    if (rho >= 1.0) {
        // Unstable / undefined -> cap so callers don't accidentally
        // allocate a huge buffer. MATLAB also caps but at a larger
        // value; 8192 is a safe practical limit.
        return 8192;
    }
    // Decay to 5e-5 of initial amplitude.
    const double decayThresh = 5e-5;
    const double n = std::log(decayThresh) / std::log(rho);
    long N = static_cast<long>(std::floor(n));
    if (N < 1) N = 1;
    if (N > 8192) N = 8192;
    return static_cast<keel::usize>(N);
}

// ── impz ──────────────────────────────────────────────────────────────
std::tuple<Value, Value>
impz(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    if (n == 0) n = impzlength(b, a, mr);
    auto x = makeColVector(n, [](keel::usize i) { return i == 0 ? 1.0 : 0.0; }, mr);
    auto h = filter(b, a, x, mr);
    return std::make_tuple(std::move(h), sampleIndex(n, mr));
}

// ── stepz ─────────────────────────────────────────────────────────────
std::tuple<Value, Value>
stepz(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    if (n == 0) n = impzlength(b, a, mr);
    auto x = makeColVector(n, [](keel::usize) { return 1.0; }, mr);
    auto s = filter(b, a, x, mr);
    return std::make_tuple(std::move(s), sampleIndex(n, mr));
}

// ── phasedelay ────────────────────────────────────────────────────────
std::tuple<Value, Value>
phasedelay(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    auto [phi, w] = phasez(b, a, n, /*fs=*/0.0, mr);
    // phi is column vector of length n; w too. Compute pd = -phi/w.
    auto pd = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *dst = pd.doubleDataMut();
    const double *phiP = phi.doubleData();
    const double *wP   = w.doubleData();
    // MATLAB: pd = -phi / w. At DC (w == 0) the ratio is 0/0 (or NaN/0
    // when the filter has a zero at DC) — both degenerate to NaN. Do
    // not extrapolate; let std::nan propagate so the caller sees the
    // documented MATLAB behaviour.
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    for (keel::usize i = 0; i < n; ++i) {
        if (wP[i] == 0.0) {
            dst[i] = kNaN;
        } else {
            dst[i] = -phiP[i] / wP[i];
        }
    }
    return std::make_tuple(std::move(pd), std::move(w));
}

// ── zerophase ─────────────────────────────────────────────────────────
// Compute the zero-phase response Hr(w): for symmetric / antisymmetric
// FIR filters this is real-valued (possibly negative). General formula:
//   Hr(w) = H(e^{jw}) * exp(j * w * (N-1)/2)         (FIR symmetric)
// We just multiply freqz by exp(j*w*tau) where tau is half the order.
std::tuple<Value, Value>
zerophase(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    auto [H, w] = freqz(b, a, n, /*whole=*/false, /*fs=*/0.0, mr);
    const double tau = (b.numel() > 0)
                       ? (static_cast<double>(b.numel()) - 1.0) / 2.0
                       : 0.0;
    auto Hr = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *dst = Hr.doubleDataMut();
    const keel::c64 *Hc = H.isComplex() ? H.complexData() : nullptr;
    const double *Hd  = H.isComplex() ? nullptr : H.doubleData();
    const double *wP = w.doubleData();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::c64 hi = Hc ? Hc[i] : keel::c64(Hd[i], 0.0);
        const keel::c64 twist = keel::polar(1.0, wP[i] * tau);
        const keel::c64 r = hi * twist;
        dst[i] = r.real();   // imag part is ~0 for true linear-phase filters
    }
    return std::make_tuple(std::move(Hr), std::move(w));
}

} // namespace numkit::signal
