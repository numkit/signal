// toolboxes/signal/src/filter_analysis/predicates.cpp
//
// isfir / isstable / isminphase / ismaxphase / islinphase / isallpass.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/filter_analysis/predicates.hpp>
#include "predicates_detail.hpp"

#include <numkit/signal/internal/shims/polyfun.hpp>             // roots()
#include <numkit/signal/filter_analysis/frequency_response.hpp> // freqz()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::signal {

namespace {

constexpr keel::f64 kTol = 1e-12;

bool isTrivialA(const Value &a)
{
    if (a.isEmpty()) return true;
    const keel::usize n = a.numel();
    if (n == 0) return true;
    if (keel::abs(a.elemAsDouble(0) - 1.0) > kTol) return false;
    for (keel::usize i = 1; i < n; ++i)
        if (keel::abs(a.elemAsDouble(i)) > kTol) return false;
    return true;
}

keel::f64 maxRootRadius(const Value &p, keel::Allocator *mr)
{
    if (p.numel() < 2) return 0.0;
    auto r = numkit::builtin::roots(p, mr);
    keel::f64 mx = 0.0;
    if (r.isComplex()) {
        const keel::c64 *src = r.complexData();
        for (keel::usize i = 0; i < r.numel(); ++i)
            mx = keel::max(mx, keel::abs(src[i]));
    } else {
        const keel::f64 *src = r.doubleData();
        for (keel::usize i = 0; i < r.numel(); ++i)
            mx = keel::max(mx, keel::abs(src[i]));
    }
    return mx;
}

void predicates_argcheck(const Value &b, const Value &a)
{
    (void)b;
    (void)a;
}

} // namespace

// Strip trailing zeros from a polynomial vector — they don't change the
// polynomial value but may confuse symmetry tests on b.
keel::Array<keel::f64> trimTrailingZeros(const Value &p)
{
    keel::Array<keel::f64> v;
    v.reserve(p.numel());
    for (keel::usize i = 0; i < p.numel(); ++i)
        v.push_back(p.elemAsDouble(i));
    while (v.size() > 1 && keel::abs(v.back()) <= kTol) v.pop_back();
    return v;
}

bool isSymmetric(keel::ArrayRef<keel::f64> v, keel::f64 scale, keel::f64 tol)
{
    const keel::usize n = v.size();
    for (keel::usize i = 0; i < n / 2; ++i)
        if (keel::abs(v[i] - scale * v[n - 1 - i]) > tol * (1.0 + keel::abs(v[i])))
            return false;
    return true;
}


// ── isfir ─────────────────────────────────────────────────────────────
bool isfir(const Value &b)
{
    predicates_argcheck(b, Value());
    return true;   // single-arg form: anything with implicit a=1 is FIR
}

bool isfir(const Value &b, const Value &a)
{
    predicates_argcheck(b, a);
    return isTrivialA(a);
}

// ── isstable ──────────────────────────────────────────────────────────
bool isstable(const Value & /*b*/, const Value &a, keel::Allocator *mr)
{
    if (isTrivialA(a)) return true;          // FIR is always stable
    return maxRootRadius(a, mr) < 1.0 - 1e-9;
}

// ── isminphase ────────────────────────────────────────────────────────
bool isminphase(const Value &b, const Value &a, keel::Allocator *mr)
{
    if (!isstable(b, a, mr)) return false;
    return maxRootRadius(b, mr) < 1.0 - 1e-9;
}

// ── ismaxphase ────────────────────────────────────────────────────────
bool ismaxphase(const Value &b, const Value &a, keel::Allocator *mr)
{
    // All zeros must be OUTSIDE the unit circle. Implementation: invert
    // the polynomial (reverse coefficients) and check that *its* roots
    // are inside — equivalent and avoids min(abs(roots(b))) edge cases
    // when some roots are exactly on the unit circle.
    if (b.numel() < 2) return false;          // constant has no zeros
    auto coeffs = trimTrailingZeros(b);
    if (coeffs.size() < 2) return false;
    keel::reverse(coeffs.begin(), coeffs.end());
    auto reversed = Value::matrix(1, coeffs.size(), ValueType::DOUBLE, mr);
    keel::f64 *dst = reversed.doubleDataMut();
    for (keel::usize i = 0; i < coeffs.size(); ++i) dst[i] = coeffs[i];
    if (maxRootRadius(reversed, mr) >= 1.0 - 1e-9)
        return false;
    // Plus the filter must be FIR (denominator trivial) — IIR maxphase
    // is rare; MATLAB checks denominator separately.
    return isTrivialA(a);
}

// ── islinphase ────────────────────────────────────────────────────────
bool islinphase(const Value &b, const Value &a)
{
    if (!isTrivialA(a)) return false;        // IIR ≠ linear phase
    auto v = trimTrailingZeros(b);
    if (v.size() <= 1) return true;
    return isSymmetric(v, +1.0) || isSymmetric(v, -1.0);
}

// ── isallpass ─────────────────────────────────────────────────────────
// True when |H(e^{jw})| is constant. For real-coefficient transfer
// functions: a == flip(b) (up to a constant scale).
bool isallpass(const Value &b, const Value &a)
{
    auto bv = trimTrailingZeros(b);
    auto av = trimTrailingZeros(a);
    if (bv.size() != av.size() || bv.empty()) return false;
    // Find the scale factor between leading coefficient of a and trailing of b.
    if (keel::abs(bv.back()) < kTol) return false;
    const keel::f64 scale = av.front() / bv.back();
    for (keel::usize i = 0; i < bv.size(); ++i) {
        const keel::f64 expected = scale * bv[bv.size() - 1 - i];
        if (keel::abs(av[i] - expected) > 1e-9 * (1.0 + keel::abs(av[i])))
            return false;
    }
    return true;
}

} // namespace numkit::signal
