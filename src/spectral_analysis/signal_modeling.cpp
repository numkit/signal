// toolboxes/signal/src/spectral_analysis/signal_modeling.cpp
//
// Parametric signal modelling: Levinson-Durbin recursion, Yule-Walker
// / Burg AR estimation, linear-prediction coefficients, plus the
// representation-conversion utilities the toolbox documents (poly ↔
// rc ↔ ac ↔ is ↔ lar). All implementations follow the standard real-
// data formulations; complex AR isn't covered here.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>

#include <numkit/signal/internal/shims/polyfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


#include <algorithm>

namespace numkit::signal {

namespace {

keel::Array<keel::f64> readVec(const Value &x)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> v(n);
    for (keel::usize i = 0; i < n; ++i) v[i] = x.elemAsDouble(i);
    return v;
}

Value rowVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

Value colVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

// Biased autocorrelation r[k] = (1/N) * sum_{i=0..N-1-k} x[i] * x[i+k],
// for k in 0..maxLag. Returns array of length maxLag+1.
keel::Array<keel::f64> biasedAutocorr(keel::ArrayRef<keel::f64> x, keel::i32 maxLag)
{
    const keel::usize N = x.size();
    keel::Array<keel::f64> R(maxLag + 1, 0.0);
    for (keel::i32 k = 0; k <= maxLag; ++k) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i + k < N; ++i) s += x[i] * x[i + k];
        R[k] = s / keel::as<keel::f64>(N);
    }
    return R;
}

// Core Levinson-Durbin recursion. Given r (length p+1), produces:
//   a (length p+1, a[0] = 1)
//   e (final prediction error variance)
//   k (reflection coefficients, length p)
// Standard Yule-Walker form: a is the coefficient vector of the AR
// poly 1 + a[1]*z^-1 + ... + a[p]*z^-p such that the filter
// 1/A(z) models the input autocorrelation.
struct LevinsonResult { keel::Array<keel::f64> a; keel::f64 e; keel::Array<keel::f64> k; };

LevinsonResult levinsonCore(keel::ArrayRef<keel::f64> r, keel::i32 p)
{
    LevinsonResult res;
    res.a.assign(p + 1, 0.0);
    res.k.assign(p, 0.0);
    res.a[0] = 1.0;
    if (p == 0) { res.e = (r.empty() ? 0.0 : r[0]); return res; }
    keel::f64 e = r[0];
    if (e <= 0.0) { res.e = 0.0; return res; }
    keel::Array<keel::f64> a_prev(p + 1, 0.0); a_prev[0] = 1.0;

    for (keel::i32 i = 1; i <= p; ++i) {
        // Numerator of the i-th reflection coefficient.
        keel::f64 num = r[i];
        for (keel::i32 j = 1; j < i; ++j) num += a_prev[j] * r[i - j];
        // MATLAB's levinson runs the full recursion even when the
        // residual energy goes non-positive (a non-PSD autocorrelation,
        // i.e. an unstable model). It does NOT early-exit: e simply
        // climbs back up through the (1 - k^2) factor. Match that — only
        // guard the exact-zero divide. (For valid PSD input e stays > 0,
        // so this changes nothing for the well-posed case.)
        const keel::f64 k = (e != 0.0) ? -num / e : 0.0;
        res.k[i - 1] = k;

        // Update a using a_prev: a_new[j] = a_prev[j] + k * a_prev[i-j]
        keel::Array<keel::f64> a_new(p + 1, 0.0);
        a_new[0] = 1.0;
        for (keel::i32 j = 1; j < i; ++j)
            a_new[j] = a_prev[j] + k * a_prev[i - j];
        a_new[i] = k;
        a_prev = std::move(a_new);
        e *= (1.0 - k * k);
    }
    res.a = std::move(a_prev);
    res.e = e;
    return res;
}

} // anonymous

// ── Levinson-Durbin ────────────────────────────────────────────────

std::tuple<Value, Value, Value>
levinson(const Value &r, keel::i32 n, keel::Allocator *mr)
{
    auto rv = readVec(r);
    if (n < 0) n = keel::as<keel::i32>(rv.size()) - 1;
    if (n < 0) n = 0;
    if (n + 1 > keel::as<keel::i32>(rv.size())) rv.resize(n + 1, 0.0);
    auto res = levinsonCore(rv, n);
    return std::make_tuple(rowVec(res.a, mr),
                           Value::scalar(res.e, mr),
                           colVec(res.k, mr));
}

// rlevinson(a, e): given AR poly a and final prediction error e,
// recover the autocorrelation sequence + reflection coefficients via
// step-down. Returns (R, k).
std::tuple<Value, Value>
rlevinson(const Value &a, keel::f64 efinal, keel::Allocator *mr)
{
    auto av = readVec(a);
    const keel::i32 p = keel::as<keel::i32>(av.size()) - 1;
    keel::Array<keel::f64> R(p + 1, 0.0, mr);
    keel::Array<keel::f64> k(p, 0.0, mr);

    // Step-down to recover reflection coefficients k[0..p-1].
    keel::Array<keel::f64> cur = av.clone();
    keel::f64 e = efinal;
    for (keel::i32 i = p; i >= 1; --i) {
        const keel::f64 ki = cur[i];
        k[i - 1] = ki;
        if (keel::abs(1.0 - ki * ki) < 1e-300) break;
        keel::Array<keel::f64> prev(i + 1, 0.0, mr);  // a_{i-1}
        prev[0] = 1.0;
        const keel::f64 denom = 1.0 - ki * ki;
        for (keel::i32 j = 1; j < i; ++j)
            prev[j] = (cur[j] - ki * cur[i - j]) / denom;
        cur = std::move(prev);
        e /= denom;  // step e back up
    }

    // Forward Levinson on the reflection coeffs, recovering R via
    // r[i] = -(a[1]*r[i-1] + ... + a[i-1]*r[1] + a[i]*r[0]).
    R[0] = e;  // reconstruct r[0] from prediction error climb
    for (keel::i32 i = 1; i <= p; ++i) {
        keel::f64 s = 0.0;
        for (keel::i32 j = 1; j <= i - 1; ++j) s += av[j] * R[i - j];
        R[i] = -av[i] * R[0] - s;
    }
    return std::make_tuple(rowVec(R, mr), colVec(k, mr));
}

// ── AR estimation ─────────────────────────────────────────────────

std::tuple<Value, Value, Value>
aryule(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto R = biasedAutocorr(v, p);
    auto res = levinsonCore(R, p);
    return std::make_tuple(rowVec(res.a, mr),
                           Value::scalar(res.e, mr),
                           colVec(res.k, mr));
}

std::tuple<Value, Value, Value>
arburg(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    // Burg's algorithm. Maintains forward / backward prediction
    // errors and updates AR coefficients via reflection coefficients
    // chosen to minimise the sum of forward + backward residual energy.
    auto v = readVec(x);
    const keel::usize N = v.size();
    keel::Array<keel::f64> ef = v.clone();
    keel::Array<keel::f64> eb = v.clone();
    keel::Array<keel::f64> a(p + 1, 0.0, mr); a[0] = 1.0;
    keel::Array<keel::f64> k(p, 0.0, mr);

    keel::f64 E = 0.0;
    for (keel::f64 y : v) E += y * y;
    E /= keel::as<keel::f64>(N);

    for (keel::i32 i = 0; i < p; ++i) {
        // Reflection coefficient: maximise correlation of forward and
        // backward errors.
        keel::f64 num = 0.0, den = 0.0;
        for (keel::usize n = i + 1; n < N; ++n) {
            num += ef[n] * eb[n - 1];
            den += ef[n] * ef[n] + eb[n - 1] * eb[n - 1];
        }
        const keel::f64 ki = (den > 0) ? -2.0 * num / den : 0.0;
        k[i] = ki;

        // Update prediction errors in-place.
        for (keel::usize n = N - 1; n > keel::as<keel::usize>(i); --n) {
            const keel::f64 f = ef[n];
            const keel::f64 b = eb[n - 1];
            ef[n] = f + ki * b;
            eb[n] = b + ki * f;
        }

        // Update AR coefficients via the Levinson update.
        keel::Array<keel::f64> a_new = a.clone();
        for (keel::i32 j = 1; j <= i; ++j)
            a_new[j] = a[j] + ki * a[i + 1 - j];
        a_new[i + 1] = ki;
        a = std::move(a_new);
        E *= (1.0 - ki * ki);
    }

    return std::make_tuple(rowVec(a, mr),
                           Value::scalar(E, mr),
                           colVec(k, mr));
}

std::tuple<Value, Value>
lpc(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    // MATLAB's lpc returns (a, g) where g is the prediction error
    // variance (NOT its sqrt — the docs are explicit). Same numbers
    // as aryule, just without the reflection-coef return.
    auto [a, e, k] = aryule(x, p, mr);
    (void)k;
    return std::make_tuple(std::move(a), std::move(e));
}

// ── Representation conversions ────────────────────────────────────

std::tuple<Value, Value>
ac2poly(const Value &R, keel::Allocator *mr)
{
    auto rv = readVec(R);
    const keel::i32 p = keel::as<keel::i32>(rv.size()) - 1;
    auto res = levinsonCore(rv, std::max(p, 0));
    return std::make_tuple(rowVec(res.a, mr), Value::scalar(res.e, mr));
}

Value poly2ac(const Value &a, keel::f64 e, keel::Allocator *mr)
{
    auto [Rv, kv] = rlevinson(a, e, mr);
    (void)kv;
    return Rv;
}

std::tuple<Value, Value>
ac2rc(const Value &R, keel::Allocator *mr)
{
    auto rv = readVec(R);
    const keel::i32 p = keel::as<keel::i32>(rv.size()) - 1;
    auto res = levinsonCore(rv, std::max(p, 0));
    return std::make_tuple(colVec(res.k, mr),
                           Value::scalar(rv.empty() ? 0.0 : rv[0], mr));
}

Value schurrc(const Value &R, keel::Allocator *mr)
{
    // Standalone Schur / Levinson reflection-coeff recursion. Does NOT
    // bail on negative residual energy — schurrc returns the math-valid
    // reflection coefficients even when |k| > 1 (non-PSD R, i.e. an
    // unstable AR model). levinsonCore early-exits on e<=0 to keep the
    // fitted poly stable; schurrc doesn't want that.
    auto rv = readVec(R);
    const keel::i32 p = keel::as<keel::i32>(rv.size()) - 1;
    if (p <= 0) return colVec(keel::ArrayRef<keel::f64>{}, mr);

    keel::Array<keel::f64> k(p, 0.0, mr);
    keel::Array<keel::f64> a_prev(p + 1, 0.0, mr); a_prev[0] = 1.0;
    keel::f64 e = rv[0];

    for (keel::i32 i = 1; i <= p; ++i) {
        keel::f64 num = rv[i];
        for (keel::i32 j = 1; j < i; ++j) num += a_prev[j] * rv[i - j];
        const keel::f64 ki = (e != 0.0) ? -num / e : 0.0;
        k[i - 1] = ki;

        keel::Array<keel::f64> a_new(p + 1, 0.0, mr); a_new[0] = 1.0;
        for (keel::i32 j = 1; j < i; ++j)
            a_new[j] = a_prev[j] + ki * a_prev[i - j];
        a_new[i] = ki;
        a_prev = std::move(a_new);
        e *= (1.0 - ki * ki);
    }
    return colVec(k, mr);
}

Value rc2ac(const Value &k, keel::f64 r0, keel::Allocator *mr)
{
    auto kv = readVec(k);
    const keel::i32 p = keel::as<keel::i32>(kv.size());
    // Build R from k via the inverse Levinson: r0 known, then
    // r[i] = -(a_{i-1}[1]*r[i-1] + ... + a_{i-1}[i-1]*r[1]) - k[i-1]*e_{i-1}
    keel::Array<keel::f64> R(p + 1, 0.0, mr);
    R[0] = r0;
    keel::Array<keel::f64> a(p + 1, 0.0, mr); a[0] = 1.0;
    keel::f64 e = r0;
    for (keel::i32 i = 1; i <= p; ++i) {
        const keel::f64 ki = kv[i - 1];
        keel::f64 s = 0.0;
        for (keel::i32 j = 1; j < i; ++j) s += a[j] * R[i - j];
        R[i] = -ki * e - s;
        // Update a by step-up.
        keel::Array<keel::f64> a_new(p + 1, 0.0, mr);
        a_new[0] = 1.0;
        for (keel::i32 j = 1; j < i; ++j) a_new[j] = a[j] + ki * a[i - j];
        a_new[i] = ki;
        a = std::move(a_new);
        e *= (1.0 - ki * ki);
    }
    return rowVec(R, mr);
}

std::tuple<Value, Value>
poly2rc(const Value &a, keel::f64 efinal, keel::Allocator *mr)
{
    auto av = readVec(a);
    const keel::i32 p = keel::as<keel::i32>(av.size()) - 1;
    keel::Array<keel::f64> k(p, 0.0, mr);
    keel::Array<keel::f64> cur = av.clone();
    for (keel::i32 i = p; i >= 1; --i) {
        const keel::f64 ki = cur[i];
        k[i - 1] = ki;
        const keel::f64 denom = 1.0 - ki * ki;
        if (keel::abs(denom) < 1e-300) break;
        keel::Array<keel::f64> prev(i + 1, 0.0, mr); prev[0] = 1.0;
        for (keel::i32 j = 1; j < i; ++j)
            prev[j] = (cur[j] - ki * cur[i - j]) / denom;
        cur = std::move(prev);
    }
    // Zero-lag autocorrelation R0 = efinal / prod_i (1 - k_i^2). This is
    // the second output [k, R0] = poly2rc(a, efinal): the energy of the
    // signal whose AR model is `a` with final prediction error `efinal`.
    keel::f64 prod = 1.0;
    for (keel::f64 ki : k) prod *= (1.0 - ki * ki);
    const keel::f64 r0 = (prod != 0.0) ? efinal / prod : 0.0;
    return std::make_tuple(colVec(k, mr), Value::scalar(r0, mr));
}

std::tuple<Value, Value>
rc2poly(const Value &k, keel::f64 r0, keel::Allocator *mr)
{
    auto kv = readVec(k);
    const keel::i32 p = keel::as<keel::i32>(kv.size());
    keel::Array<keel::f64> a(p + 1, 0.0, mr); a[0] = 1.0;
    for (keel::i32 i = 1; i <= p; ++i) {
        const keel::f64 ki = kv[i - 1];
        keel::Array<keel::f64> a_new(p + 1, 0.0, mr);
        a_new[0] = 1.0;
        for (keel::i32 j = 1; j < i; ++j) a_new[j] = a[j] + ki * a[i - j];
        a_new[i] = ki;
        a = std::move(a_new);
    }
    // efinal = r0 * prod_i (1 - k_i^2): the final prediction error of the
    // AR model, the inverse of poly2rc's R0 relation. Second output of
    // [a, efinal] = rc2poly(k, r0).
    keel::f64 prod = 1.0;
    for (keel::f64 ki : kv) prod *= (1.0 - ki * ki);
    return std::make_tuple(rowVec(a, mr), Value::scalar(r0 * prod, mr));
}

// is2rc / rc2is — inverse-sine parameterisation. MATLAB scales the inverse
// sine into [-1, 1]: isin = (2/pi)*asin(k), so k = sin((pi/2)*isin).
Value is2rc(const Value &is, keel::Allocator *mr)
{
    constexpr keel::f64 kHalfPi = keel::kHalfPi;  // pi/2
    auto v = readVec(is);
    for (auto &y : v) y = keel::sin(kHalfPi * y);
    return colVec(v, mr);
}
Value rc2is(const Value &k, keel::Allocator *mr)
{
    constexpr keel::f64 kTwoOverPi = 0.6366197723675814;  // 2/pi
    auto v = readVec(k);
    for (auto &y : v) {
        // Clamp into [-1, 1] to avoid asin domain errors on numerical
        // overshoots.
        const keel::f64 c = std::max(-1.0, std::min(1.0, y));
        y = kTwoOverPi * keel::asin(c);
    }
    return colVec(v, mr);
}

// lar2rc / rc2lar — log-area-ratio: lar = log((1+k)/(1-k)).
Value lar2rc(const Value &g, keel::Allocator *mr)
{
    auto v = readVec(g);
    for (auto &y : v) {
        const keel::f64 e = keel::exp(y);
        y = (e - 1.0) / (e + 1.0);
    }
    return colVec(v, mr);
}
Value rc2lar(const Value &k, keel::Allocator *mr)
{
    auto v = readVec(k);
    for (auto &y : v) {
        const keel::f64 c = std::max(-0.99999999, std::min(0.99999999, y));
        y = keel::log((1.0 + c) / (1.0 - c));
    }
    return colVec(v, mr);
}

// ════════════════════════════════════════════════════════════════════
// Covariance / modified-covariance AR + Prony + corrmtx
// ════════════════════════════════════════════════════════════════════

namespace {

// Solve a small symmetric positive-definite system A·x = b in place
// using Gaussian elimination with partial pivoting. A is row-major
// p×p; returns x. For ill-conditioned inputs (singular matrix)
// returns the partial result.
keel::Array<keel::f64>
solveSPD(keel::Array<keel::f64> &A, keel::Array<keel::f64> &b, keel::i32 p)
{
    for (keel::i32 k = 0; k < p; ++k) {
        // Pivot.
        keel::i32 piv = k;
        keel::f64 pivVal = keel::abs(A[k * p + k]);
        for (keel::i32 i = k + 1; i < p; ++i)
            if (keel::abs(A[i * p + k]) > pivVal) { pivVal = keel::abs(A[i * p + k]); piv = i; }
        if (pivVal < 1e-300) continue;
        if (piv != k) {
            for (keel::i32 j = k; j < p; ++j)
                std::swap(A[k * p + j], A[piv * p + j]);
            std::swap(b[k], b[piv]);
        }
        const keel::f64 diag = A[k * p + k];
        for (keel::i32 i = k + 1; i < p; ++i) {
            const keel::f64 f = A[i * p + k] / diag;
            for (keel::i32 j = k; j < p; ++j) A[i * p + j] -= f * A[k * p + j];
            b[i] -= f * b[k];
        }
    }
    keel::Array<keel::f64> x(p, 0.0);
    for (keel::i32 i = p - 1; i >= 0; --i) {
        keel::f64 s = b[i];
        for (keel::i32 j = i + 1; j < p; ++j) s -= A[i * p + j] * x[j];
        const keel::f64 diag = A[i * p + i];
        x[i] = (keel::abs(diag) > 1e-300) ? s / diag : 0.0;
    }
    return x;
}

// Build the covariance correlation R_{ij} = sum_{n=p..N-1} x[n-i] * x[n-j]
// for i, j in 0..p. Used by arcov / armcov.
keel::Array<keel::f64>
covMatrixForward(keel::ArrayRef<keel::f64> x, keel::i32 p)
{
    const keel::i32 N = keel::as<keel::i32>(x.size());
    keel::Array<keel::f64> R(keel::as<keel::usize>(p + 1) * keel::as<keel::usize>(p + 1), 0.0);
    for (keel::i32 n = p; n < N; ++n) {
        for (keel::i32 i = 0; i <= p; ++i)
            for (keel::i32 j = 0; j <= p; ++j)
                R[i * (p + 1) + j] += x[n - i] * x[n - j];
    }
    return R;
}

keel::Array<keel::f64>
covMatrixForwardBackward(keel::ArrayRef<keel::f64> x, keel::i32 p)
{
    const keel::i32 N = keel::as<keel::i32>(x.size());
    auto R = covMatrixForward(x, p);
    // Backward: R += sum_{n=0..N-1-p} x[n+i] x[n+j]
    for (keel::i32 n = 0; n + p < N; ++n) {
        for (keel::i32 i = 0; i <= p; ++i)
            for (keel::i32 j = 0; j <= p; ++j)
                R[i * (p + 1) + j] += x[n + i] * x[n + j];
    }
    return R;
}

// Given a (p+1)×(p+1) covariance matrix R indexed R[i,j] for i,j in
// 0..p, solve for AR coefficients a[1..p] from R[1..p, 1..p] · a =
// -R[1..p, 0]. Returns full a vector with a[0] = 1 prepended, plus
// the prediction error e = R[0,0] + sum_k a[k] * R[0,k].
struct ArCovResult { keel::Array<keel::f64> a; keel::f64 e; };
ArCovResult solveArCov(keel::ArrayRef<keel::f64> R, keel::i32 p)
{
    keel::Array<keel::f64> A(keel::as<keel::usize>(p) * keel::as<keel::usize>(p), 0.0);
    keel::Array<keel::f64> b(p, 0.0);
    for (keel::i32 i = 0; i < p; ++i) {
        for (keel::i32 j = 0; j < p; ++j) A[i * p + j] = R[(i + 1) * (p + 1) + (j + 1)];
        b[i] = -R[(i + 1) * (p + 1) + 0];
    }
    auto a_tail = solveSPD(A, b, p);
    keel::Array<keel::f64> a(p + 1, 0.0);
    a[0] = 1.0;
    for (keel::i32 k = 0; k < p; ++k) a[k + 1] = a_tail[k];
    keel::f64 e = R[0];
    for (keel::i32 k = 1; k <= p; ++k) e += a[k] * R[k];
    return {std::move(a), e};
}

} // anonymous

std::tuple<Value, Value>
arcov(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto R = covMatrixForward(v, p);
    // Normalise by the count of summed terms (N - p) so e has the
    // same units as the residual variance.
    const keel::i32 N = keel::as<keel::i32>(v.size());
    const keel::f64 norm = keel::as<keel::f64>(std::max(N - p, 1));
    for (auto &r : R) r /= norm;
    auto res = solveArCov(R, p);
    return std::make_tuple(rowVec(res.a, mr), Value::scalar(res.e, mr));
}

std::tuple<Value, Value>
armcov(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto R = covMatrixForwardBackward(v, p);
    const keel::i32 N = keel::as<keel::i32>(v.size());
    const keel::f64 norm = keel::as<keel::f64>(std::max(2 * (N - p), 1));
    for (auto &r : R) r /= norm;
    auto res = solveArCov(R, p);
    return std::make_tuple(rowVec(res.a, mr), Value::scalar(res.e, mr));
}

// ── prony ─────────────────────────────────────────────────────────
// Identify a numerator b (length nb+1) and denominator a (length na+1)
// such that the IIR filter b(z)/a(z) has impulse response approximately
// equal to h. Standard formulation:
//   1. Form the Hankel-like matrix H of `h` shifted, build the system
//      H_{na rows after first nb+1 samples} · [a[1..na]] = -h[nb+1..]
//   2. Solve by least squares.
//   3. b = (a convolved with h) truncated to nb+1 terms.
std::tuple<Value, Value>
prony(const Value &h, keel::i32 nb, keel::i32 na, keel::Allocator *mr)
{
    auto hv = readVec(h);
    const keel::i32 N = keel::as<keel::i32>(hv.size());
    if (na > 0) {
        // Build linear system M · a_tail = -rhs, where M is (N-1-nb) × na
        // and rhs is length (N-1-nb). M[i, j] = h[nb+1+i-1-j] (padded
        // with zeros on the left), rhs[i] = h[nb+1+i].
        const keel::i32 rows = std::max(N - 1 - nb, 1);
        keel::Array<keel::f64> M(keel::as<keel::usize>(rows) * keel::as<keel::usize>(na), 0.0, mr);
        keel::Array<keel::f64> rhs(rows, 0.0, mr);
        for (keel::i32 i = 0; i < rows; ++i) {
            for (keel::i32 j = 0; j < na; ++j) {
                const keel::i32 idx = nb + i - j;  // h index = nb+1+i - (j+1) = nb+i-j
                M[i * na + j] = (idx >= 0 && idx < N) ? hv[idx] : 0.0;
            }
            const keel::i32 rhs_idx = nb + 1 + i;
            rhs[i] = (rhs_idx < N) ? -hv[rhs_idx] : 0.0;
        }
        // Normal equations: (M^T M) · a_tail = M^T rhs
        keel::Array<keel::f64> MtM(keel::as<keel::usize>(na) * keel::as<keel::usize>(na), 0.0, mr);
        keel::Array<keel::f64> Mtb(na, 0.0, mr);
        for (keel::i32 i = 0; i < na; ++i) {
            for (keel::i32 j = 0; j < na; ++j) {
                keel::f64 s = 0.0;
                for (keel::i32 k = 0; k < rows; ++k) s += M[k * na + i] * M[k * na + j];
                MtM[i * na + j] = s;
            }
            keel::f64 s = 0.0;
            for (keel::i32 k = 0; k < rows; ++k) s += M[k * na + i] * rhs[k];
            Mtb[i] = s;
        }
        auto a_tail = solveSPD(MtM, Mtb, na);
        keel::Array<keel::f64> a(na + 1, 0.0, mr);
        a[0] = 1.0;
        for (keel::i32 k = 0; k < na; ++k) a[k + 1] = a_tail[k];

        // b = (a * h) truncated to nb+1 terms.
        keel::Array<keel::f64> b(nb + 1, 0.0, mr);
        for (keel::i32 i = 0; i <= nb; ++i) {
            keel::f64 s = 0.0;
            for (keel::i32 k = 0; k <= std::min(i, na); ++k)
                if (i - k < N) s += a[k] * hv[i - k];
            b[i] = s;
        }
        return std::make_tuple(rowVec(b, mr), rowVec(a, mr));
    }
    // na == 0: pure FIR. b[i] = h[i] for i = 0..nb, a = [1].
    keel::Array<keel::f64> b(nb + 1, 0.0, mr);
    for (keel::i32 i = 0; i <= nb && i < N; ++i) b[i] = hv[i];
    const keel::f64 one = 1.0;
    return std::make_tuple(rowVec(b, mr), rowVec(keel::ArrayRef<keel::f64>(&one, 1), mr));
}

// ── stmcb ─────────────────────────────────────────────────────────
// Steiglitz-McBride iteration: identify B(z)/A(z) (nb zeros, na poles) with
// approximate impulse response h. Initialise A via Prony, then repeat `niter`
// times (MATLAB default 5): prefilter the unit impulse and h by 1/A, solve the
// linear least-squares [E | -G]·[b; a_tail] ≈ g for the new coefficients.
namespace {

void stmcb_argcheck(const Value &y, const Value &x, keel::i32 nb, keel::i32 na, const Value &ai)
{
    if (nb < 0)
        throw Error("stmcb: numerator order nb must be non-negative",
                    0, 0, "stmcb", "", "numkit:stmcb:InvalidNb");
    if (na < 0)
        throw Error("stmcb: denominator order na must be non-negative",
                    0, 0, "stmcb", "", "numkit:stmcb:InvalidNa");
    if (!x.isEmpty() && x.numel() != y.numel())
        throw Error("stmcb: input and output signal vectors must have the same length",
                    0, 0, "stmcb", "", "numkit:stmcb:InvalidDimensions");
    if (!ai.isEmpty()) {
        const keel::usize n_ai = ai.numel();
        if (n_ai != keel::as<keel::usize>(na + 1) && n_ai != keel::as<keel::usize>(na))
            throw Error("stmcb: initial denominator estimate ai must have length na+1 (or na)",
                        0, 0, "stmcb", "", "numkit:stmcb:InvalidAiLength");
        if (n_ai == keel::as<keel::usize>(na + 1) && ai.elemAsDouble(0) == 0.0)
            throw Error("stmcb: leading coefficient of ai must be nonzero",
                        0, 0, "stmcb", "", "numkit:stmcb:ZeroLeadingCoefficient");
    }
}

} // anonymous

std::tuple<Value, Value>
stmcb(const Value &y, const Value &x, keel::i32 nb, keel::i32 na, keel::i32 niter,
      const Value &ai, keel::Allocator *mr)
{
    stmcb_argcheck(y, x, nb, na, ai);
    auto      yv = readVec(y);
    const keel::i32 N  = keel::as<keel::i32>(yv.size());
    if (niter < 1) niter = 5;

    keel::Array<keel::f64> xv(mr);
    if (!x.isEmpty()) {
        xv = readVec(x);
    } else {
        xv.assign(N, 0.0);
        if (N > 0) xv[0] = 1.0;
    }

    keel::Array<keel::f64> b(nb + 1, 0.0, mr);
    keel::Array<keel::f64> a(na + 1, 0.0, mr);
    a[0] = 1.0;
    if (ai.isEmpty()) {
        // Initialise the denominator estimate via Prony (default `ai`).
        auto [b0, a0] = prony(y, x.isEmpty() ? nb : 0, na, mr);
        b = readVec(b0);
        a = readVec(a0);
        b.resize(nb + 1, 0.0);
        a.resize(na + 1, 0.0);
        a[0] = 1.0;
    } else {
        auto aiv = readVec(ai);
        if (keel::as<keel::i32>(aiv.size()) == na + 1) {
            const keel::f64 a0_val = aiv[0];
            for (keel::i32 k = 0; k <= na; ++k) a[k] = aiv[k] / a0_val;
            a[0] = 1.0;
        } else {
            a[0] = 1.0;
            for (keel::i32 k = 0; k < na; ++k) a[k + 1] = aiv[k];
        }
    }

    const keel::i32           P = nb + 1 + na;   // unknowns: b[0..nb] then a[1..na]
    keel::Array<keel::f64> e(N, 0.0, mr), g(keel::as<keel::usize>(N), 0.0, mr);
    for (keel::i32 it = 0; it < niter; ++it) {
        // Prefilter the input x and output y by 1/A.
        for (keel::i32 n = 0; n < N; ++n) {
            keel::f64    se  = xv[n];
            keel::f64    sg  = yv[n];
            const keel::i32 lim = std::min(na, n);
            for (keel::i32 i = 1; i <= lim; ++i) { se -= a[i] * e[n - i]; sg -= a[i] * g[n - i]; }
            e[n] = se;
            g[n] = sg;
        }
        // [E | -G]·c ≈ g  (E: e shifted 0..nb, G: g shifted 1..na), normal eqs.
        auto Mcol = [&](keel::i32 n, keel::i32 col) -> keel::f64 {
            if (col <= nb) { const keel::i32 s = col; return (n - s >= 0) ? e[n - s] : 0.0; }
            const keel::i32 s = col - nb;             // 1..na
            return (n - s >= 0) ? -g[n - s] : 0.0;
        };
        keel::Array<keel::f64> MtM(keel::as<keel::usize>(P) * P, 0.0, mr), Mtg(keel::as<keel::usize>(P), 0.0, mr);
        for (keel::i32 n = 0; n < N; ++n)
            for (keel::i32 i = 0; i < P; ++i) {
                const keel::f64 mi = Mcol(n, i);
                if (mi == 0.0) continue;
                Mtg[i] += mi * g[n];
                for (keel::i32 j = 0; j < P; ++j) MtM[i * P + j] += mi * Mcol(n, j);
            }
        auto c = solveSPD(MtM, Mtg, P);
        for (keel::i32 k = 0; k <= nb; ++k) b[k]     = c[k];
        for (keel::i32 k = 0; k < na;  ++k) a[k + 1] = c[nb + 1 + k];
    }
    return std::make_tuple(rowVec(b, mr), rowVec(a, mr));
}

std::tuple<Value, Value>
stmcb(const Value &h, keel::i32 nb, keel::i32 na, keel::i32 niter, keel::Allocator *mr)
{
    return stmcb(h, Value(), nb, na, niter, Value(), mr);
}

// ── corrmtx ───────────────────────────────────────────────────────
// MATLAB default 'autocorrelation' method: produces the (n+m)×(m+1)
// data matrix X with X(i, j) = x(i-j+1), zero-padded outside the
// input range, such that X' * X is the (m+1)×(m+1) autocorrelation
// matrix.
Value corrmtx(const Value &x, keel::i32 m, keel::Allocator *mr)
{
    auto v = readVec(x);
    const keel::i32 N = keel::as<keel::i32>(v.size());
    const keel::i32 rows = N + m;
    const keel::i32 cols = m + 1;
    auto X = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    keel::f64 *d = X.doubleDataMut();
    std::fill(d, d + rows * cols, 0.0);
    for (keel::i32 j = 0; j < cols; ++j) {
        for (keel::i32 i = 0; i < rows; ++i) {
            const keel::i32 idx = i - j;     // x[i-j], zero outside [0, N)
            if (idx >= 0 && idx < N) d[i + j * rows] = v[idx];
        }
    }
    return X;
}

// ── LSF ↔ AR poly ────────────────────────────────────────────────
// LSF is built from P(z) = A(z) + z^{-(N+1)} A_R(z) and Q(z) = A(z)
// - z^{-(N+1)} A_R(z), where A_R is the reverse of A. The roots of
// P and Q lie on the unit circle, alternating; the LSF angles are
// the angles of those roots in (0, π).

namespace {

keel::Array<keel::f64> reverseVec(keel::ArrayRef<keel::f64> v)
{
    keel::Array<keel::f64> r(v.size());
    for (keel::usize i = 0; i < v.size(); ++i) r[i] = v[v.size() - 1 - i];
    return r;
}

// Pad poly p (descending power) to total length n with leading zeros.
[[maybe_unused]] keel::Array<keel::f64> padLeft(keel::ArrayRef<keel::f64> p, keel::i32 n)
{
    keel::Array<keel::f64> r(n, 0.0);
    const keel::i32 off = n - keel::as<keel::i32>(p.size());
    for (keel::usize i = 0; i < p.size(); ++i) r[off + i] = p[i];
    return r;
}

} // anonymous

// AR polynomial A(z) -> line-spectral frequencies.
// Form the symmetric / antisymmetric pair P(z) = A(z) + z^-(N+1)·A(z^-1)
// and Q(z) = A(z) - z^-(N+1)·A(z^-1); their roots lie on the unit
// circle and the LSFs are those roots' angles in (0, π).
// References: F. Itakura, "Line spectrum representation of linear
// predictor coefficients of speech signals", J. Acoust. Soc. Am.
// 57(S1):S35, 1975; P. Kabal & R. P. Ramachandran, "The computation of
// line spectral frequencies using Chebyshev polynomials", IEEE Trans.
// ASSP 34(6):1419-1426, 1986.
Value poly2lsf(const Value &a, keel::Allocator *mr)
{
    auto av = readVec(a);
    const keel::i32 N = keel::as<keel::i32>(av.size()) - 1;
    if (N <= 0) return colVec(keel::ArrayRef<keel::f64>{}, mr);

    // Build P = A + A_R_shifted, Q = A - A_R_shifted, both length N+2.
    auto ar = reverseVec(av);
    keel::Array<keel::f64> P(N + 2, 0.0, mr), Q(keel::as<keel::usize>(N + 2), 0.0, mr);
    for (keel::i32 i = 0; i <= N; ++i) {
        P[i] = av[i];
        Q[i] = av[i];
    }
    for (keel::i32 i = 1; i <= N + 1; ++i) {
        P[i] += ar[i - 1];
        Q[i] -= ar[i - 1];
    }
    // Find roots and pick angles in (0, π).
    keel::Array<keel::f64> angles(mr);
    const keel::ArrayRef<keel::f64> polys[2] = {P.as_ref(), Q.as_ref()};
    for (auto poly : polys) {
        Value polyV = rowVec(poly, mr);
        Value rts = numkit::builtin::roots(polyV, mr);
        const keel::usize n = rts.numel();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::c64 c = rts.complexData()[i];
            const keel::f64 a_ang = keel::atan2(c.imag(), c.real());
            if (a_ang > 1e-9 && a_ang < keel::kPi - 1e-9) angles.push_back(a_ang);
        }
    }
    std::sort(angles.begin(), angles.end());
    // Should yield N angles.
    if (keel::as<keel::i32>(angles.size()) > N) angles.resize(N);
    return colVec(angles, mr);
}

Value lsf2poly(const Value &lsf, keel::Allocator *mr)
{
    auto lv = readVec(lsf);
    const keel::i32 m = keel::as<keel::i32>(lv.size());
    if (m == 0) {
        const keel::f64 one = 1.0;
        return rowVec(keel::ArrayRef<keel::f64>(&one, 1), mr);
    }

    // LSF -> LPC conversion (Kabal & Ramachandran, IEEE TASSP, 1986).
    // The LSF angles split into two interleaved sets (even / odd
    // index) that build the symmetric / antisymmetric polynomials
    // P and Q; a = ((P + Q) / 2) with the trailing element dropped.
    //
    // Each conjugate-pair root z = e^{±iθ} contributes the real
    // quadratic factor (1 - 2cos(θ) z^-1 + z^-2). The boundary roots
    // z = -1 and z = +1 contribute the linear factors (1 + z^-1) and
    // (1 - z^-1) respectively.
    //
    // The boundary-root distribution differs by parity so that A and
    // B always end up with the same degree (m+1); their trailing
    // coefficients cancel exactly in (A+B)/2 and we drop the result.
    auto convOne = [mr](keel::ArrayRef<keel::f64> a,
                        keel::ArrayRef<keel::f64> b) {
        keel::Array<keel::f64> r(a.size() + b.size() - 1, 0.0, mr);
        for (keel::usize i = 0; i < a.size(); ++i)
            for (keel::usize j = 0; j < b.size(); ++j)
                r[i + j] += a[i] * b[j];
        return r;
    };

    keel::Array<keel::f64> P(mr), Q(mr);
    if (m % 2 == 1) {
        // m odd: P has no boundary roots, Q gets both (-1 and +1) =
        // (1-z^-2). Both end up degree m+1.
        P = keel::Array<keel::f64>({1.0}, mr);
        Q = keel::Array<keel::f64>({1.0, 0.0, -1.0}, mr);
    } else {
        // m even: P gets z=-1 → (1+z^-1); Q gets z=+1 → (1-z^-1).
        // Both end up degree m+1.
        P = keel::Array<keel::f64>({1.0,  1.0}, mr);
        Q = keel::Array<keel::f64>({1.0, -1.0}, mr);
    }
    for (keel::i32 i = 0; i < m; ++i) {
        const keel::f64 q[3] = {1.0, -2.0 * keel::cos(lv[i]), 1.0};
        if (i % 2 == 0) P = convOne(P, keel::ArrayRef<keel::f64>(q, 3));
        else            Q = convOne(Q, keel::ArrayRef<keel::f64>(q, 3));
    }

    // P and Q have identical length now. a = (P + Q) / 2, then drop
    // the trailing element (it cancels to ~0 by construction).
    keel::Array<keel::f64> a(P.size(), 0.0, mr);
    for (keel::usize i = 0; i < P.size(); ++i) a[i] = 0.5 * (P[i] + Q[i]);
    if (a.size() > 1) a.pop_back();
    return rowVec(a, mr);
}

// ════════════════════════════════════════════════════════════════════
// invfreqs / invfreqz — Levi equation-error LSQ
// ════════════════════════════════════════════════════════════════════
//
// Given a desired complex frequency response H(jω_k) (or H(e^{jω_k}))
// at K frequency points, fit (b, a) such that B(jω) ≈ H · A(jω).
// With a_0 = 1 fixed, each frequency point contributes one complex
// equation linear in the (nb + 1 + na) real unknowns:
//
//   -b_0 - b_1·xᵏ - … - b_nb·xᵏⁿᵇ + a_1·H_k·xᵏ + … + a_na·H_k·xᵏⁿᵃ = -H_k
//
// where x = jω_k for invfreqs and x = e^{-jω_k} for invfreqz (z⁻¹ form).
// Stack real and imaginary parts → 2K real equations in nv = nb+1+na
// real unknowns; solve via normal equations using solveSPD.

namespace {

keel::Array<keel::c64> readComplexVecLocal(const Value &v) {
    const keel::usize n = v.numel();
    keel::Array<keel::c64> out(n);
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

// Common solver: x is the "step" variable per frequency point —
// jω_k for invfreqs, e^{-jω_k} for invfreqz.
std::tuple<keel::Array<keel::f64>, keel::Array<keel::f64>>
invfreqLSQ(keel::ArrayRef<keel::c64> H,
           keel::ArrayRef<keel::c64> x,
           keel::i32 nb, keel::i32 na)
{
    const keel::i32 K  = keel::as<keel::i32>(H.size());
    const keel::i32 nv = nb + 1 + na;
    if (K * 2 < nv)
        throw numkit::Error("invfreq: not enough frequency points to fit (nb, na)");

    // Build A (2K × nv) row-major, plus rhs (length 2K).
    keel::Array<keel::f64> Ar(keel::as<keel::usize>(2 * K) * keel::as<keel::usize>(nv), 0.0);
    keel::Array<keel::f64> rhs(2 * K, 0.0);
    for (keel::i32 k = 0; k < K; ++k) {
        // Build x^0..x^max(nb, na) once.
        const keel::i32 M = std::max(nb, na);
        keel::Array<keel::c64> xpow(M + 1);
        xpow[0] = keel::c64(1.0, 0.0);
        for (keel::i32 i = 1; i <= M; ++i) xpow[i] = xpow[i - 1] * x[k];

        // Row k contains:
        //   -1, -x, -x², …, -x^nb, H·x, H·x², …, H·x^na
        keel::Array<keel::c64> row(nv);
        for (keel::i32 i = 0; i <= nb; ++i) row[i] = -xpow[i];
        for (keel::i32 j = 1; j <= na; ++j) row[nb + j] = H[k] * xpow[j];

        // Stack real / imag into Ar.
        for (keel::i32 c = 0; c < nv; ++c) {
            Ar[(2 * k    ) * nv + c] = row[c].real();
            Ar[(2 * k + 1) * nv + c] = row[c].imag();
        }
        rhs[2 * k    ] = -H[k].real();
        rhs[2 * k + 1] = -H[k].imag();
    }

    // Normal equations: (Aᵀ A) θ = Aᵀ rhs.
    keel::Array<keel::f64> AtA(keel::as<keel::usize>(nv) * keel::as<keel::usize>(nv), 0.0);
    keel::Array<keel::f64> Atb(nv, 0.0);
    for (keel::i32 i = 0; i < nv; ++i) {
        for (keel::i32 j = 0; j < nv; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 r = 0; r < 2 * K; ++r) s += Ar[r * nv + i] * Ar[r * nv + j];
            AtA[i * nv + j] = s;
        }
        keel::f64 s = 0.0;
        for (keel::i32 r = 0; r < 2 * K; ++r) s += Ar[r * nv + i] * rhs[r];
        Atb[i] = s;
    }
    auto theta = solveSPD(AtA, Atb, nv);

    // theta = [b_0, ..., b_nb, a_1, ..., a_na]
    keel::Array<keel::f64> b(nb + 1), a(keel::as<keel::usize>(na + 1));
    for (keel::i32 i = 0; i <= nb; ++i) b[i] = theta[i];
    a[0] = 1.0;
    for (keel::i32 j = 1; j <= na; ++j) a[j] = theta[nb + j];
    return std::make_tuple(std::move(b), std::move(a));
}

void invfreqs_argcheck(const Value &H, const Value &w, keel::i32 nb, keel::i32 na)
{
    if (H.numel() != w.numel())
        throw numkit::Error("invfreqs: H and w must be the same length");
    const keel::i32 K  = keel::as<keel::i32>(H.numel());
    const keel::i32 nv = nb + 1 + na;
    if (K * 2 < nv)
        throw numkit::Error("invfreq: not enough frequency points to fit (nb, na)");
}

void invfreqz_argcheck(const Value &H, const Value &w, keel::i32 nb, keel::i32 na)
{
    if (H.numel() != w.numel())
        throw numkit::Error("invfreqz: H and w must be the same length");
    const keel::i32 K  = keel::as<keel::i32>(H.numel());
    const keel::i32 nv = nb + 1 + na;
    if (K * 2 < nv)
        throw numkit::Error("invfreq: not enough frequency points to fit (nb, na)");
}

} // anonymous

std::tuple<Value, Value>
invfreqs(const Value &H, const Value &w, keel::i32 nb, keel::i32 na, keel::Allocator *mr)
{
    invfreqs_argcheck(H, w, nb, na);
    auto Hv = readComplexVecLocal(H);
    auto wv = readVec(w);
    keel::Array<keel::c64> x(wv.size(), mr);
    for (keel::usize k = 0; k < wv.size(); ++k) x[k] = keel::c64(0.0, wv[k]); // jω
    auto [b, a] = invfreqLSQ(Hv, x, nb, na);
    return std::make_tuple(rowVec(b, mr), rowVec(a, mr));
}

std::tuple<Value, Value>
invfreqz(const Value &H, const Value &w, keel::i32 nb, keel::i32 na, keel::Allocator *mr)
{
    invfreqz_argcheck(H, w, nb, na);
    auto Hv = readComplexVecLocal(H);
    auto wv = readVec(w);
    keel::Array<keel::c64> x(wv.size(), mr);
    for (keel::usize k = 0; k < wv.size(); ++k)
        x[k] = keel::exp(keel::c64(0.0, -wv[k])); // z⁻¹
    auto [b, a] = invfreqLSQ(Hv, x, nb, na);
    return std::make_tuple(rowVec(b, mr), rowVec(a, mr));
}

} // namespace numkit::signal
