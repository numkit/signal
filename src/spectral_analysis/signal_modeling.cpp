// toolboxes/signal/src/spectral_analysis/signal_modeling.cpp
//
// Parametric signal modelling: Levinson-Durbin recursion, Yule-Walker
// / Burg AR estimation, linear-prediction coefficients, plus the
// representation-conversion utilities the toolbox documents (poly ↔
// rc ↔ ac ↔ is ↔ lar). All implementations follow the standard real-
// data formulations; complex AR isn't covered here.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>

#include <numkit/signal/internal/shims/polyfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace numkit::signal {

namespace {

keel::Array<double> readVec(const Value &x)
{
    const keel::usize n = x.numel();
    keel::Array<double> v(n);
    for (keel::usize i = 0; i < n; ++i) v[i] = x.elemAsDouble(i);
    return v;
}

Value rowVec(keel::ArrayRef<double> v, keel::Allocator *mr)
{
    auto out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::memcpy(out.doubleDataMut(), v.data(), v.size() * sizeof(double));
    return out;
}

Value colVec(keel::ArrayRef<double> v, keel::Allocator *mr)
{
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) std::memcpy(out.doubleDataMut(), v.data(), v.size() * sizeof(double));
    return out;
}

// Biased autocorrelation r[k] = (1/N) * sum_{i=0..N-1-k} x[i] * x[i+k],
// for k in 0..maxLag. Returns array of length maxLag+1.
keel::Array<double> biasedAutocorr(keel::ArrayRef<double> x, int maxLag)
{
    const keel::usize N = x.size();
    keel::Array<double> R(maxLag + 1, 0.0);
    for (int k = 0; k <= maxLag; ++k) {
        double s = 0.0;
        for (keel::usize i = 0; i + k < N; ++i) s += x[i] * x[i + k];
        R[k] = s / static_cast<double>(N);
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
struct LevinsonResult { keel::Array<double> a; double e; keel::Array<double> k; };

LevinsonResult levinsonCore(keel::ArrayRef<double> r, int p)
{
    LevinsonResult res;
    res.a.assign(p + 1, 0.0);
    res.k.assign(p, 0.0);
    res.a[0] = 1.0;
    if (p == 0) { res.e = (r.empty() ? 0.0 : r[0]); return res; }
    double e = r[0];
    if (e <= 0.0) { res.e = 0.0; return res; }
    keel::Array<double> a_prev(p + 1, 0.0); a_prev[0] = 1.0;

    for (int i = 1; i <= p; ++i) {
        // Numerator of the i-th reflection coefficient.
        double num = r[i];
        for (int j = 1; j < i; ++j) num += a_prev[j] * r[i - j];
        // MATLAB's levinson runs the full recursion even when the
        // residual energy goes non-positive (a non-PSD autocorrelation,
        // i.e. an unstable model). It does NOT early-exit: e simply
        // climbs back up through the (1 - k^2) factor. Match that — only
        // guard the exact-zero divide. (For valid PSD input e stays > 0,
        // so this changes nothing for the well-posed case.)
        const double k = (e != 0.0) ? -num / e : 0.0;
        res.k[i - 1] = k;

        // Update a using a_prev: a_new[j] = a_prev[j] + k * a_prev[i-j]
        keel::Array<double> a_new(p + 1, 0.0);
        a_new[0] = 1.0;
        for (int j = 1; j < i; ++j)
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
levinson(const Value &r, int n, keel::Allocator *mr)
{
    auto rv = readVec(r);
    if (n < 0) n = static_cast<int>(rv.size()) - 1;
    if (n < 0) n = 0;
    if (n + 1 > static_cast<int>(rv.size())) rv.resize(n + 1, 0.0);
    auto res = levinsonCore(rv, n);
    return std::make_tuple(rowVec(res.a, mr),
                           Value::scalar(res.e, mr),
                           colVec(res.k, mr));
}

// rlevinson(a, e): given AR poly a and final prediction error e,
// recover the autocorrelation sequence + reflection coefficients via
// step-down. Returns (R, k).
std::tuple<Value, Value>
rlevinson(const Value &a, double efinal, keel::Allocator *mr)
{
    auto av = readVec(a);
    const int p = static_cast<int>(av.size()) - 1;
    keel::Array<double> R(p + 1, 0.0, mr);
    keel::Array<double> k(p, 0.0, mr);

    // Step-down to recover reflection coefficients k[0..p-1].
    keel::Array<double> cur = av.clone();
    double e = efinal;
    for (int i = p; i >= 1; --i) {
        const double ki = cur[i];
        k[i - 1] = ki;
        if (keel::abs(1.0 - ki * ki) < 1e-300) break;
        keel::Array<double> prev(i + 1, 0.0, mr);  // a_{i-1}
        prev[0] = 1.0;
        const double denom = 1.0 - ki * ki;
        for (int j = 1; j < i; ++j)
            prev[j] = (cur[j] - ki * cur[i - j]) / denom;
        cur = std::move(prev);
        e /= denom;  // step e back up
    }

    // Forward Levinson on the reflection coeffs, recovering R via
    // r[i] = -(a[1]*r[i-1] + ... + a[i-1]*r[1] + a[i]*r[0]).
    R[0] = e;  // reconstruct r[0] from prediction error climb
    for (int i = 1; i <= p; ++i) {
        double s = 0.0;
        for (int j = 1; j <= i - 1; ++j) s += av[j] * R[i - j];
        R[i] = -av[i] * R[0] - s;
    }
    return std::make_tuple(rowVec(R, mr), colVec(k, mr));
}

// ── AR estimation ─────────────────────────────────────────────────

std::tuple<Value, Value, Value>
aryule(const Value &x, int p, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto R = biasedAutocorr(v, p);
    auto res = levinsonCore(R, p);
    return std::make_tuple(rowVec(res.a, mr),
                           Value::scalar(res.e, mr),
                           colVec(res.k, mr));
}

std::tuple<Value, Value, Value>
arburg(const Value &x, int p, keel::Allocator *mr)
{
    // Burg's algorithm. Maintains forward / backward prediction
    // errors and updates AR coefficients via reflection coefficients
    // chosen to minimise the sum of forward + backward residual energy.
    auto v = readVec(x);
    const keel::usize N = v.size();
    keel::Array<double> ef = v.clone();
    keel::Array<double> eb = v.clone();
    keel::Array<double> a(p + 1, 0.0, mr); a[0] = 1.0;
    keel::Array<double> k(p, 0.0, mr);

    double E = 0.0;
    for (double y : v) E += y * y;
    E /= static_cast<double>(N);

    for (int i = 0; i < p; ++i) {
        // Reflection coefficient: maximise correlation of forward and
        // backward errors.
        double num = 0.0, den = 0.0;
        for (keel::usize n = i + 1; n < N; ++n) {
            num += ef[n] * eb[n - 1];
            den += ef[n] * ef[n] + eb[n - 1] * eb[n - 1];
        }
        const double ki = (den > 0) ? -2.0 * num / den : 0.0;
        k[i] = ki;

        // Update prediction errors in-place.
        for (keel::usize n = N - 1; n > static_cast<keel::usize>(i); --n) {
            const double f = ef[n];
            const double b = eb[n - 1];
            ef[n] = f + ki * b;
            eb[n] = b + ki * f;
        }

        // Update AR coefficients via the Levinson update.
        keel::Array<double> a_new = a.clone();
        for (int j = 1; j <= i; ++j)
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
lpc(const Value &x, int p, keel::Allocator *mr)
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
    const int p = static_cast<int>(rv.size()) - 1;
    auto res = levinsonCore(rv, std::max(p, 0));
    return std::make_tuple(rowVec(res.a, mr), Value::scalar(res.e, mr));
}

Value poly2ac(const Value &a, double e, keel::Allocator *mr)
{
    auto [Rv, kv] = rlevinson(a, e, mr);
    (void)kv;
    return Rv;
}

std::tuple<Value, Value>
ac2rc(const Value &R, keel::Allocator *mr)
{
    auto rv = readVec(R);
    const int p = static_cast<int>(rv.size()) - 1;
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
    const int p = static_cast<int>(rv.size()) - 1;
    if (p <= 0) return colVec(keel::ArrayRef<double>{}, mr);

    keel::Array<double> k(p, 0.0, mr);
    keel::Array<double> a_prev(p + 1, 0.0, mr); a_prev[0] = 1.0;
    double e = rv[0];

    for (int i = 1; i <= p; ++i) {
        double num = rv[i];
        for (int j = 1; j < i; ++j) num += a_prev[j] * rv[i - j];
        const double ki = (e != 0.0) ? -num / e : 0.0;
        k[i - 1] = ki;

        keel::Array<double> a_new(p + 1, 0.0, mr); a_new[0] = 1.0;
        for (int j = 1; j < i; ++j)
            a_new[j] = a_prev[j] + ki * a_prev[i - j];
        a_new[i] = ki;
        a_prev = std::move(a_new);
        e *= (1.0 - ki * ki);
    }
    return colVec(k, mr);
}

Value rc2ac(const Value &k, double r0, keel::Allocator *mr)
{
    auto kv = readVec(k);
    const int p = static_cast<int>(kv.size());
    // Build R from k via the inverse Levinson: r0 known, then
    // r[i] = -(a_{i-1}[1]*r[i-1] + ... + a_{i-1}[i-1]*r[1]) - k[i-1]*e_{i-1}
    keel::Array<double> R(p + 1, 0.0, mr);
    R[0] = r0;
    keel::Array<double> a(p + 1, 0.0, mr); a[0] = 1.0;
    double e = r0;
    for (int i = 1; i <= p; ++i) {
        const double ki = kv[i - 1];
        double s = 0.0;
        for (int j = 1; j < i; ++j) s += a[j] * R[i - j];
        R[i] = -ki * e - s;
        // Update a by step-up.
        keel::Array<double> a_new(p + 1, 0.0, mr);
        a_new[0] = 1.0;
        for (int j = 1; j < i; ++j) a_new[j] = a[j] + ki * a[i - j];
        a_new[i] = ki;
        a = std::move(a_new);
        e *= (1.0 - ki * ki);
    }
    return rowVec(R, mr);
}

std::tuple<Value, Value>
poly2rc(const Value &a, double efinal, keel::Allocator *mr)
{
    auto av = readVec(a);
    const int p = static_cast<int>(av.size()) - 1;
    keel::Array<double> k(p, 0.0, mr);
    keel::Array<double> cur = av.clone();
    for (int i = p; i >= 1; --i) {
        const double ki = cur[i];
        k[i - 1] = ki;
        const double denom = 1.0 - ki * ki;
        if (keel::abs(denom) < 1e-300) break;
        keel::Array<double> prev(i + 1, 0.0, mr); prev[0] = 1.0;
        for (int j = 1; j < i; ++j)
            prev[j] = (cur[j] - ki * cur[i - j]) / denom;
        cur = std::move(prev);
    }
    // Zero-lag autocorrelation R0 = efinal / prod_i (1 - k_i^2). This is
    // the second output [k, R0] = poly2rc(a, efinal): the energy of the
    // signal whose AR model is `a` with final prediction error `efinal`.
    double prod = 1.0;
    for (double ki : k) prod *= (1.0 - ki * ki);
    const double r0 = (prod != 0.0) ? efinal / prod : 0.0;
    return std::make_tuple(colVec(k, mr), Value::scalar(r0, mr));
}

std::tuple<Value, Value>
rc2poly(const Value &k, double r0, keel::Allocator *mr)
{
    auto kv = readVec(k);
    const int p = static_cast<int>(kv.size());
    keel::Array<double> a(p + 1, 0.0, mr); a[0] = 1.0;
    for (int i = 1; i <= p; ++i) {
        const double ki = kv[i - 1];
        keel::Array<double> a_new(p + 1, 0.0, mr);
        a_new[0] = 1.0;
        for (int j = 1; j < i; ++j) a_new[j] = a[j] + ki * a[i - j];
        a_new[i] = ki;
        a = std::move(a_new);
    }
    // efinal = r0 * prod_i (1 - k_i^2): the final prediction error of the
    // AR model, the inverse of poly2rc's R0 relation. Second output of
    // [a, efinal] = rc2poly(k, r0).
    double prod = 1.0;
    for (double ki : kv) prod *= (1.0 - ki * ki);
    return std::make_tuple(rowVec(a, mr), Value::scalar(r0 * prod, mr));
}

// is2rc / rc2is — inverse-sine parameterisation. MATLAB scales the inverse
// sine into [-1, 1]: isin = (2/pi)*asin(k), so k = sin((pi/2)*isin).
Value is2rc(const Value &is, keel::Allocator *mr)
{
    constexpr double kHalfPi = 1.5707963267948966;  // pi/2
    auto v = readVec(is);
    for (auto &y : v) y = std::sin(kHalfPi * y);
    return colVec(v, mr);
}
Value rc2is(const Value &k, keel::Allocator *mr)
{
    constexpr double kTwoOverPi = 0.6366197723675814;  // 2/pi
    auto v = readVec(k);
    for (auto &y : v) {
        // Clamp into [-1, 1] to avoid asin domain errors on numerical
        // overshoots.
        const double c = std::max(-1.0, std::min(1.0, y));
        y = kTwoOverPi * std::asin(c);
    }
    return colVec(v, mr);
}

// lar2rc / rc2lar — log-area-ratio: lar = log((1+k)/(1-k)).
Value lar2rc(const Value &g, keel::Allocator *mr)
{
    auto v = readVec(g);
    for (auto &y : v) {
        const double e = std::exp(y);
        y = (e - 1.0) / (e + 1.0);
    }
    return colVec(v, mr);
}
Value rc2lar(const Value &k, keel::Allocator *mr)
{
    auto v = readVec(k);
    for (auto &y : v) {
        const double c = std::max(-0.99999999, std::min(0.99999999, y));
        y = std::log((1.0 + c) / (1.0 - c));
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
keel::Array<double>
solveSPD(keel::Array<double> &A, keel::Array<double> &b, int p)
{
    for (int k = 0; k < p; ++k) {
        // Pivot.
        int piv = k;
        double pivVal = keel::abs(A[k * p + k]);
        for (int i = k + 1; i < p; ++i)
            if (keel::abs(A[i * p + k]) > pivVal) { pivVal = keel::abs(A[i * p + k]); piv = i; }
        if (pivVal < 1e-300) continue;
        if (piv != k) {
            for (int j = k; j < p; ++j)
                std::swap(A[k * p + j], A[piv * p + j]);
            std::swap(b[k], b[piv]);
        }
        const double diag = A[k * p + k];
        for (int i = k + 1; i < p; ++i) {
            const double f = A[i * p + k] / diag;
            for (int j = k; j < p; ++j) A[i * p + j] -= f * A[k * p + j];
            b[i] -= f * b[k];
        }
    }
    keel::Array<double> x(p, 0.0);
    for (int i = p - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < p; ++j) s -= A[i * p + j] * x[j];
        const double diag = A[i * p + i];
        x[i] = (keel::abs(diag) > 1e-300) ? s / diag : 0.0;
    }
    return x;
}

// Build the covariance correlation R_{ij} = sum_{n=p..N-1} x[n-i] * x[n-j]
// for i, j in 0..p. Used by arcov / armcov.
keel::Array<double>
covMatrixForward(keel::ArrayRef<double> x, int p)
{
    const int N = static_cast<int>(x.size());
    keel::Array<double> R(static_cast<keel::usize>(p + 1) * static_cast<keel::usize>(p + 1), 0.0);
    for (int n = p; n < N; ++n) {
        for (int i = 0; i <= p; ++i)
            for (int j = 0; j <= p; ++j)
                R[i * (p + 1) + j] += x[n - i] * x[n - j];
    }
    return R;
}

keel::Array<double>
covMatrixForwardBackward(keel::ArrayRef<double> x, int p)
{
    const int N = static_cast<int>(x.size());
    auto R = covMatrixForward(x, p);
    // Backward: R += sum_{n=0..N-1-p} x[n+i] x[n+j]
    for (int n = 0; n + p < N; ++n) {
        for (int i = 0; i <= p; ++i)
            for (int j = 0; j <= p; ++j)
                R[i * (p + 1) + j] += x[n + i] * x[n + j];
    }
    return R;
}

// Given a (p+1)×(p+1) covariance matrix R indexed R[i,j] for i,j in
// 0..p, solve for AR coefficients a[1..p] from R[1..p, 1..p] · a =
// -R[1..p, 0]. Returns full a vector with a[0] = 1 prepended, plus
// the prediction error e = R[0,0] + sum_k a[k] * R[0,k].
struct ArCovResult { keel::Array<double> a; double e; };
ArCovResult solveArCov(keel::ArrayRef<double> R, int p)
{
    keel::Array<double> A(static_cast<keel::usize>(p) * static_cast<keel::usize>(p), 0.0);
    keel::Array<double> b(p, 0.0);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) A[i * p + j] = R[(i + 1) * (p + 1) + (j + 1)];
        b[i] = -R[(i + 1) * (p + 1) + 0];
    }
    auto a_tail = solveSPD(A, b, p);
    keel::Array<double> a(p + 1, 0.0);
    a[0] = 1.0;
    for (int k = 0; k < p; ++k) a[k + 1] = a_tail[k];
    double e = R[0];
    for (int k = 1; k <= p; ++k) e += a[k] * R[k];
    return {std::move(a), e};
}

} // anonymous

std::tuple<Value, Value>
arcov(const Value &x, int p, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto R = covMatrixForward(v, p);
    // Normalise by the count of summed terms (N - p) so e has the
    // same units as the residual variance.
    const int N = static_cast<int>(v.size());
    const double norm = static_cast<double>(std::max(N - p, 1));
    for (auto &r : R) r /= norm;
    auto res = solveArCov(R, p);
    return std::make_tuple(rowVec(res.a, mr), Value::scalar(res.e, mr));
}

std::tuple<Value, Value>
armcov(const Value &x, int p, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto R = covMatrixForwardBackward(v, p);
    const int N = static_cast<int>(v.size());
    const double norm = static_cast<double>(std::max(2 * (N - p), 1));
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
prony(const Value &h, int nb, int na, keel::Allocator *mr)
{
    auto hv = readVec(h);
    const int N = static_cast<int>(hv.size());
    if (na > 0) {
        // Build linear system M · a_tail = -rhs, where M is (N-1-nb) × na
        // and rhs is length (N-1-nb). M[i, j] = h[nb+1+i-1-j] (padded
        // with zeros on the left), rhs[i] = h[nb+1+i].
        const int rows = std::max(N - 1 - nb, 1);
        keel::Array<double> M(static_cast<keel::usize>(rows) * static_cast<keel::usize>(na), 0.0, mr);
        keel::Array<double> rhs(rows, 0.0, mr);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < na; ++j) {
                const int idx = nb + i - j;  // h index = nb+1+i - (j+1) = nb+i-j
                M[i * na + j] = (idx >= 0 && idx < N) ? hv[idx] : 0.0;
            }
            const int rhs_idx = nb + 1 + i;
            rhs[i] = (rhs_idx < N) ? -hv[rhs_idx] : 0.0;
        }
        // Normal equations: (M^T M) · a_tail = M^T rhs
        keel::Array<double> MtM(static_cast<keel::usize>(na) * static_cast<keel::usize>(na), 0.0, mr);
        keel::Array<double> Mtb(na, 0.0, mr);
        for (int i = 0; i < na; ++i) {
            for (int j = 0; j < na; ++j) {
                double s = 0.0;
                for (int k = 0; k < rows; ++k) s += M[k * na + i] * M[k * na + j];
                MtM[i * na + j] = s;
            }
            double s = 0.0;
            for (int k = 0; k < rows; ++k) s += M[k * na + i] * rhs[k];
            Mtb[i] = s;
        }
        auto a_tail = solveSPD(MtM, Mtb, na);
        keel::Array<double> a(na + 1, 0.0, mr);
        a[0] = 1.0;
        for (int k = 0; k < na; ++k) a[k + 1] = a_tail[k];

        // b = (a * h) truncated to nb+1 terms.
        keel::Array<double> b(nb + 1, 0.0, mr);
        for (int i = 0; i <= nb; ++i) {
            double s = 0.0;
            for (int k = 0; k <= std::min(i, na); ++k)
                if (i - k < N) s += a[k] * hv[i - k];
            b[i] = s;
        }
        return std::make_tuple(rowVec(b, mr), rowVec(a, mr));
    }
    // na == 0: pure FIR. b[i] = h[i] for i = 0..nb, a = [1].
    keel::Array<double> b(nb + 1, 0.0, mr);
    for (int i = 0; i <= nb && i < N; ++i) b[i] = hv[i];
    const double one = 1.0;
    return std::make_tuple(rowVec(b, mr), rowVec(keel::ArrayRef<double>(&one, 1), mr));
}

// ── stmcb ─────────────────────────────────────────────────────────
// Steiglitz-McBride iteration: identify B(z)/A(z) (nb zeros, na poles) with
// approximate impulse response h. Initialise A via Prony, then repeat `niter`
// times (MATLAB default 5): prefilter the unit impulse and h by 1/A, solve the
// linear least-squares [E | -G]·[b; a_tail] ≈ g for the new coefficients.
namespace {

void stmcb_argcheck(const Value &y, const Value &x, int nb, int na, const Value &ai)
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
        if (n_ai != static_cast<keel::usize>(na + 1) && n_ai != static_cast<keel::usize>(na))
            throw Error("stmcb: initial denominator estimate ai must have length na+1 (or na)",
                        0, 0, "stmcb", "", "numkit:stmcb:InvalidAiLength");
        if (n_ai == static_cast<keel::usize>(na + 1) && ai.elemAsDouble(0) == 0.0)
            throw Error("stmcb: leading coefficient of ai must be nonzero",
                        0, 0, "stmcb", "", "numkit:stmcb:ZeroLeadingCoefficient");
    }
}

} // anonymous

std::tuple<Value, Value>
stmcb(const Value &y, const Value &x, int nb, int na, int niter,
      const Value &ai, keel::Allocator *mr)
{
    stmcb_argcheck(y, x, nb, na, ai);
    auto      yv = readVec(y);
    const int N  = static_cast<int>(yv.size());
    if (niter < 1) niter = 5;

    keel::Array<double> xv(mr);
    if (!x.isEmpty()) {
        xv = readVec(x);
    } else {
        xv.assign(N, 0.0);
        if (N > 0) xv[0] = 1.0;
    }

    keel::Array<double> b(nb + 1, 0.0, mr);
    keel::Array<double> a(na + 1, 0.0, mr);
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
        if (static_cast<int>(aiv.size()) == na + 1) {
            const double a0_val = aiv[0];
            for (int k = 0; k <= na; ++k) a[k] = aiv[k] / a0_val;
            a[0] = 1.0;
        } else {
            a[0] = 1.0;
            for (int k = 0; k < na; ++k) a[k + 1] = aiv[k];
        }
    }

    const int           P = nb + 1 + na;   // unknowns: b[0..nb] then a[1..na]
    keel::Array<double> e(N, 0.0, mr), g(static_cast<keel::usize>(N), 0.0, mr);
    for (int it = 0; it < niter; ++it) {
        // Prefilter the input x and output y by 1/A.
        for (int n = 0; n < N; ++n) {
            double    se  = xv[n];
            double    sg  = yv[n];
            const int lim = std::min(na, n);
            for (int i = 1; i <= lim; ++i) { se -= a[i] * e[n - i]; sg -= a[i] * g[n - i]; }
            e[n] = se;
            g[n] = sg;
        }
        // [E | -G]·c ≈ g  (E: e shifted 0..nb, G: g shifted 1..na), normal eqs.
        auto Mcol = [&](int n, int col) -> double {
            if (col <= nb) { const int s = col; return (n - s >= 0) ? e[n - s] : 0.0; }
            const int s = col - nb;             // 1..na
            return (n - s >= 0) ? -g[n - s] : 0.0;
        };
        keel::Array<double> MtM(static_cast<keel::usize>(P) * P, 0.0, mr), Mtg(static_cast<keel::usize>(P), 0.0, mr);
        for (int n = 0; n < N; ++n)
            for (int i = 0; i < P; ++i) {
                const double mi = Mcol(n, i);
                if (mi == 0.0) continue;
                Mtg[i] += mi * g[n];
                for (int j = 0; j < P; ++j) MtM[i * P + j] += mi * Mcol(n, j);
            }
        auto c = solveSPD(MtM, Mtg, P);
        for (int k = 0; k <= nb; ++k) b[k]     = c[k];
        for (int k = 0; k < na;  ++k) a[k + 1] = c[nb + 1 + k];
    }
    return std::make_tuple(rowVec(b, mr), rowVec(a, mr));
}

std::tuple<Value, Value>
stmcb(const Value &h, int nb, int na, int niter, keel::Allocator *mr)
{
    return stmcb(h, Value(), nb, na, niter, Value(), mr);
}

// ── corrmtx ───────────────────────────────────────────────────────
// MATLAB default 'autocorrelation' method: produces the (n+m)×(m+1)
// data matrix X with X(i, j) = x(i-j+1), zero-padded outside the
// input range, such that X' * X is the (m+1)×(m+1) autocorrelation
// matrix.
Value corrmtx(const Value &x, int m, keel::Allocator *mr)
{
    auto v = readVec(x);
    const int N = static_cast<int>(v.size());
    const int rows = N + m;
    const int cols = m + 1;
    auto X = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    double *d = X.doubleDataMut();
    std::fill(d, d + rows * cols, 0.0);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const int idx = i - j;     // x[i-j], zero outside [0, N)
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

keel::Array<double> reverseVec(keel::ArrayRef<double> v)
{
    keel::Array<double> r(v.size());
    for (keel::usize i = 0; i < v.size(); ++i) r[i] = v[v.size() - 1 - i];
    return r;
}

// Pad poly p (descending power) to total length n with leading zeros.
[[maybe_unused]] keel::Array<double> padLeft(keel::ArrayRef<double> p, int n)
{
    keel::Array<double> r(n, 0.0);
    const int off = n - static_cast<int>(p.size());
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
    const int N = static_cast<int>(av.size()) - 1;
    if (N <= 0) return colVec(keel::ArrayRef<double>{}, mr);

    // Build P = A + A_R_shifted, Q = A - A_R_shifted, both length N+2.
    auto ar = reverseVec(av);
    keel::Array<double> P(N + 2, 0.0, mr), Q(static_cast<keel::usize>(N + 2), 0.0, mr);
    for (int i = 0; i <= N; ++i) {
        P[i] = av[i];
        Q[i] = av[i];
    }
    for (int i = 1; i <= N + 1; ++i) {
        P[i] += ar[i - 1];
        Q[i] -= ar[i - 1];
    }
    // Find roots and pick angles in (0, π).
    keel::Array<double> angles(mr);
    const keel::ArrayRef<double> polys[2] = {P.as_ref(), Q.as_ref()};
    for (auto poly : polys) {
        Value polyV = rowVec(poly, mr);
        Value rts = numkit::builtin::roots(polyV, mr);
        const keel::usize n = rts.numel();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::c64 c = rts.complexData()[i];
            const double a_ang = std::atan2(c.imag(), c.real());
            if (a_ang > 1e-9 && a_ang < M_PI - 1e-9) angles.push_back(a_ang);
        }
    }
    std::sort(angles.begin(), angles.end());
    // Should yield N angles.
    if (static_cast<int>(angles.size()) > N) angles.resize(N);
    return colVec(angles, mr);
}

Value lsf2poly(const Value &lsf, keel::Allocator *mr)
{
    auto lv = readVec(lsf);
    const int m = static_cast<int>(lv.size());
    if (m == 0) {
        const double one = 1.0;
        return rowVec(keel::ArrayRef<double>(&one, 1), mr);
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
    auto convOne = [mr](keel::ArrayRef<double> a,
                        keel::ArrayRef<double> b) {
        keel::Array<double> r(a.size() + b.size() - 1, 0.0, mr);
        for (keel::usize i = 0; i < a.size(); ++i)
            for (keel::usize j = 0; j < b.size(); ++j)
                r[i + j] += a[i] * b[j];
        return r;
    };

    keel::Array<double> P(mr), Q(mr);
    if (m % 2 == 1) {
        // m odd: P has no boundary roots, Q gets both (-1 and +1) =
        // (1-z^-2). Both end up degree m+1.
        P = keel::Array<double>({1.0}, mr);
        Q = keel::Array<double>({1.0, 0.0, -1.0}, mr);
    } else {
        // m even: P gets z=-1 → (1+z^-1); Q gets z=+1 → (1-z^-1).
        // Both end up degree m+1.
        P = keel::Array<double>({1.0,  1.0}, mr);
        Q = keel::Array<double>({1.0, -1.0}, mr);
    }
    for (int i = 0; i < m; ++i) {
        const double q[3] = {1.0, -2.0 * std::cos(lv[i]), 1.0};
        if (i % 2 == 0) P = convOne(P, keel::ArrayRef<double>(q, 3));
        else            Q = convOne(Q, keel::ArrayRef<double>(q, 3));
    }

    // P and Q have identical length now. a = (P + Q) / 2, then drop
    // the trailing element (it cancels to ~0 by construction).
    keel::Array<double> a(P.size(), 0.0, mr);
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
std::tuple<keel::Array<double>, keel::Array<double>>
invfreqLSQ(keel::ArrayRef<keel::c64> H,
           keel::ArrayRef<keel::c64> x,
           int nb, int na)
{
    const int K  = static_cast<int>(H.size());
    const int nv = nb + 1 + na;
    if (K * 2 < nv)
        throw std::runtime_error("invfreq: not enough frequency points to fit (nb, na)");

    // Build A (2K × nv) row-major, plus rhs (length 2K).
    keel::Array<double> Ar(static_cast<keel::usize>(2 * K) * static_cast<keel::usize>(nv), 0.0);
    keel::Array<double> rhs(2 * K, 0.0);
    for (int k = 0; k < K; ++k) {
        // Build x^0..x^max(nb, na) once.
        const int M = std::max(nb, na);
        keel::Array<keel::c64> xpow(M + 1);
        xpow[0] = keel::c64(1.0, 0.0);
        for (int i = 1; i <= M; ++i) xpow[i] = xpow[i - 1] * x[k];

        // Row k contains:
        //   -1, -x, -x², …, -x^nb, H·x, H·x², …, H·x^na
        keel::Array<keel::c64> row(nv);
        for (int i = 0; i <= nb; ++i) row[i] = -xpow[i];
        for (int j = 1; j <= na; ++j) row[nb + j] = H[k] * xpow[j];

        // Stack real / imag into Ar.
        for (int c = 0; c < nv; ++c) {
            Ar[(2 * k    ) * nv + c] = row[c].real();
            Ar[(2 * k + 1) * nv + c] = row[c].imag();
        }
        rhs[2 * k    ] = -H[k].real();
        rhs[2 * k + 1] = -H[k].imag();
    }

    // Normal equations: (Aᵀ A) θ = Aᵀ rhs.
    keel::Array<double> AtA(static_cast<keel::usize>(nv) * static_cast<keel::usize>(nv), 0.0);
    keel::Array<double> Atb(nv, 0.0);
    for (int i = 0; i < nv; ++i) {
        for (int j = 0; j < nv; ++j) {
            double s = 0.0;
            for (int r = 0; r < 2 * K; ++r) s += Ar[r * nv + i] * Ar[r * nv + j];
            AtA[i * nv + j] = s;
        }
        double s = 0.0;
        for (int r = 0; r < 2 * K; ++r) s += Ar[r * nv + i] * rhs[r];
        Atb[i] = s;
    }
    auto theta = solveSPD(AtA, Atb, nv);

    // theta = [b_0, ..., b_nb, a_1, ..., a_na]
    keel::Array<double> b(nb + 1), a(static_cast<keel::usize>(na + 1));
    for (int i = 0; i <= nb; ++i) b[i] = theta[i];
    a[0] = 1.0;
    for (int j = 1; j <= na; ++j) a[j] = theta[nb + j];
    return std::make_tuple(std::move(b), std::move(a));
}

void invfreqs_argcheck(const Value &H, const Value &w, int nb, int na)
{
    if (H.numel() != w.numel())
        throw std::runtime_error("invfreqs: H and w must be the same length");
    const int K  = static_cast<int>(H.numel());
    const int nv = nb + 1 + na;
    if (K * 2 < nv)
        throw std::runtime_error("invfreq: not enough frequency points to fit (nb, na)");
}

void invfreqz_argcheck(const Value &H, const Value &w, int nb, int na)
{
    if (H.numel() != w.numel())
        throw std::runtime_error("invfreqz: H and w must be the same length");
    const int K  = static_cast<int>(H.numel());
    const int nv = nb + 1 + na;
    if (K * 2 < nv)
        throw std::runtime_error("invfreq: not enough frequency points to fit (nb, na)");
}

} // anonymous

std::tuple<Value, Value>
invfreqs(const Value &H, const Value &w, int nb, int na, keel::Allocator *mr)
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
invfreqz(const Value &H, const Value &w, int nb, int na, keel::Allocator *mr)
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
