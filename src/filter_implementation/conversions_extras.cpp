// toolboxes/signal/src/filter_implementation/conversions_extras.cpp
// sos2tf / sos2zp / tf2zpk + tf↔ss + sos↔ss + zpk↔ss adapters.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/filter_implementation/conversions_extras.hpp>

#include <numkit/signal/internal/shims/polyfun.hpp>      // tf2zp / zp2tf / roots
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <tuple>
#include <numkit/signal/filter_implementation/conversions.hpp>  // tf2sos

#include <algorithm>

namespace numkit::signal {

namespace {

// Trim trailing zeros from a polynomial coefficient vector. Returns
// the kept length (1 = constant, 0 = empty input).
keel::usize trimTrailingZeros(keel::Array<keel::f64> &v)
{
    while (v.size() > 1 && keel::abs(v.back()) < 1e-15) v.pop_back();
    return v.size();
}

// Convolve two real polynomials (coefficient-wise multiplication).
keel::Array<keel::f64> polyConv(keel::ArrayRef<keel::f64> a, keel::ArrayRef<keel::f64> b)
{
    if (a.empty() || b.empty()) return {};
    keel::Array<keel::f64> out(a.size() + b.size() - 1, 0.0);
    for (keel::usize i = 0; i < a.size(); ++i)
        for (keel::usize j = 0; j < b.size(); ++j)
            out[i + j] += a[i] * b[j];
    return out;
}

Value rowFromVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < v.size(); ++i) dst[i] = v[i];
    return out;
}

// Append row's roots (as Complex) to the running list.
void appendRoots(const keel::f64 *coeffs, keel::usize n, keel::Array<keel::c64> &out, keel::Allocator *mr)
{
    // Build a 1×n DOUBLE Value to feed roots(). Trim trailing zeros so
    // roots() doesn't add spurious roots at 0.
    keel::Array<keel::f64> trimmed(keel::ArrayRef<keel::f64>(coeffs, n), mr);
    trimTrailingZeros(trimmed);
    if (trimmed.size() < 2) return;
    auto p = Value::matrix(1, trimmed.size(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < trimmed.size(); ++i)
        p.doubleDataMut()[i] = trimmed[i];
    auto r = numkit::builtin::roots(p, mr);
    if (r.isComplex()) {
        const keel::c64 *src = r.complexData();
        for (keel::usize i = 0; i < r.numel(); ++i) out.push_back(src[i]);
    } else {
        const keel::f64 *src = r.doubleData();
        for (keel::usize i = 0; i < r.numel(); ++i) out.emplace_back(src[i], 0.0);
    }
}

Value complexColFromVec(keel::ArrayRef<keel::c64> v, keel::Allocator *mr)
{
    bool anyImag = false;
    for (const auto &c : v)
        if (keel::abs(c.imag()) > 1e-12) { anyImag = true; break; }
    if (anyImag) {
        auto out = Value::complexMatrix(v.size(), 1, mr);
        keel::c64 *dst = out.complexDataMut();
        for (keel::usize i = 0; i < v.size(); ++i) dst[i] = v[i];
        return out;
    }
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < v.size(); ++i) dst[i] = v[i].real();
    return out;
}

void requireSosShape(const Value &sos, const char *fn)
{
    if (sos.dims().cols() != 6 || sos.dims().rows() == 0)
        throw Error(keel::format("{}: sos must be an L×6 matrix", fn),
                     0, 0, fn, "", keel::format("numkit:{}:badShape", fn));
}

void sos2tf_argcheck(const Value &sos)
{
    requireSosShape(sos, "sos2tf");
}

void sos2zp_argcheck(const Value &sos)
{
    requireSosShape(sos, "sos2zp");
}

} // namespace

// ── sos2tf ────────────────────────────────────────────────────────────
std::tuple<Value, Value>
sos2tf(const Value &sos, keel::f64 g, keel::Allocator *mr)
{
    sos2tf_argcheck(sos);
    const keel::usize L = sos.dims().rows();
    keel::Array<keel::f64> b({1.0}, mr), a({1.0}, mr);
    for (keel::usize i = 0; i < L; ++i) {
        const keel::f64 bs[3] = {sos(i, 0), sos(i, 1), sos(i, 2)};
        const keel::f64 as[3] = {sos(i, 3), sos(i, 4), sos(i, 5)};
        b = polyConv(b, keel::ArrayRef<keel::f64>(bs, 3));
        a = polyConv(a, keel::ArrayRef<keel::f64>(as, 3));
    }
    if (g != 1.0)
        for (auto &v : b) v *= g;
    return std::make_tuple(rowFromVec(b, mr), rowFromVec(a, mr));
}

// ── sos2zp ────────────────────────────────────────────────────────────
std::tuple<Value, Value, keel::f64>
sos2zp(const Value &sos, keel::f64 g, keel::Allocator *mr)
{
    sos2zp_argcheck(sos);
    const keel::usize L = sos.dims().rows();
    keel::Array<keel::c64> zeros(mr), poles(mr);
    keel::f64 gain = g;
    for (keel::usize i = 0; i < L; ++i) {
        const keel::f64 bArr[3] = {sos(i, 0), sos(i, 1), sos(i, 2)};
        const keel::f64 aArr[3] = {sos(i, 3), sos(i, 4), sos(i, 5)};
        appendRoots(bArr, 3, zeros, mr);
        appendRoots(aArr, 3, poles, mr);
        // Per-section gain factor: b0 / a0 for that section.
        if (keel::abs(aArr[0]) > 1e-15)
            gain *= bArr[0] / aArr[0];
    }
    return std::make_tuple(complexColFromVec(zeros, mr),
                            complexColFromVec(poles, mr),
                            gain);
}

// ── tf2zpk ────────────────────────────────────────────────────────────
std::tuple<Value, Value, keel::f64>
tf2zpk(const Value &b, const Value &a, keel::Allocator *mr)
{
    auto [z, p, k] = numkit::builtin::tf2zp(b, a, mr);
    return std::make_tuple(std::move(z), std::move(p),
                            (k.numel() == 0) ? 0.0 : k.toScalar());
}

// ── tf2ss (controllable canonical form) ───────────────────────────────
std::tuple<Value, Value, Value, Value>
tf2ss(const Value &b, const Value &a, keel::Allocator *mr)
{
    if (a.numel() == 0)
        throw Error("tf2ss: denominator a must be non-empty",
                     0, 0, "tf2ss", "", "numkit:tf2ss:badArg");
    const keel::f64 a0 = a.elemAsDouble(0);
    if (keel::abs(a0) < 1e-15)
        throw Error("tf2ss: a(1) must be non-zero",
                     0, 0, "tf2ss", "", "numkit:tf2ss:badArg");

    // Normalise so a(1) = 1.
    const keel::usize na = a.numel();
    keel::Array<keel::f64> ah(na, mr);
    for (keel::usize i = 0; i < na; ++i) ah[i] = a.elemAsDouble(i) / a0;

    // Pad b to the same length as a (left-pad with zeros — matches
    // MATLAB convention for proper rationals).
    const keel::usize nb = b.numel();
    keel::Array<keel::f64> bh(na, 0.0, mr);
    if (nb >= na) {
        for (keel::usize i = 0; i < na; ++i)
            bh[i] = b.elemAsDouble(nb - na + i) / a0;
    } else {
        for (keel::usize i = 0; i < nb; ++i)
            bh[na - nb + i] = b.elemAsDouble(i) / a0;
    }

    // Denominator order N. State dimension = N.
    if (na < 2)
        throw Error("tf2ss: order must be >= 1 (denominator length >= 2)",
                     0, 0, "tf2ss", "", "numkit:tf2ss:badArg");
    const keel::usize N = na - 1;

    // Direct feed-through D = bh[0]   (since b is normalised + padded
    // to length N+1, this is the leading coeff).
    const keel::f64 D = bh[0];

    // MATLAB tf2ss returns the controller canonical form:
    //   A = [-a2 -a3 ... -a(N+1);
    //         1   0  ...  0;
    //         0   1  ...  0;
    //         ...
    //         0   0  ...  0]    (companion in TOP row)
    //   B = [1; 0; 0; ...; 0]
    //   C = [b2 - a2*b1, b3 - a3*b1, ..., b(N+1) - a(N+1)*b1]
    //   D = b1
    // Here ah[0]=1 (post-normalisation) and ah[1..N] are -a2..-a(N+1)
    // when negated.
    auto AVal = Value::matrix(N, N, ValueType::DOUBLE, mr);
    keel::f64 *Ap = AVal.doubleDataMut();
    std::fill(Ap, Ap + N * N, 0.0);
    // Top row (column-major: Ap[r + c*N]).
    for (keel::usize j = 0; j < N; ++j)
        Ap[0 + j * N] = -ah[j + 1];
    // Identity sub-diagonal: A(i+1, i) = 1 for i = 0..N-2.
    for (keel::usize i = 0; i + 1 < N; ++i)
        Ap[(i + 1) + i * N] = 1.0;

    // B: column vector with 1 in FIRST entry.
    auto BVal = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *Bp = BVal.doubleDataMut();
    std::fill(Bp, Bp + N, 0.0);
    Bp[0] = 1.0;

    // C: 1xN row, C(j) = bh[j+1] - ah[j+1]*D for j = 0..N-1.
    auto CVal = Value::matrix(1, N, ValueType::DOUBLE, mr);
    keel::f64 *Cp = CVal.doubleDataMut();
    for (keel::usize j = 0; j < N; ++j)
        Cp[j] = bh[j + 1] - ah[j + 1] * D;

    auto DVal = Value::scalar(D, mr);
    return std::make_tuple(std::move(AVal), std::move(BVal),
                            std::move(CVal), std::move(DVal));
}

// ── ss2tf ─────────────────────────────────────────────────────────────
// SISO. Returns (b, a) where:
//   a = char poly of A   (Faddeev–LeVerrier or characteristic polynomial)
//   b = numerator built from the resolvent (a*D + C * adj(zI - A) * B)
// This is heavy linear algebra; use Faddeev's algorithm.
std::tuple<Value, Value>
ss2tf(const Value &A, const Value &B, const Value &C, keel::f64 D,
      keel::Allocator *mr)
{
    if (A.dims().rows() != A.dims().cols())
        throw Error("ss2tf: A must be square",
                     0, 0, "ss2tf", "", "numkit:ss2tf:badShape");
    const keel::usize N = A.dims().rows();
    if (B.dims().rows() != N || B.dims().cols() != 1)
        throw Error("ss2tf: B must be N×1",
                     0, 0, "ss2tf", "", "numkit:ss2tf:badShape");
    if (C.dims().rows() != 1 || C.dims().cols() != N)
        throw Error("ss2tf: C must be 1×N",
                     0, 0, "ss2tf", "", "numkit:ss2tf:badShape");

    const keel::f64 Dscalar = D;

    // Faddeev–LeVerrier:
    //   M_0 = I,   c_n = 1
    //   M_k = A * M_{k-1} + c_{n-k+1} * I
    //   c_{n-k} = -trace(A * M_k) / k
    // The polynomial coefficients c_0..c_n satisfy
    //   p(λ) = c_0 + c_1 λ + ... + c_n λ^n   (with c_n = 1).
    // We also accumulate b(λ) = D * a(λ) + Σ C * M_k * B * λ^k.

    auto matMul = [N](keel::ArrayRef<keel::f64> X,
                      keel::ArrayRef<keel::f64> Y,
                      keel::Array<keel::f64> &out) {
        out.assign(N * N, 0.0);
        for (keel::usize i = 0; i < N; ++i)
            for (keel::usize j = 0; j < N; ++j) {
                keel::f64 s = 0.0;
                for (keel::usize k = 0; k < N; ++k)
                    s += X[k * N + i] * Y[j * N + k];   // column-major
                out[j * N + i] = s;
            }
    };
    auto matVec = [N](keel::ArrayRef<keel::f64> X,
                      const keel::f64 *y,
                      keel::Array<keel::f64> &out) {
        out.assign(N, 0.0);
        for (keel::usize i = 0; i < N; ++i)
            for (keel::usize k = 0; k < N; ++k)
                out[i] += X[k * N + i] * y[k];
    };

    keel::Array<keel::f64> A_(N * N, mr);
    for (keel::usize i = 0; i < N; ++i)
        for (keel::usize j = 0; j < N; ++j)
            A_[j * N + i] = A(i, j);
    const keel::f64 *Bp = B.doubleData();
    const keel::f64 *Cp = C.doubleData();

    // Faddeev–LeVerrier driving the characteristic polynomial AND
    // adj(sI - A) = sum_{k=0}^{N-1} M_k s^{N-1-k}.
    // a_coeffs are stored from s^0 (index 0) up to s^N (index N);
    // adjoint coefficients C·M_k·B are accumulated into CMB[k] for
    // k=0..N-1 and then placed into b_coeffs[N-1-k]:
    //   b_coeffs[N]       = D
    //   b_coeffs[N-1-k]   = D · a_coeffs[N-1-k] + (C · M_k · B)
    keel::Array<keel::f64> M(N * N, 0.0, mr), AM(N * N, mr);
    for (keel::usize i = 0; i < N; ++i) M[i * N + i] = 1.0;   // M_0 = I

    keel::Array<keel::f64> a_coeffs(N + 1, 0.0, mr);
    a_coeffs[N] = 1.0;
    keel::Array<keel::f64> CMB(N, 0.0, mr);

    keel::Array<keel::f64> tmp(N, mr);
    auto dotCMB = [&](keel::ArrayRef<keel::f64> Mat) {
        matVec(Mat, Bp, tmp);
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < N; ++i) s += Cp[i] * tmp[i];
        return s;
    };
    CMB[0] = dotCMB(M);   // C * M_0 * B = C * B

    for (keel::usize k = 1; k <= N; ++k) {
        matMul(A_, M, AM);
        keel::f64 trace = 0.0;
        for (keel::usize i = 0; i < N; ++i) trace += AM[i * N + i];
        const keel::f64 ck = -trace / keel::as<keel::f64>(k);
        a_coeffs[N - k] = ck;
        // M_k = AM + ck * I
        for (keel::usize i = 0; i < N * N; ++i) M[i] = AM[i];
        for (keel::usize i = 0; i < N; ++i) M[i * N + i] += ck;
        // Capture C·M_k·B for k=1..N-1 (M_N is unused for adjoint).
        if (k < N) CMB[k] = dotCMB(M);
    }

    keel::Array<keel::f64> b_coeffs(N + 1, 0.0, mr);
    b_coeffs[N] = Dscalar;
    for (keel::usize k = 0; k < N; ++k)
        b_coeffs[N - 1 - k] = Dscalar * a_coeffs[N - 1 - k] + CMB[k];

    // Reverse to MATLAB ordering (highest-power first).
    std::reverse(a_coeffs.begin(), a_coeffs.end());
    std::reverse(b_coeffs.begin(), b_coeffs.end());

    return std::make_tuple(rowFromVec(b_coeffs, mr), rowFromVec(a_coeffs, mr));
}

// ── ss2zp ─────────────────────────────────────────────────────────────
std::tuple<Value, Value, keel::f64>
ss2zp(const Value &A, const Value &B, const Value &C, keel::f64 D,
      keel::Allocator *mr)
{
    auto [b, a] = ss2tf(A, B, C, D, mr);
    return tf2zpk(b, a, mr);
}

// ── zp2ss ─────────────────────────────────────────────────────────────
std::tuple<Value, Value, Value, Value>
zp2ss(const Value &z, const Value &p, keel::f64 k, keel::Allocator *mr)
{
    auto [b, a] = numkit::builtin::zp2tf(z, p, k, mr);
    return tf2ss(b, a, mr);
}

// ── sos2ss ────────────────────────────────────────────────────────────
std::tuple<Value, Value, Value, Value>
sos2ss(const Value &sos, keel::f64 g, keel::Allocator *mr)
{
    auto [b, a] = sos2tf(sos, g, mr);
    return tf2ss(b, a, mr);
}

// ── ss2sos ────────────────────────────────────────────────────────────
Value ss2sos(const Value &A, const Value &B, const Value &C, keel::f64 D,
             keel::Allocator *mr)
{
    auto [b, a] = ss2tf(A, B, C, D, mr);
    return tf2sos(b, a, mr);
}

// ── ctf2zp (Phase 4.11) ───────────────────────────────────────────────
// Cascaded transfer function (NUM, DEN, SV) → zero/pole/gain.
// Loops over each section: tf2zpk(NUM(i,:), DEN(i,:)) → accumulate.
// Final gain = prod(SV) * prod(per-section gains).
std::tuple<Value, Value, keel::f64>
ctf2zp(const Value &NUM, const Value &DEN, const Value &SV, keel::Allocator *mr)
{
    // Vector inputs → reshape to single-row "section".
    auto rowsOf = [](const Value &v) -> keel::usize {
        return (v.dims().rows() == 1 && v.dims().cols() > 1) ? 1 : v.dims().rows();
    };
    auto colsOf = [](const Value &v) -> keel::usize {
        return (v.dims().rows() == 1 && v.dims().cols() > 1) ? v.dims().cols() : v.dims().cols();
    };
    const keel::usize Knum = rowsOf(NUM);
    const keel::usize Kden = rowsOf(DEN);
    const keel::usize K = std::max(Knum, Kden);
    if (K == 0)
        throw Error("ctf2zp: NUM/DEN must be non-empty",
                    0, 0, "ctf2zp", "", "numkit:ctf2zp:Empty");

    // SV defaults to 1 if not provided.
    keel::Array<keel::f64> sv(mr);
    if (SV.isEmpty() || SV.isEmpty()) {
        sv.assign(K + 1, 1.0);
    } else if (SV.numel() == 1) {
        sv.assign(K + 1, 1.0);
        sv[0] = SV.toScalar();  // overall gain in first slot — others stay 1.
    } else if (SV.numel() == K + 1) {
        sv.resize(K + 1);
        for (keel::usize i = 0; i <= K; ++i) sv[i] = SV.elemAsDouble(i);
    } else {
        throw Error("ctf2zp: SV must be scalar or K+1 vector",
                    0, 0, "ctf2zp", "", "numkit:ctf2zp:invalidSVDims");
    }

    // Helper to extract row i of a matrix (or replicate if matrix is just
    // a scalar/vector representing the "all sections same" case).
    auto extractRow = [&](const Value &M, keel::usize i, keel::usize Krows) {
        const keel::usize cols = colsOf(M);
        Value row = Value::matrix(1, cols, ValueType::DOUBLE, mr);
        keel::f64 *rd = row.doubleDataMut();
        if (M.dims().rows() == 1 && M.dims().cols() > 1) {
            // Vector input — used for all sections.
            for (keel::usize k = 0; k < cols; ++k) rd[k] = M.elemAsDouble(k);
        } else if (Krows == 1 && M.numel() == 1) {
            rd[0] = M.toScalar();
        } else {
            // Matrix: column-major, element at (i, k) = i + k * rows.
            const keel::usize rows = M.dims().rows();
            for (keel::usize k = 0; k < cols; ++k) rd[k] = M.elemAsDouble(i + k * rows);
        }
        return row;
    };

    keel::Array<keel::c64> zeros(mr), poles(mr);
    keel::f64 gainAccum = 1.0;
    for (keel::usize i = 0; i < K; ++i) {
        Value brow = extractRow(NUM, std::min(i, Knum - 1), Knum);
        Value arow = extractRow(DEN, std::min(i, Kden - 1), Kden);
        auto [zsec, psec, gsec] = tf2zpk(brow, arow, mr);
        const keel::usize nz = zsec.numel();
        const keel::usize np = psec.numel();
        // Append zeros/poles
        if (zsec.type() == ValueType::COMPLEX) {
            const keel::c64 *zd = zsec.complexData();
            for (keel::usize k = 0; k < nz; ++k) zeros.push_back(zd[k]);
        } else {
            for (keel::usize k = 0; k < nz; ++k)
                zeros.push_back(keel::c64(zsec.elemAsDouble(k), 0.0));
        }
        if (psec.type() == ValueType::COMPLEX) {
            const keel::c64 *pd = psec.complexData();
            for (keel::usize k = 0; k < np; ++k) poles.push_back(pd[k]);
        } else {
            for (keel::usize k = 0; k < np; ++k)
                poles.push_back(keel::c64(psec.elemAsDouble(k), 0.0));
        }
        gainAccum *= gsec;
    }
    // Multiply overall gain by prod(SV).
    keel::f64 svProd = 1.0;
    for (keel::f64 v : sv) svProd *= v;
    gainAccum *= svProd;

    // Build output Z, P as complex column vectors.
    Value Z = Value::matrix(zeros.size(), zeros.empty() ? 0 : 1,
                             ValueType::COMPLEX, mr);
    Value P = Value::matrix(poles.size(), poles.empty() ? 0 : 1,
                             ValueType::COMPLEX, mr);
    if (!zeros.empty())
        std::copy(zeros.begin(), zeros.end(), Z.complexDataMut());
    if (!poles.empty())
        std::copy(poles.begin(), poles.end(), P.complexDataMut());
    return {Z, P, gainAccum};
}

// ── scaleFilterSections — scale cascaded-transfer-function numerators ─
// Clean-room implementation written from public references
// and the public references it cites:
//   * L. B. Jackson, Digital Filters and Signal Processing, 3rd ed.,
//     1996 — cascade (series) IIR realisation and distribution of an
//     overall gain across the cascaded sections;
//   * A. V. Oppenheim & R. W. Schafer, Discrete-Time Signal Processing,
//     3rd ed., 2010 — §6.3, cascade-form filter structures (the overall
//     transfer function is the product of the section transfer
//     functions).
// Bg = scaleFilterSections(B, g). B (CTFNum) is a K×Q matrix — row k is
// the length-Q numerator polynomial of cascade section k. g (SV) is a
// scalar, or a vector of length K+1: the first K entries are per-section
// factors, the (K+1)-th an extra overall gain distributed as a K-th
// root. The overall gain magnitude is spread as |g|^(1/K) across all
// sections; its sign is concentrated on the last section.

namespace {

// Read element `i` (column-major linear index) of any numeric Value
// uniformly as a Complex.
inline keel::c64 sfs_elemAsComplex(const Value &v, keel::usize i)
{
    if (v.type() == ValueType::COMPLEX)
        return v.complexData()[i];
    return keel::c64(v.elemAsDouble(i), 0.0);
}

// MATLAB sign(): real x → +1 / 0 / -1; complex x → x/|x| (0 when x==0).
inline keel::c64 sfs_sign(keel::c64 x)
{
    if (x.imag() == 0.0) {
        const keel::f64 r = x.real();
        if (r > 0.0) return keel::c64(1.0, 0.0);
        if (r < 0.0) return keel::c64(-1.0, 0.0);
        return keel::c64(0.0, 0.0);
    }
    const keel::f64 mag = keel::abs(x);
    if (mag == 0.0) return keel::c64(0.0, 0.0);
    return x / mag;
}

} // namespace

Value scaleFilterSections(const Value &CTFNum, const Value &SV,
                          keel::Allocator *mr)
{
    const Dims &bd = CTFNum.dims();
    const keel::usize K = bd.rows();   // cascade sections
    const keel::usize Q = bd.cols();   // numerator polynomial length

    const keel::usize nsv = SV.numel();

    // g must be a scalar or a vector of length K + 1.
    const bool scalarG = (nsv == 1);
    const bool vectorG = (nsv == K + 1);
    if (!scalarG && !vectorG) {
        throw numkit::Error(
            "Invalid number of scale values. Specify either a scalar "
            "or a vector of length equal to the number of sections + 1.",
            0, 0, "scaleFilterSections", "",
            "numkit:scaleFilterSections:invalidNumberOfScaleValues");
    }

    // Result type: COMPLEX iff either input carries complex data.
    const bool resultComplex = (CTFNum.type() == ValueType::COMPLEX)
                            || (SV.type() == ValueType::COMPLEX);
    const ValueType outType = resultComplex ? ValueType::COMPLEX
                                            : ValueType::DOUBLE;

    Value out = Value::matrix(K, Q, outType, mr);

    // The overall extra gain is g (scalar case) or g[K] (vector case).
    const keel::c64 overall = scalarG ? sfs_elemAsComplex(SV, 0)
                                    : sfs_elemAsComplex(SV, K);

    // K-th root of the overall gain magnitude.
    const keel::f64 root = (K == 0)
        ? 0.0
        : keel::pow(keel::abs(overall), 1.0 / keel::as<keel::f64>(K));

    // Sign of the overall gain — concentrated on the last section.
    const keel::c64 sgn = sfs_sign(overall);

    if (resultComplex) {
        keel::c64 *od = out.complexDataMut();
        for (keel::usize k = 0; k < K; ++k) {
            // Per-section factor: root (scalar g) or root*g[k] (vector g);
            // the last section additionally carries the sign.
            keel::c64 factor = vectorG ? root * sfs_elemAsComplex(SV, k)
                                     : keel::c64(root, 0.0);
            if (k + 1 == K)
                factor *= sgn;
            for (keel::usize j = 0; j < Q; ++j) {
                const keel::usize idx = k + j * K;  // column-major
                od[idx] = factor * sfs_elemAsComplex(CTFNum, idx);
            }
        }
    } else {
        // Real fast path — neither input is complex.
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize k = 0; k < K; ++k) {
            keel::f64 factor = vectorG ? root * SV.elemAsDouble(k) : root;
            if (k + 1 == K)
                factor *= sgn.real();
            for (keel::usize j = 0; j < Q; ++j) {
                const keel::usize idx = k + j * K;  // column-major
                od[idx] = factor * CTFNum.elemAsDouble(idx);
            }
        }
    }

    return out;
}

} // namespace numkit::signal
