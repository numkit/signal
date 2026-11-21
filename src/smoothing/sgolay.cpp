// toolboxes/signal/src/smoothing/sgolay.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/math.h>
#include <numkit/signal/smoothing/sgolay.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <cmath>
#include <cstring>

namespace numkit::signal {

namespace {

// Solve A · X = B in place (Gauss-Jordan with partial pivoting).
// A is N×N row-major, B is N×M row-major.
void gaussJordan(keel::f64 *A, keel::f64 *B, keel::i32 N, keel::i32 M)
{
    for (keel::i32 k = 0; k < N; ++k) {
        // Pivot.
        keel::i32 piv = k;
        keel::f64 maxAbs = keel::abs(A[k * N + k]);
        for (keel::i32 r = k + 1; r < N; ++r) {
            const keel::f64 v = keel::abs(A[r * N + k]);
            if (v > maxAbs) { maxAbs = v; piv = r; }
        }
        if (maxAbs < 1e-300)
            throw Error("sgolay: singular normal equations "
                         "(framelen too small for order)",
                         0, 0, "sgolay", "", "numkit:sgolay:singular");
        if (piv != k) {
            for (keel::i32 c = 0; c < N; ++c) std::swap(A[k * N + c], A[piv * N + c]);
            for (keel::i32 c = 0; c < M; ++c) std::swap(B[k * M + c], B[piv * M + c]);
        }
        // Normalise pivot row.
        const keel::f64 inv = 1.0 / A[k * N + k];
        for (keel::i32 c = 0; c < N; ++c) A[k * N + c] *= inv;
        for (keel::i32 c = 0; c < M; ++c) B[k * M + c] *= inv;
        // Eliminate.
        for (keel::i32 r = 0; r < N; ++r) {
            if (r == k) continue;
            const keel::f64 f = A[r * N + k];
            if (f == 0.0) continue;
            for (keel::i32 c = 0; c < N; ++c) A[r * N + c] -= f * A[k * N + c];
            for (keel::i32 c = 0; c < M; ++c) B[r * M + c] -= f * B[k * M + c];
        }
    }
}

// Build the (framelen × (order+1)) Vandermonde matrix V where
// V[i, k] = (i - center)^k for i = 0..framelen-1 and k = 0..order.
keel::Array<keel::f64> buildVandermonde(keel::i32 order, keel::i32 framelen, keel::Allocator *mr)
{
    const keel::i32 n = framelen;
    const keel::i32 p = order + 1;
    keel::Array<keel::f64> V(n * p, mr);
    const keel::f64 half = keel::as<keel::f64>(framelen / 2);
    for (keel::i32 i = 0; i < n; ++i) {
        keel::f64 v = 1.0;
        const keel::f64 x = keel::as<keel::f64>(i) - half;
        for (keel::i32 k = 0; k < p; ++k) {
            V[i * p + k] = v;
            v *= x;
        }
    }
    return V;
}

// Compute B = V · (V' · W · V)^-1 · V' · W   (the framelen × framelen
// projection matrix), where W = diag(w) is the optional weighting. Each
// row r of B gives the filter coefficients for sample r in the window:
// y_r = B[r, :] · x_window. `w` is nullptr (unweighted, W = I) or a
// framelen-length vector of positive weights.
keel::Array<keel::f64> buildProjection(keel::i32 order, keel::i32 framelen,
                                   keel::Allocator *mr,
                                   const keel::f64 *w = nullptr)
{
    const keel::i32 n = framelen;
    const keel::i32 p = order + 1;
    auto V = buildVandermonde(order, framelen, mr);      // n × p

    // Form V'·W·V  (p × p) and V'·W  (p × n) on the side. With w == nullptr
    // the weight factor is 1.0 and this reduces to the ordinary V'V / V'.
    keel::Array<keel::f64> VtV(p * p, mr);
    keel::Array<keel::f64> Vt (p * n, mr);
    for (keel::i32 k = 0; k < p; ++k)
        for (keel::i32 i = 0; i < n; ++i)
            Vt[k * n + i] = V[i * p + k] * (w ? w[i] : 1.0);
    for (keel::i32 i = 0; i < p; ++i)
        for (keel::i32 j = 0; j < p; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 t = 0; t < n; ++t)
                s += V[t * p + i] * (w ? w[t] : 1.0) * V[t * p + j];
            VtV[i * p + j] = s;
        }

    // Solve VtV · X = Vt → X is p × n; then B = V · X (n × n).
    keel::Array<keel::f64> X(Vt, mr);
    gaussJordan(VtV.data(), X.data(), p, n);

    keel::Array<keel::f64> B(n * n, mr);
    for (keel::i32 i = 0; i < n; ++i)
        for (keel::i32 j = 0; j < n; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 k = 0; k < p; ++k)
                s += V[i * p + k] * X[k * n + j];
            B[i * n + j] = s;
        }
    return B;
}

// Compute G = V · (V'·V)^-1  (the framelen × (order+1) differentiation-
// filter matrix, row-major). Column j of G is the FIR filter that estimates
// the polynomial coefficient a_j (a_1 = value, a_2 = 1st-deriv term, …) at
// the central point. This is the unweighted second output of MATLAB's
// [B,G] = sgolay(order,framelen).
keel::Array<keel::f64> buildDiffMatrix(keel::i32 order, keel::i32 framelen, keel::Allocator *mr)
{
    const keel::i32 n = framelen;
    const keel::i32 p = order + 1;
    auto V = buildVandermonde(order, framelen, mr);      // n × p

    // VtV = V'·V (p × p).
    keel::Array<keel::f64> VtV(p * p, mr);
    for (keel::i32 i = 0; i < p; ++i)
        for (keel::i32 j = 0; j < p; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 t = 0; t < n; ++t)
                s += V[t * p + i] * V[t * p + j];
            VtV[i * p + j] = s;
        }

    // Invert VtV: solve VtV · inv = I.
    keel::Array<keel::f64> inv(p * p, mr);
    for (keel::i32 i = 0; i < p; ++i)
        for (keel::i32 j = 0; j < p; ++j)
            inv[i * p + j] = (i == j) ? 1.0 : 0.0;
    gaussJordan(VtV.data(), inv.data(), p, p);           // inv := (V'V)^-1

    // G = V · inv (n × p).
    keel::Array<keel::f64> G(n * p, mr);
    for (keel::i32 i = 0; i < n; ++i)
        for (keel::i32 j = 0; j < p; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 k = 0; k < p; ++k)
                s += V[i * p + k] * inv[k * p + j];
            G[i * p + j] = s;
        }
    return G;
}

// Shared validation for sgolay / sgolayDiff.
void sgolay_argcheck(keel::i32 order, keel::i32 framelen)
{
    if (framelen <= 0)
        throw Error("sgolay: framelen must be positive",
                     0, 0, "sgolay", "", "numkit:sgolay:badArg");
    if ((framelen & 1) == 0)
        throw Error("sgolay: framelen must be odd",
                     0, 0, "sgolay", "", "numkit:sgolay:evenFramelen");
    if (order < 0)
        throw Error("sgolay: order must be non-negative",
                     0, 0, "sgolay", "", "numkit:sgolay:badArg");
    if (order >= framelen)
        throw Error("sgolay: order must be less than framelen",
                     0, 0, "sgolay", "", "numkit:sgolay:orderTooHigh");
}

} // namespace

Value sgolay(keel::i32 order, keel::i32 framelen, keel::Allocator *mr)
{
    sgolay_argcheck(order, framelen);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto B = buildProjection(order, framelen, scratch.allocator());
    // Convert row-major B to column-major Value (R = framelen, C = framelen).
    auto out = Value::matrix(framelen, framelen, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::i32 i = 0; i < framelen; ++i)
        for (keel::i32 j = 0; j < framelen; ++j)
            dst[j * framelen + i] = B[i * framelen + j];
    return out;
}

Value sgolayDiff(keel::i32 order, keel::i32 framelen, keel::Allocator *mr)
{
    sgolay_argcheck(order, framelen);

    const keel::i32 p = order + 1;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto G = buildDiffMatrix(order, framelen, scratch.allocator());   // row-major n × p
    // Convert row-major G to column-major Value (R = framelen, C = order+1).
    auto out = Value::matrix(framelen, p, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::i32 i = 0; i < framelen; ++i)
        for (keel::i32 j = 0; j < p; ++j)
            dst[j * framelen + i] = G[i * p + j];
    return out;
}

namespace {

// Filter one length-n 1-D slice with the framelen×framelen projection
// matrix B (row-major). Interior samples use the central (symmetric) row
// of B; the edges use the asymmetric rows so no zero-padding artefacts
// appear. n must be >= framelen.
void sgolayfiltSlice(const keel::f64 *src, keel::f64 *dst, keel::i32 n,
                     const keel::f64 *B, keel::i32 framelen)
{
    const keel::i32 half = framelen / 2;
    const keel::f64 *Bcenter = &B[half * framelen];
    for (keel::i32 i = half; i < n - half; ++i) {
        keel::f64 s = 0.0;
        for (keel::i32 k = 0; k < framelen; ++k)
            s += Bcenter[k] * src[i - half + k];
        dst[i] = s;
    }
    for (keel::i32 i = 0; i < half; ++i) {                 // leading edge
        const keel::f64 *Brow = &B[i * framelen];
        keel::f64 s = 0.0;
        for (keel::i32 k = 0; k < framelen; ++k)
            s += Brow[k] * src[k];
        dst[i] = s;
    }
    for (keel::i32 i = n - half; i < n; ++i) {             // trailing edge
        const keel::i32 rowIdx = framelen - 1 - (n - 1 - i);
        const keel::f64 *Brow = &B[rowIdx * framelen];
        keel::f64 s = 0.0;
        for (keel::i32 k = 0; k < framelen; ++k)
            s += Brow[k] * src[n - framelen + k];
        dst[i] = s;
    }
}

// Core sgolayfilt: handles vectors and matrices along `dim` (1 or 2;
// 0 = auto = first non-singleton dimension, matching MATLAB), with an
// optional framelen-length weight vector `w` (nullptr = unweighted).
Value sgolayfiltImpl(const Value &x, keel::i32 order, keel::i32 framelen,
                     const keel::f64 *w, keel::i32 dim, keel::Allocator *mr)
{
    if (x.type() == ValueType::COMPLEX)
        throw Error("sgolayfilt: complex inputs are not supported",
                     0, 0, "sgolayfilt", "", "numkit:sgolayfilt:complex");
    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("sgolayfilt: N-D (>2) inputs are not supported",
                     0, 0, "sgolayfilt", "", "numkit:sgolayfilt:ndims");

    const keel::usize R = d.rows(), C = d.cols();
    keel::i32 useDim = dim;
    if (useDim <= 0) useDim = (R > 1) ? 1 : 2;       // first non-singleton
    if (useDim != 1 && useDim != 2)
        throw Error("sgolayfilt: dim must be 1 or 2",
                     0, 0, "sgolayfilt", "", "numkit:sgolayfilt:badDim");

    const keel::i32 n = keel::as<keel::i32>(useDim == 1 ? R : C);
    if (n < framelen)
        throw Error("sgolayfilt: signal length along dim must be >= framelen",
                     0, 0, "sgolayfilt", "", "numkit:sgolayfilt:tooShort");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto B = buildProjection(order, framelen, scratch.allocator(), w);  // throws if shape invalid

    auto out = createLike(x, ValueType::DOUBLE, mr);
    if (x.numel() == 0) return out;
    keel::f64 *dst = out.doubleDataMut();

    // Gather source as DOUBLE (column-major).
    auto src = keel::Array<keel::f64>(x.numel(), scratch.allocator());
    for (keel::usize i = 0; i < x.numel(); ++i) src[i] = x.elemAsDouble(i);

    if (useDim == 1) {
        // Each column is contiguous (length R) in column-major storage.
        for (keel::usize c = 0; c < C; ++c)
            sgolayfiltSlice(src.data() + c * R, dst + c * R,
                            keel::as<keel::i32>(R), B.data(), framelen);
    } else {
        // Each row is strided by R: gather → filter → scatter.
        auto rbuf = keel::Array<keel::f64>(C, scratch.allocator());
        auto obuf = keel::Array<keel::f64>(C, scratch.allocator());
        for (keel::usize r = 0; r < R; ++r) {
            for (keel::usize c = 0; c < C; ++c) rbuf[c] = src[r + c * R];
            sgolayfiltSlice(rbuf.data(), obuf.data(),
                            keel::as<keel::i32>(C), B.data(), framelen);
            for (keel::usize c = 0; c < C; ++c) dst[r + c * R] = obuf[c];
        }
    }
    return out;
}

} // namespace

Value sgolayfilt(const Value &x, keel::i32 order, keel::i32 framelen, keel::Allocator *mr)
{
    return sgolayfiltImpl(x, order, framelen, nullptr, 0, mr);
}

Value sgolayfilt(const Value &x, keel::i32 order, keel::i32 framelen,
                 const Value &weights, keel::i32 dim, keel::Allocator *mr)
{
    if (weights.isEmpty())
        return sgolayfiltImpl(x, order, framelen, nullptr, dim, mr);
    if (weights.type() == ValueType::COMPLEX)
        throw Error("sgolayfilt: weights must be real and positive",
                     0, 0, "sgolayfilt", "", "numkit:sgolayfilt:weights");
    if (keel::as<keel::i32>(weights.numel()) != framelen)
        throw Error("sgolayfilt: weights must have length framelen",
                     0, 0, "sgolayfilt", "", "numkit:sgolayfilt:weightsLen");
    keel::ArenaTemp wa = keel::get_scratch(mr);
    auto w = keel::Array<keel::f64>(framelen, wa.allocator());
    for (keel::i32 i = 0; i < framelen; ++i) {
        const keel::f64 wi = weights.elemAsDouble(i);
        if (!(wi > 0.0))
            throw Error("sgolayfilt: weights must be real and positive",
                         0, 0, "sgolayfilt", "", "numkit:sgolayfilt:weights");
        w[i] = wi;
    }
    return sgolayfiltImpl(x, order, framelen, w.data(), dim, mr);
}

} // namespace numkit::signal
