// toolboxes/signal/src/multirate/multirate.cpp

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/multirate/multirate.hpp>
#include <numkit/signal/multirate/extras.hpp>          // upfirdn
#include <numkit/signal/filter_design/filter_design.hpp> // firls
#include <numkit/signal/filter_design/iir_designs.hpp>   // cheby1
#include <numkit/signal/digital_filtering/filter.hpp>    // filtfilt
#include <numkit/signal/windows/windows.hpp>           // kaiser

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName / isIntegerType
#include <numkit/ops/helpers.hpp>

#include "../signal_detail.hpp"     // shared argcheck cores
#include "../dsp_helpers.hpp"       // Complex typedef

#include <algorithm>
#include <numeric>


namespace numkit::signal {

namespace {

// ── downsample / upsample argchecks ───────────────────────────────────
// Probed R2025b: x accepts every numeric class plus logical and complex
// (each preserved in the result), vectors AND matrices (downsampled
// column-wise); N-D is accepted by the reference but the kernel here is
// 2-D (bugs/opened/signal/nd-and-matrix-gaps.md); integer classes would
// need integer-class outputs the Value layer cannot allocate yet
// (bugs/opened/signal/integer-class-outputs.md).
void multirateGatherArgcheck(const Value &x, const char *fn)
{
    const ValueType t = x.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL
        || isIntegerType(t)) {
        return;
    }
    detail::throwInputType(x, fn, "x", 1, "numeric or logical");
}

void multiratePhaseArgcheck(keel::usize n, keel::usize phase, const char *fn)
{
    if (phase >= n)
        throw Error(keel::format("{}: phase must be an integer in [0, n-1] (got {} with n = {})",
                                 fn, phase, n),
                    0, 0, fn, "", keel::format("numkit:{}:badPhase", fn));
}

// Column count / orientation helpers shared by the typed gathers.
struct GatherShape {
    keel::usize rows;      // per-column length
    keel::usize cols;      // product of trailing dimensions
    bool   isRow;     // input was a 2-D row vector → row output
};
GatherShape gatherShape(const Value &x)
{
    if (x.dims().rows() == 1 && x.dims().ndim() == 2)
        return {x.numel(), 1, true};  // row vector (incl. 1×1): row output
    if (x.dims().isVector() || x.isScalar())
        return {x.numel(), 1, false}; // column vector / scalar
    const keel::usize r = x.dims().rows();
    const keel::usize c = (r > 0) ? (x.numel() / r) : x.dims().cols();
    return {r, c, false};
}

Value allocMultirateOut(const Value &x, const GatherShape &gs, keel::usize outRows,
                        keel::Allocator *mr)
{
    const ValueType t = x.type();
    if (gs.isRow)
        return Value::matrix(1, outRows * gs.cols, t, mr);
    const keel::i32 nd = x.dims().ndim();
    if (nd <= 2)
        return Value::matrix(outRows, gs.cols, t, mr);
    keel::Array<keel::usize> od(nd, mr);
    od[0] = outRows;
    for (keel::i32 d = 1; d < nd; ++d)
        od[keel::as<keel::usize>(d)] = x.dims().dim(d);
    return Value::matrixND(od.data(), nd, t, mr);
}

} // anonymous namespace

namespace {

// Windowed-sinc lowpass FIR, Hamming window, cutoff wc (radians).
// Normalized so DC gain is 1. Order is filtLen - 1; filtLen must be >= 2.
[[maybe_unused]] keel::Array<keel::f64> designLowpassFir(keel::usize filtLen, keel::f64 wc, keel::Allocator *mr)
{
    const keel::usize filtOrder = filtLen - 1;
    const keel::f64 half = filtOrder / 2.0;

    keel::Array<keel::f64> h(filtLen, mr);
    keel::f64 hSum = 0.0;
    for (keel::usize i = 0; i < filtLen; ++i) {
        const keel::f64 n = i - half;
        const keel::f64 sinc = (keel::abs(n) < 1e-12)
                                ? wc / keel::kPi
                                : keel::sin(wc * n) / (keel::kPi * n);
        const keel::f64 win = 0.54 - 0.46 * keel::cos(2.0 * keel::kPi * i / filtOrder);
        h[i] = sinc * win;
        hSum += h[i];
    }
    for (keel::usize i = 0; i < filtLen; ++i)
        h[i] /= hSum;
    return h;
}

// Direct Form II transposed FIR apply — matches filter.cpp's core for
// the a = [1] denominator case. Used by decimate and resample.
[[maybe_unused]] keel::Array<keel::f64> applyFirDf2t(const keel::f64 *h, keel::usize filtLen, const keel::f64 *x, keel::usize nx, keel::Allocator *mr)
{
    keel::Array<keel::f64> out(nx, mr);
    keel::Array<keel::f64> z(filtLen, mr);
    for (keel::usize n = 0; n < nx; ++n) {
        out[n] = h[0] * x[n] + z[0];
        for (keel::usize i = 1; i < filtLen; ++i)
            z[i - 1] = h[i] * x[n] + (i < filtLen - 1 ? z[i] : 0.0);
    }
    return out;
}

void downsample_argcheck(const Value &x, keel::usize n, keel::usize phase)
{
    multirateGatherArgcheck(x, "downsample");
    multiratePhaseArgcheck(n, phase, "downsample");
}

void upsample_argcheck(const Value &x, keel::usize n, keel::usize phase)
{
    multirateGatherArgcheck(x, "upsample");
    multiratePhaseArgcheck(n, phase, "upsample");
}

void decimate_argcheck(const Value &x)
{
    detail::requireFloatInput(x, "decimate");
}

void resample_argcheck(const Value &x, keel::usize p, keel::usize q)
{
    detail::requireFloatInput(x, "resample");
    if (p == 0 || q == 0)
        throw Error("resample: P and Q must be positive integers",
                    0, 0, "resample", "", "numkit:resample:pq");
}

} // anonymous namespace

// ── downsample ────────────────────────────────────────────────────────
Value downsample(const Value &x, keel::usize n, keel::usize phase, keel::Allocator *mr)
{
    downsample_argcheck(x, n, phase);
    const GatherShape gs = gatherShape(x);
    const keel::usize outRows = (phase < gs.rows)
                               ? (gs.rows - phase + n - 1) / n
                               : 0;
    Value r = allocMultirateOut(x, gs, outRows, mr);
    if (outRows == 0 || gs.cols == 0 || x.numel() == 0)
        return r;
    const keel::usize es = x.rawBytes() / x.numel();
    const char *xd = static_cast<const char *>(x.rawData());
    char *od = static_cast<char *>(r.rawDataMut());
    for (keel::usize c = 0; c < gs.cols; ++c)
        for (keel::usize i = 0; i < outRows; ++i)
            keel::mem_copy(od + (c * outRows + i) * es,
                           xd + (c * gs.rows + phase + i * n) * es,
                           es);
    return r;
}

// ── upsample ──────────────────────────────────────────────────────────
Value upsample(const Value &x, keel::usize n, keel::usize phase, keel::Allocator *mr)
{
    upsample_argcheck(x, n, phase);
    const GatherShape gs = gatherShape(x);
    const keel::usize outRows = gs.rows * n;
    Value r = allocMultirateOut(x, gs, outRows, mr);
    if (outRows == 0 || gs.cols == 0 || x.numel() == 0)
        return r;
    const keel::usize es = x.rawBytes() / x.numel();
    const char *xd = static_cast<const char *>(x.rawData());
    char *od = static_cast<char *>(r.rawDataMut());
    for (keel::usize c = 0; c < gs.cols; ++c)
        for (keel::usize i = 0; i < gs.rows; ++i)
            keel::mem_copy(od + (c * outRows + phase + i * n) * es,
                           xd + (c * gs.rows + i) * es,
                           es);
    return r;
}

// ── decimate ──────────────────────────────────────────────────────────
// Type contract (probed R2025b): x accepts double/single/complex — the
// pipeline here is double-buffer only, so single and complex are
// honest capability gaps; logical and integer classes are rejected by
// the reference outright.
Value decimate(const Value &x, keel::usize factor, keel::Allocator *mr)
{
    decimate_argcheck(x);
    const bool xSingle = (x.type() == ValueType::SINGLE);
    if (x.isComplex()) {
        const keel::usize nx = x.numel();
        Value xr = createLike(x, ValueType::DOUBLE, mr);
        Value xi = createLike(x, ValueType::DOUBLE, mr);
        keel::f64 *xrd = xr.doubleDataMut();
        keel::f64 *xid = xi.doubleDataMut();
        const keel::c64 *xcd = x.complexData();
        for (keel::usize i = 0; i < nx; ++i) {
            xrd[i] = xcd[i].real();
            xid[i] = xcd[i].imag();
        }
        Value yr = decimate(xr, factor, mr);
        Value yi = decimate(xi, factor, mr);
        Value out = createLike(yr, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        const keel::f64 *yrd = yr.doubleData();
        const keel::f64 *yid = yi.doubleData();
        for (keel::usize i = 0; i < yr.numel(); ++i)
            od[i] = keel::c64(yrd[i], yid[i]);
        return out;
    }
    Value xD = detail::widenSingleToDouble(x, mr);
    const keel::usize nx = xD.numel();
    const bool isRow = xD.dims().rows() == 1;
    if (factor <= 1 || nx == 0) {
        auto r = isRow ? Value::matrix(1, nx, ValueType::DOUBLE, mr)
                       : Value::matrix(nx, 1, ValueType::DOUBLE, mr);
        if (nx > 0)
            std::copy(xD.doubleData(), xD.doubleData() + nx, r.doubleDataMut());
        return detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr);
    }

    auto [b, a] = cheby1(8, 0.05, Value::scalar(0.8 / keel::as<keel::f64>(factor), mr),
                         FilterType::Lowpass, /*analog=*/false, mr);
    Value filtered = filtfilt(b, a, xD, mr);
    const keel::f64 *fd = filtered.doubleData();

    const keel::usize outLen = (nx + factor - 1) / factor;
    const keel::usize nbeg = (nx - 1) % factor;
    auto r = isRow ? Value::matrix(1, outLen, ValueType::DOUBLE, mr)
                   : Value::matrix(outLen, 1, ValueType::DOUBLE, mr);
    keel::f64 *rd = r.doubleDataMut();
    for (keel::usize i = 0; i < outLen; ++i)
        rd[i] = fd[nbeg + i * factor];
    return detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr);
}

// ── resample ──────────────────────────────────────────────────────────
Value resample(const Value &x, keel::usize p, keel::usize q, keel::Allocator *mr)
{
    resample_argcheck(x, p, q);
    return resampleWithFilter(x, p, q, 10, 5.0, mr).first;
}

Value resample(const Value &x, keel::usize p, keel::usize q, keel::i32 n_order, keel::f64 beta_in, keel::Allocator *mr)
{
    return resampleWithFilter(x, p, q, n_order, beta_in, mr).first;
}

keel::Pair<Value, Value> resampleWithFilter(const Value &x, keel::usize p, keel::usize q, keel::i32 n_order, keel::f64 beta_in, keel::Allocator *mr)
{
    detail::requireFloatInput(x, "resample");
    if (p == 0 || q == 0)
        throw Error("resample: P and Q must be positive integers",
                    0, 0, "resample", "", "numkit:resample:pq");
    const bool xSingle = (x.type() == ValueType::SINGLE);
    if (x.isComplex()) {
        const keel::usize nx = x.numel();
        Value xr = createLike(x, ValueType::DOUBLE, mr);
        Value xi = createLike(x, ValueType::DOUBLE, mr);
        keel::f64 *xrd = xr.doubleDataMut();
        keel::f64 *xid = xi.doubleDataMut();
        const keel::c64 *xcd = x.complexData();
        for (keel::usize i = 0; i < nx; ++i) {
            xrd[i] = xcd[i].real();
            xid[i] = xcd[i].imag();
        }
        auto [yr, b] = resampleWithFilter(xr, p, q, n_order, beta_in, mr);
        Value yi = resampleWithFilter(xi, p, q, n_order, beta_in, mr).first;
        Value out = createLike(yr, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        const keel::f64 *yrd = yr.doubleData();
        const keel::f64 *yid = yi.doubleData();
        for (keel::usize i = 0; i < yr.numel(); ++i)
            od[i] = keel::c64(yrd[i], yid[i]);
        return {keel::move(out), keel::move(b)};
    }

    Value xD = detail::widenSingleToDouble(x, mr);

    // Reduce P/Q by their GCD first.
    const keel::usize g = std::gcd(p, q);
    p /= g;
    q /= g;

    const keel::usize nx = xD.numel();
    const bool isRow = xD.dims().rows() == 1;

    // Anti-alias filter: fc = 1/(2·max(p,q)); L = 2·N·max(p,q)+1.
    // Because the bands [0, 2*fc] and [2*fc, 1] partition [0, 1] with zero
    // transition band and unit weight, the firls Gram matrix is exactly diagonal
    // and its exact solution is the ideal sinc lowpass:
    //   fb[i] = W * sinc(W * (i - M)), W = 2*fc, M = (L-1)/2.
    // Computing it in closed form is O(L) (avoiding O(L^3) dense solve for large L)
    // and matches firls(L-1, [0 W W 1], [1 1 0 0]) to machine precision.
    const keel::i32    N    = (n_order > 0) ? n_order : 10;
    const keel::f64 beta = beta_in;
    const keel::usize pqmax = std::max(p, q);
    const keel::f64 fc = 1.0 / (2.0 * keel::as<keel::f64>(pqmax));
    const keel::f64 W  = 2.0 * fc;
    const keel::i32    L  = 2 * N * keel::as<keel::i32>(pqmax) + 1;
    const keel::i32    M  = (L - 1) / 2;

    Value kw = kaiser(keel::as<keel::usize>(L), beta, mr); // length L
    const keel::f64 *kwd = kw.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto fk = keel::Array<keel::f64>(L, scratch.allocator());
    keel::f64 sumfk = 0.0;
    constexpr keel::f64 kPi = keel::kPi;
    for (keel::i32 i = 0; i < L; ++i) {
        const keel::i32 k = i - M;
        const keel::f64 fbv = (k == 0)
            ? W
            : keel::sin(kPi * W * keel::as<keel::f64>(k)) / (kPi * keel::as<keel::f64>(k));
        fk[i] = fbv * kwd[i];
        sumfk += fk[i];
    }

    const keel::f64 scale = keel::as<keel::f64>(p) / sumfk;
    Value bOut = Value::matrix(1, keel::as<keel::usize>(L), ValueType::DOUBLE, mr);
    {
        keel::f64 *bd = bOut.doubleDataMut();
        for (keel::i32 i = 0; i < L; ++i) bd[i] = scale * fk[i];
    }

    // Front-pad the filter so the group delay lands on an output sample.
    const keel::i64 Lhalf  = (L - 1) / 2;
    const keel::i64 nz     = keel::as<keel::i64>(q) - (Lhalf % keel::as<keel::i64>(q));
    const keel::i64 Lhalf2 = Lhalf + nz;

    Value hh = Value::matrix(1, keel::as<keel::usize>(nz) + keel::as<keel::usize>(L),
                             ValueType::DOUBLE, mr);
    {
        keel::f64 *hd = hh.doubleDataMut();
        for (keel::i64 i = 0; i < nz; ++i) hd[i] = 0.0;
        for (keel::i32 i = 0; i < L; ++i) hd[nz + i] = scale * fk[i];
    }

    Value yfull = upfirdn(xD, hh, p, q, mr);
    const keel::usize  yfn = yfull.numel();
    const keel::f64 *yfd = yfull.doubleData();

    const keel::i64 delay = Lhalf2 / keel::as<keel::i64>(q);     // floor(ceil(Lhalf2)/q)
    const keel::i64 Ly    = keel::as<keel::i64>((nx * p + q - 1) / q); // ceil(Lx·p/q)
    const keel::usize    outLen = (Ly > 0) ? keel::as<keel::usize>(Ly) : 0;

    Value r = isRow ? Value::matrix(1, outLen, ValueType::DOUBLE, mr)
                    : Value::matrix(outLen, 1, ValueType::DOUBLE, mr);
    keel::f64 *rd = r.doubleDataMut();
    for (keel::usize i = 0; i < outLen; ++i) {
        const keel::i64 idx = delay + keel::as<keel::i64>(i);
        rd[i] = (idx >= 0 && keel::as<keel::usize>(idx) < yfn) ? yfd[idx] : 0.0;
    }
    return {detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr), keel::move(bOut)};
}

} // namespace numkit::signal
