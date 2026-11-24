// toolboxes/signal/src/multirate/multirate.cpp
// downsample / upsample / decimate / resample (decimate & resample delegate to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/multirate/multirate.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

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

struct GatherShape {
    keel::usize rows;
    keel::usize cols;
    bool isRow;
};

GatherShape gatherShape(const Value &x)
{
    if (x.dims().rows() == 1 && x.dims().ndim() == 2)
        return {x.numel(), 1, true};
    if (x.dims().isVector() || x.isScalar())
        return {x.numel(), 1, false};
    const keel::usize r = x.dims().rows();
    const keel::usize c = (r > 0) ? (x.numel() / r) : x.dims().cols();
    return {r, c, false};
}

Value allocMultirateOut(const Value &x, const GatherShape &gs, keel::usize outRows,
                        keel::Allocator *mr)
{
    const ValueType t = x.type();
    if (gs.isRow) return Value::matrix(1, outRows * gs.cols, t, mr);
    const keel::i32 nd = x.dims().ndim();
    if (nd <= 2) return Value::matrix(outRows, gs.cols, t, mr);
    keel::Array<keel::usize> od(nd, mr);
    od[0] = outRows;
    for (keel::i32 d = 1; d < nd; ++d) od[keel::as<keel::usize>(d)] = x.dims().dim(d);
    return Value::matrixND(od.data(), nd, t, mr);
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

Value downsample(const Value &x, keel::usize n, keel::usize phase, keel::Allocator *mr)
{
    downsample_argcheck(x, n, phase);
    const GatherShape gs = gatherShape(x);
    const keel::usize outRows = (phase < gs.rows) ? (gs.rows - phase + n - 1) / n : 0;
    Value r = allocMultirateOut(x, gs, outRows, mr);
    if (outRows == 0 || gs.cols == 0 || x.numel() == 0) return r;
    const keel::usize es = x.rawBytes() / x.numel();
    const char *xd = static_cast<const char *>(x.rawData());
    char *od = static_cast<char *>(r.rawDataMut());
    for (keel::usize c = 0; c < gs.cols; ++c)
        for (keel::usize i = 0; i < outRows; ++i)
            keel::mem_copy(od + (c * outRows + i) * es,
                           xd + (c * gs.rows + phase + i * n) * es, es);
    return r;
}

Value upsample(const Value &x, keel::usize n, keel::usize phase, keel::Allocator *mr)
{
    upsample_argcheck(x, n, phase);
    const GatherShape gs = gatherShape(x);
    const keel::usize outRows = gs.rows * n;
    Value r = allocMultirateOut(x, gs, outRows, mr);
    if (outRows == 0 || gs.cols == 0 || x.numel() == 0) return r;
    const keel::usize es = x.rawBytes() / x.numel();
    const char *xd = static_cast<const char *>(x.rawData());
    char *od = static_cast<char *>(r.rawDataMut());
    for (keel::usize c = 0; c < gs.cols; ++c)
        for (keel::usize i = 0; i < gs.rows; ++i)
            keel::mem_copy(od + (c * outRows + phase + i * n) * es,
                           xd + (c * gs.rows + i) * es, es);
    return r;
}

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
        for (keel::usize i = 0; i < yr.numel(); ++i) od[i] = keel::c64(yrd[i], yid[i]);
        return out;
    }

    Value xD = detail::widenSingleToDouble(x, mr);
    const bool isRow = (xD.dims().rows() == 1);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> outBuf(scratch.allocator());
    (void)keel::signal::decimate(
        keel::ArrayRef<keel::f64>(xD.doubleData(), xD.numel()),
        factor, outBuf, scratch.allocator());
    Value r = isRow ? Value::matrix(1, outBuf.size(), ValueType::DOUBLE, mr)
                    : Value::matrix(outBuf.size(), 1, ValueType::DOUBLE, mr);
    if (!outBuf.empty()) keel::copy_n(r.doubleDataMut(), outBuf.data(), outBuf.size());
    return detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr);
}

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
    resample_argcheck(x, p, q);
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
        for (keel::usize i = 0; i < yr.numel(); ++i) od[i] = keel::c64(yrd[i], yid[i]);
        return {keel::move(out), keel::move(b)};
    }

    Value xD = detail::widenSingleToDouble(x, mr);
    const bool isRow = (xD.dims().rows() == 1);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> yBuf(scratch.allocator()), bBuf(scratch.allocator());
    (void)keel::signal::resample(
        keel::ArrayRef<keel::f64>(xD.doubleData(), xD.numel()),
        p, q, n_order, beta_in, yBuf, &bBuf, scratch.allocator());

    Value r = isRow ? Value::matrix(1, yBuf.size(), ValueType::DOUBLE, mr)
                    : Value::matrix(yBuf.size(), 1, ValueType::DOUBLE, mr);
    if (!yBuf.empty()) keel::copy_n(r.doubleDataMut(), yBuf.data(), yBuf.size());
    Value bOut = Value::matrix(1, bBuf.size(), ValueType::DOUBLE, mr);
    if (!bBuf.empty()) keel::copy_n(bOut.doubleDataMut(), bBuf.data(), bBuf.size());
    return {detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr), keel::move(bOut)};
}

} // namespace numkit::signal
