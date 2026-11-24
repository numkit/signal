// toolboxes/signal/src/measurements/sigroi.cpp
//
// Signal Processing Toolbox region-of-interest (ROI) utilities
// (ROI array operations delegate to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

inline bool truthyAt(const Value &m, keel::usize i)
{
    if (m.isLogical()) return m.logicalData()[i] != 0;
    return m.elemAsDouble(i) != 0.0;
}

inline keel::i64 roiStart(const Value &roi, keel::usize i)
{
    return keel::as<keel::i64>(roi.elemAsDouble(i + 0 * roi.dims().rows()));
}

inline keel::i64 roiEnd(const Value &roi, keel::usize i)
{
    return keel::as<keel::i64>(roi.elemAsDouble(i + 1 * roi.dims().rows()));
}

keel::Array<keel::signal::SigRoi> readRois(const Value &roi, keel::Allocator *mr)
{
    const keel::usize n = roi.dims().rows();
    keel::Array<keel::signal::SigRoi> out(n, keel::signal::SigRoi{}, mr);
    for (keel::usize i = 0; i < n; ++i) {
        out[i] = keel::signal::SigRoi{roiStart(roi, i), roiEnd(roi, i)};
    }
    return out;
}

Value roisToValue(keel::ArrayRef<keel::signal::SigRoi> rois, bool narrowSingle, keel::Allocator *mr)
{
    const keel::usize m = rois.size();
    Value out = Value::matrix(m, m == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (m == 0) return detail::narrowDoubleToSingleIf(keel::move(out), narrowSingle, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < m; ++i) {
        od[i + 0 * m] = keel::as<keel::f64>(rois[i].start);
        od[i + 1 * m] = keel::as<keel::f64>(rois[i].end);
    }
    return detail::narrowDoubleToSingleIf(keel::move(out), narrowSingle, mr);
}

inline void sigroi_argcheck(const Value &roi, const char *fn)
{
    const ValueType t = roi.type();
    if (t != ValueType::DOUBLE && t != ValueType::SINGLE && t != ValueType::COMPLEX && !isIntegerType(t))
        detail::throwInputType(roi, fn, "roi", 1, "numeric");
    if (roi.dims().ndim() != 2 || roi.dims().cols() != 2)
        throw Error(keel::format("{}: roi must be an Nx2 matrix of [start, end] pairs", fn),
                    0, 0, fn, "", keel::format("numkit:{}:roiCols", fn));
}

Value allocLogicalLike(const Value &x, keel::Allocator *mr)
{
    const keel::usize L = x.numel();
    if (x.dims().rows() == 1)
        return Value::matrix(1, L, ValueType::LOGICAL, mr);
    return Value::matrix(L, L == 0 ? 0 : 1, ValueType::LOGICAL, mr);
}

} // namespace

Value binmask2sigroi(const Value &m, keel::Allocator *mr)
{
    const keel::usize L = m.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> mask(L, keel::u8{0}, scratch.allocator());
    for (keel::usize i = 0; i < L; ++i) mask[i] = truthyAt(m, i) ? 1 : 0;
    keel::Array<keel::signal::SigRoi> rois(scratch.allocator());
    (void)keel::signal::binmask2sigroi(mask.as_ref(), rois, scratch.allocator());
    return roisToValue(rois.as_ref(), false, mr);
}

Value sigroi2binmask(const Value &roi, keel::i64 len_user, keel::Allocator *mr)
{
    sigroi_argcheck(roi, "sigroi2binmask");
    if (len_user == 0)
        throw Error("sigroi2binmask: L must be positive",
                    0, 0, "sigroi2binmask", "", "numkit:sigroi2binmask:lenPositive");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rois = readRois(roi, scratch.allocator());
    keel::Array<keel::u8> mask(scratch.allocator());
    (void)keel::signal::sigroi2binmask(rois.as_ref(), len_user, mask, scratch.allocator());
    const keel::usize len = mask.size();
    Value out = Value::matrix(len, len == 0 ? 0 : 1, ValueType::LOGICAL, mr);
    if (len > 0) keel::copy_n(out.logicalDataMut(), mask.data(), len);
    return out;
}

Value extendsigroi(const Value &roi, keel::i64 Lpre, keel::i64 Lpost, keel::Allocator *mr)
{
    sigroi_argcheck(roi, "extendsigroi");
    if (Lpre < 0 || Lpost < 0)
        throw Error("extendsigroi: SL and SR must be nonnegative",
                    0, 0, "extendsigroi", "", "numkit:extendsigroi:nonnegative");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rois = readRois(roi, scratch.allocator());
    keel::Array<keel::signal::SigRoi> outRois(scratch.allocator());
    (void)keel::signal::extendsigroi(rois.as_ref(), Lpre, Lpost, outRois, scratch.allocator());
    return roisToValue(outRois.as_ref(), roi.type() == ValueType::SINGLE, mr);
}

Value shortensigroi(const Value &roi, keel::i64 Lpre, keel::i64 Lpost, keel::Allocator *mr)
{
    sigroi_argcheck(roi, "shortensigroi");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rois = readRois(roi, scratch.allocator());
    keel::Array<keel::signal::SigRoi> outRois(scratch.allocator());
    (void)keel::signal::shortensigroi(rois.as_ref(), Lpre, Lpost, outRois, scratch.allocator());
    return roisToValue(outRois.as_ref(), roi.type() == ValueType::SINGLE, mr);
}

Value mergesigroi(const Value &roi, keel::i64 sep, keel::Allocator *mr)
{
    sigroi_argcheck(roi, "mergesigroi");
    if (sep < 0)
        throw Error("mergesigroi: S must be nonnegative",
                    0, 0, "mergesigroi", "", "numkit:mergesigroi:nonnegative");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rois = readRois(roi, scratch.allocator());
    keel::Array<keel::signal::SigRoi> outRois(scratch.allocator());
    (void)keel::signal::mergesigroi(rois.as_ref(), sep, outRois, scratch.allocator());
    return roisToValue(outRois.as_ref(), roi.type() == ValueType::SINGLE, mr);
}

Value removesigroi(const Value &roi, keel::i64 maxLen, keel::Allocator *mr)
{
    sigroi_argcheck(roi, "removesigroi");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rois = readRois(roi, scratch.allocator());
    keel::Array<keel::signal::SigRoi> outRois(scratch.allocator());
    (void)keel::signal::removesigroi(rois.as_ref(), maxLen, outRois, scratch.allocator());
    return roisToValue(outRois.as_ref(), roi.type() == ValueType::SINGLE, mr);
}

Value extractsigroi(const Value &x, const Value &roi, bool concat, keel::Allocator *mr)
{
    sigroi_argcheck(roi, "extractsigroi");
    detail::requireFloatInput(x, "extractsigroi", "x", 1);
    detail::requireVectorInput(x, "extractsigroi", "x", 1);
    const bool xComplex = x.isComplex();
    const bool xSingle = (x.type() == ValueType::SINGLE);
    const keel::usize N = roi.dims().rows();
    const keel::i64 L = keel::as<keel::i64>(x.numel());
    if (concat) {
        keel::i64 totalLen = 0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::i64 s = keel::max<keel::i64>(1, roiStart(roi, i));
            const keel::i64 e = keel::min<keel::i64>(L, roiEnd(roi, i));
            if (e >= s) totalLen += (e - s + 1);
        }
        if (totalLen == 0)
            return xComplex ? Value::complexMatrix(0, 1, mr)
                            : Value::matrix(0, 1, xSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        if (xComplex) {
            auto out = Value::complexMatrix(keel::as<keel::usize>(totalLen), 1, mr);
            keel::c64 *od = out.complexDataMut();
            const keel::c64 *xd = x.complexData();
            keel::usize pos = 0;
            for (keel::usize i = 0; i < N; ++i) {
                const keel::i64 s = keel::max<keel::i64>(1, roiStart(roi, i));
                const keel::i64 e = keel::min<keel::i64>(L, roiEnd(roi, i));
                for (keel::i64 k = s; k <= e; ++k) od[pos++] = xd[k - 1];
            }
            return out;
        }
        Value out = Value::matrix(keel::as<keel::usize>(totalLen), 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        keel::usize pos = 0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::i64 s = keel::max<keel::i64>(1, roiStart(roi, i));
            const keel::i64 e = keel::min<keel::i64>(L, roiEnd(roi, i));
            for (keel::i64 k = s; k <= e; ++k) od[pos++] = x.elemAsDouble(keel::as<keel::usize>(k - 1));
        }
        return detail::narrowDoubleToSingleIf(keel::move(out), xSingle, mr);
    }
    Value out = Value::cell(N, N == 0 ? 0 : 1, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 s = keel::max<keel::i64>(1, roiStart(roi, i));
        const keel::i64 e = keel::min<keel::i64>(L, roiEnd(roi, i));
        const keel::i64 segLen = (e >= s) ? (e - s + 1) : 0;
        if (xComplex) {
            Value seg = Value::complexMatrix(keel::as<keel::usize>(segLen), 1, mr);
            if (segLen > 0) {
                keel::c64 *sd = seg.complexDataMut();
                const keel::c64 *xd = x.complexData();
                for (keel::i64 k = 0; k < segLen; ++k) sd[k] = xd[s - 1 + k];
            }
            out.cellAt(i) = keel::move(seg);
        } else {
            Value seg = Value::matrix(keel::as<keel::usize>(segLen), segLen == 0 ? 0 : 1, ValueType::DOUBLE, mr);
            if (segLen > 0) {
                keel::f64 *sd = seg.doubleDataMut();
                for (keel::i64 k = 0; k < segLen; ++k)
                    sd[k] = x.elemAsDouble(keel::as<keel::usize>(s - 1 + k));
            }
            out.cellAt(i) = detail::narrowDoubleToSingleIf(keel::move(seg), xSingle, mr);
        }
    }
    return out;
}

Value sigrangebinmask(const Value &x, keel::f64 threshold, keel::Allocator *mr)
{
    Value out = allocLogicalLike(x, mr);
    const keel::usize L = x.numel();
    if (L == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < L; ++i) od[i] = (x.elemAsDouble(i) > threshold) ? 1 : 0;
    return out;
}

Value sigrangebinmask(const Value &x, keel::f64 lo, keel::f64 hi, keel::Allocator *mr)
{
    Value out = allocLogicalLike(x, mr);
    const keel::usize L = x.numel();
    if (L == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < L; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        od[i] = (v >= lo && v <= hi) ? 1 : 0;
    }
    return out;
}

} // namespace numkit::signal
