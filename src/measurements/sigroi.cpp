// toolboxes/signal/src/measurements/sigroi.cpp
//
// Signal Processing Toolbox region-of-interest (ROI) utilities. All 8
// functions live in one TU since they share the same data shape:
// 2-column [start, end] matrix of 1-based ROI indices.
//
//   binmask2sigroi(m)              binary mask  → ROI [start end] pairs
//   sigroi2binmask(roi [, len])    ROIs → binary mask (column vector)
//   extendsigroi(roi, Lpre, Lpost) extend each ROI in both directions
//   shortensigroi(roi, Lpre, Lpost) opposite (drop ROIs that collapse)
//   mergesigroi(roi, sep)          merge ROIs with gap ≤ sep
//   removesigroi(roi, idx)         drop ROIs at idx (1-based)
//   extractsigroi(x, roi [, concat])  extract signal slices
//                                  (cell array by default; concatenated
//                                  vector when concat=true)
//   sigrangebinmask(x, vmin, vmax) mask where vmin ≤ x ≤ vmax
//
// PMR HARD RULE: every fn takes std::pmr::memory_resource *mr. Scratch
// via ScratchArena/ScratchVec.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (argcheck got-clauses)

#include <algorithm>

#include "../signal_detail.hpp"     // shared argcheck cores
#include "../dsp_helpers.hpp"       // Complex typedef (extractsigroi)

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

// Shared roi contract (probed R2025b): an N×2 matrix whose class is one
// of double/single/integer (LOGICAL is rejected by the reference); the
// nk kernels read doubles, so single is an honest gap
// (bugs/opened/signal/single-float-inputs.md).
inline void sigroi_argcheck(const Value &roi, const char *fn)
{
    const ValueType t = roi.type();
    if (t != ValueType::DOUBLE && t != ValueType::SINGLE && t != ValueType::COMPLEX && !isIntegerType(t))
        detail::throwInputType(roi, fn, "roi", 1, "numeric");
    if (roi.dims().ndim() != 2 || roi.dims().cols() != 2)
        throw Error(keel::format("{}: roi must be an N×2 matrix of [start, end] pairs", fn),
                    0, 0, fn, "", keel::format("numkit:{}:roiCols", fn));
}

inline void sigroiRoiArgcheck(const Value &roi, const char *fn)
{
    sigroi_argcheck(roi, fn);
}

} // namespace

// ── binmask2sigroi ────────────────────────────────────────────────────
// Find runs of true values in mask m. Returns N×2 [start, end] matrix
// (1-based, inclusive).
Value binmask2sigroi(const Value &m, keel::Allocator *mr)
{
    const keel::usize L = m.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i64> starts(0, scratch.allocator()), ends(0, scratch.allocator());
    bool inRun = false;
    keel::i64 curStart = 0;
    for (keel::usize i = 0; i < L; ++i) {
        const bool v = truthyAt(m, i);
        if (v && !inRun) { inRun = true; curStart = keel::as<keel::i64>(i + 1); }
        else if (!v && inRun) {
            inRun = false;
            starts.push_back(curStart);
            ends.push_back(keel::as<keel::i64>(i));  // i = position past last
        }
    }
    if (inRun) {
        starts.push_back(curStart);
        ends.push_back(keel::as<keel::i64>(L));
    }
    const keel::usize N = starts.size();
    Value out = Value::matrix(N, N == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        od[i + 0 * N] = keel::as<keel::f64>(starts[i]);
        od[i + 1 * N] = keel::as<keel::f64>(ends[i]);
    }
    return out;
}

// ── sigroi2binmask ────────────────────────────────────────────────────
// Build a logical column vector of length `len` (auto = max end) with
// true on each [start..end] segment.
Value sigroi2binmask(const Value &roi, keel::i64 len_user,
                     keel::Allocator *mr)
{
    sigroiRoiArgcheck(roi, "sigroi2binmask");
    if (len_user == 0)
        throw Error("sigroi2binmask: L must be positive",
                    0, 0, "sigroi2binmask", "", "numkit:sigroi2binmask:lenPositive");
    const keel::usize N = roi.dims().rows();
    keel::i64 len = len_user;
    if (len < 0) {
        len = 0;
        for (keel::usize i = 0; i < N; ++i)
            len = std::max(len, roiEnd(roi, i));
    }
    Value out = Value::matrix(keel::as<keel::usize>(len), len == 0 ? 0 : 1,
                              ValueType::LOGICAL, mr);
    if (len == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    std::fill(od, od + len, keel::u8(0));
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 s = std::max<keel::i64>(1, roiStart(roi, i));
        const keel::i64 e = std::min<keel::i64>(len, roiEnd(roi, i));
        for (keel::i64 k = s; k <= e; ++k) od[k - 1] = 1;
    }
    return out;
}

// ── extendsigroi ──────────────────────────────────────────────────────
// Extend each ROI by Lpre on the left and Lpost on the right (clamping
// start to ≥ 1). End is unbounded (no signal length provided).
Value extendsigroi(const Value &roi, keel::i64 Lpre, keel::i64 Lpost,
                   keel::Allocator *mr)
{
    sigroiRoiArgcheck(roi, "extendsigroi");
    if (Lpre < 0 || Lpost < 0)
        throw Error("extendsigroi: SL and SR must be nonnegative",
                    0, 0, "extendsigroi", "", "numkit:extendsigroi:nonnegative");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const keel::usize N = roi.dims().rows();
    Value out = Value::matrix(N, N == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (N == 0) return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        keel::i64 s = roiStart(roi, i) - Lpre;
        keel::i64 e = roiEnd(roi, i)   + Lpost;
        if (s < 1) s = 1;
        od[i + 0 * N] = keel::as<keel::f64>(s);
        od[i + 1 * N] = keel::as<keel::f64>(e);
    }
    return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
}

// ── shortensigroi ─────────────────────────────────────────────────────
// Inverse of extend: shrink each ROI by Lpre on the left, Lpost on right.
// Drop ROIs that collapse (start > end).
Value shortensigroi(const Value &roi, keel::i64 Lpre, keel::i64 Lpost,
                    keel::Allocator *mr)
{
    sigroiRoiArgcheck(roi, "shortensigroi");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const keel::usize N = roi.dims().rows();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i64> starts(0, scratch.allocator()), ends(0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        keel::i64 s = roiStart(roi, i) + Lpre;
        keel::i64 e = roiEnd(roi, i)   - Lpost;
        if (s <= e) { starts.push_back(s); ends.push_back(e); }
    }
    const keel::usize M = starts.size();
    Value out = Value::matrix(M, M == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (M == 0) return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) {
        od[i + 0 * M] = keel::as<keel::f64>(starts[i]);
        od[i + 1 * M] = keel::as<keel::f64>(ends[i]);
    }
    return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
}

// ── mergesigroi ───────────────────────────────────────────────────────
// Sort ROIs by start, merge those with gap ≤ sep.
//   sep == 0 → only overlapping ROIs merge
//   sep ≥ 1 → adjacent or near-adjacent merge
Value mergesigroi(const Value &roi, keel::i64 sep, keel::Allocator *mr)
{
    sigroiRoiArgcheck(roi, "mergesigroi");
    if (sep < 0)
        throw Error("mergesigroi: S must be nonnegative",
                    0, 0, "mergesigroi", "", "numkit:mergesigroi:nonnegative");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const keel::usize N = roi.dims().rows();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> idx(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](keel::usize a, keel::usize b) { return roiStart(roi, a) < roiStart(roi, b); });

    keel::Array<keel::i64> starts(0, scratch.allocator()), ends(0, scratch.allocator());
    for (keel::usize k = 0; k < N; ++k) {
        const keel::i64 s = roiStart(roi, idx[k]);
        const keel::i64 e = roiEnd(roi, idx[k]);
        if (!starts.empty() && s - ends.back() <= sep + 1) {
            // Merge: extend the previous end if e is larger.
            if (e > ends.back()) ends.back() = e;
        } else {
            starts.push_back(s);
            ends.push_back(e);
        }
    }
    const keel::usize M = starts.size();
    Value out = Value::matrix(M, M == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (M == 0) return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) {
        od[i + 0 * M] = keel::as<keel::f64>(starts[i]);
        od[i + 1 * M] = keel::as<keel::f64>(ends[i]);
    }
    return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
}

// ── removesigroi ──────────────────────────────────────────────────────
// Drop ROIs whose length (end - start + 1) is ≤ maxLen. Per MATLAB doc:
// "removes signal regions of interest specified in roilims that have a
// length of s samples or less."
Value removesigroi(const Value &roi, keel::i64 maxLen,
                   keel::Allocator *mr)
{
    sigroiRoiArgcheck(roi, "removesigroi");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const keel::usize N = roi.dims().rows();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i64> starts(0, scratch.allocator()), ends(0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 s = roiStart(roi, i);
        const keel::i64 e = roiEnd(roi, i);
        const keel::i64 len = e - s + 1;
        if (len > maxLen) {
            starts.push_back(s);
            ends.push_back(e);
        }
    }
    const keel::usize M = starts.size();
    Value out = Value::matrix(M, M == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (M == 0) return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) {
        od[i + 0 * M] = keel::as<keel::f64>(starts[i]);
        od[i + 1 * M] = keel::as<keel::f64>(ends[i]);
    }
    return detail::narrowDoubleToSingleIf(keel::move(out), roiSingle, mr);
}

// ── extractsigroi ─────────────────────────────────────────────────────
// Default: cell array (one cell per ROI, each a column vector slice).
// concat=true: concatenate all slices into a single column vector.
// Contract (probed R2025b): x must be a non-empty double/single/complex
// VECTOR (logical rejected, matrices rejected); complex slices keep the
// imaginary part.
Value extractsigroi(const Value &x, const Value &roi, bool concat,
                    keel::Allocator *mr)
{
    sigroiRoiArgcheck(roi, "extractsigroi");
    detail::requireFloatInput(x, "extractsigroi", "x", 1);
    detail::requireVectorInput(x, "extractsigroi", "x", 1);
    const bool xComplex = x.isComplex();
    const bool xSingle = (x.type() == ValueType::SINGLE);
    const keel::usize N = roi.dims().rows();
    const keel::i64 L = keel::as<keel::i64>(x.numel());
    if (concat) {
        // Compute total length first.
        keel::i64 totalLen = 0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::i64 s = std::max<keel::i64>(1, roiStart(roi, i));
            const keel::i64 e = std::min<keel::i64>(L, roiEnd(roi, i));
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
                const keel::i64 s = std::max<keel::i64>(1, roiStart(roi, i));
                const keel::i64 e = std::min<keel::i64>(L, roiEnd(roi, i));
                for (keel::i64 k = s; k <= e; ++k)
                    od[pos++] = xd[k - 1];
            }
            return out;
        }
        Value out = Value::matrix(keel::as<keel::usize>(totalLen),
                                  totalLen == 0 ? 0 : 1,
                                  ValueType::DOUBLE, mr);
        if (totalLen == 0) return detail::narrowDoubleToSingleIf(keel::move(out), xSingle, mr);
        keel::f64 *od = out.doubleDataMut();
        keel::usize pos = 0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::i64 s = std::max<keel::i64>(1, roiStart(roi, i));
            const keel::i64 e = std::min<keel::i64>(L, roiEnd(roi, i));
            for (keel::i64 k = s; k <= e; ++k)
                od[pos++] = x.elemAsDouble(keel::as<keel::usize>(k - 1));
        }
        return detail::narrowDoubleToSingleIf(keel::move(out), xSingle, mr);
    }
    // Cell-array form.
    Value out = Value::cell(N, N == 0 ? 0 : 1, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 s = std::max<keel::i64>(1, roiStart(roi, i));
        const keel::i64 e = std::min<keel::i64>(L, roiEnd(roi, i));
        const keel::i64 segLen = (e >= s) ? (e - s + 1) : 0;
        if (xComplex) {
            Value seg = segLen == 0
                ? Value::complexMatrix(0, 1, mr)
                : Value::complexMatrix(keel::as<keel::usize>(segLen), 1, mr);
            if (segLen > 0) {
                keel::c64 *sd = seg.complexDataMut();
                const keel::c64 *xd = x.complexData();
                for (keel::i64 k = 0; k < segLen; ++k)
                    sd[k] = xd[s - 1 + k];
            }
            out.cellAt(i) = keel::move(seg);
        } else {
            Value seg = Value::matrix(keel::as<keel::usize>(segLen),
                                      segLen == 0 ? 0 : 1,
                                      ValueType::DOUBLE, mr);
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

// ── sigrangebinmask ───────────────────────────────────────────────────
// MATLAB signature: sigrangebinmask(x, bound).
//   bound is scalar  → mask where x > bound  (default Relationship='above')
//   bound is 2-vec   → mask where bound(1) <= x <= bound(2)
//                      (default Relationship='inside', closed interval)
// 'Relationship' / 'IntervalType' name-value args deferred (KNOWN GAP).
namespace {
// Common shape-preserving LOGICAL output allocator.
Value allocLogicalLike(const Value &x, keel::Allocator *mr)
{
    const keel::usize L = x.numel();
    if (x.dims().rows() == 1)
        return Value::matrix(1, L, ValueType::LOGICAL, mr);
    return Value::matrix(L, L == 0 ? 0 : 1, ValueType::LOGICAL, mr);
}
} // anon

Value sigrangebinmask(const Value &x, keel::f64 threshold,
                      keel::Allocator *mr)
{
    Value out = allocLogicalLike(x, mr);
    const keel::usize L = x.numel();
    if (L == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < L; ++i)
        od[i] = (x.elemAsDouble(i) > threshold) ? 1 : 0;
    return out;
}

Value sigrangebinmask(const Value &x, keel::f64 lo, keel::f64 hi,
                      keel::Allocator *mr)
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
