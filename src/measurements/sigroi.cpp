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

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (argcheck got-clauses)

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../signal_detail.hpp"     // shared argcheck cores
#include "../dsp_helpers.hpp"       // Complex typedef (extractsigroi)

namespace numkit::signal {

namespace {

inline bool truthyAt(const Value &m, size_t i)
{
    if (m.isLogical()) return m.logicalData()[i] != 0;
    return m.elemAsDouble(i) != 0.0;
}

inline int64_t roiStart(const Value &roi, size_t i)
{
    return static_cast<int64_t>(roi.elemAsDouble(i + 0 * roi.dims().rows()));
}
inline int64_t roiEnd(const Value &roi, size_t i)
{
    return static_cast<int64_t>(roi.elemAsDouble(i + 1 * roi.dims().rows()));
}

// Shared roi contract (probed R2025b): an N×2 matrix whose class is one
// of double/single/integer (LOGICAL is rejected by the reference); the
// nk kernels read doubles, so single is an honest gap
// (bugs/opened/signal/single-float-inputs.md).
inline void sigroiRoiArgcheck(const Value &roi, const char *fn)
{
    const ValueType t = roi.type();
    if (t != ValueType::DOUBLE && t != ValueType::SINGLE && t != ValueType::COMPLEX && !isIntegerType(t))
        detail::throwInputType(roi, fn, "roi", 1, "numeric");
    if (roi.dims().ndim() != 2 || roi.dims().cols() != 2)
        throw Error(std::string(fn) + ": roi must be an N×2 matrix of "
                    "[start, end] pairs",
                    0, 0, fn, "", std::string("numkit:") + fn + ":roiCols");
}

} // namespace

// ── binmask2sigroi ────────────────────────────────────────────────────
// Find runs of true values in mask m. Returns N×2 [start, end] matrix
// (1-based, inclusive).
Value binmask2sigroi(const Value &m, std::pmr::memory_resource *mr)
{
    const size_t L = m.numel();
    ScratchArena scratch(mr);
    ScratchVec<int64_t> starts(0, &scratch), ends(0, &scratch);
    bool inRun = false;
    int64_t curStart = 0;
    for (size_t i = 0; i < L; ++i) {
        const bool v = truthyAt(m, i);
        if (v && !inRun) { inRun = true; curStart = static_cast<int64_t>(i + 1); }
        else if (!v && inRun) {
            inRun = false;
            starts.push_back(curStart);
            ends.push_back(static_cast<int64_t>(i));  // i = position past last
        }
    }
    if (inRun) {
        starts.push_back(curStart);
        ends.push_back(static_cast<int64_t>(L));
    }
    const size_t N = starts.size();
    Value out = Value::matrix(N, N == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    double *od = out.doubleDataMut();
    for (size_t i = 0; i < N; ++i) {
        od[i + 0 * N] = static_cast<double>(starts[i]);
        od[i + 1 * N] = static_cast<double>(ends[i]);
    }
    return out;
}

// ── sigroi2binmask ────────────────────────────────────────────────────
// Build a logical column vector of length `len` (auto = max end) with
// true on each [start..end] segment.
Value sigroi2binmask(const Value &roi, int64_t len_user,
                     std::pmr::memory_resource *mr)
{
    sigroiRoiArgcheck(roi, "sigroi2binmask");
    if (len_user == 0)
        throw Error("sigroi2binmask: L must be positive",
                    0, 0, "sigroi2binmask", "", "numkit:sigroi2binmask:lenPositive");
    const size_t N = roi.dims().rows();
    int64_t len = len_user;
    if (len < 0) {
        len = 0;
        for (size_t i = 0; i < N; ++i)
            len = std::max(len, roiEnd(roi, i));
    }
    Value out = Value::matrix(static_cast<size_t>(len), len == 0 ? 0 : 1,
                              ValueType::LOGICAL, mr);
    if (len == 0) return out;
    uint8_t *od = out.logicalDataMut();
    std::fill(od, od + len, uint8_t(0));
    for (size_t i = 0; i < N; ++i) {
        const int64_t s = std::max<int64_t>(1, roiStart(roi, i));
        const int64_t e = std::min<int64_t>(len, roiEnd(roi, i));
        for (int64_t k = s; k <= e; ++k) od[k - 1] = 1;
    }
    return out;
}

// ── extendsigroi ──────────────────────────────────────────────────────
// Extend each ROI by Lpre on the left and Lpost on the right (clamping
// start to ≥ 1). End is unbounded (no signal length provided).
Value extendsigroi(const Value &roi, int64_t Lpre, int64_t Lpost,
                   std::pmr::memory_resource *mr)
{
    sigroiRoiArgcheck(roi, "extendsigroi");
    if (Lpre < 0 || Lpost < 0)
        throw Error("extendsigroi: SL and SR must be nonnegative",
                    0, 0, "extendsigroi", "", "numkit:extendsigroi:nonnegative");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const size_t N = roi.dims().rows();
    Value out = Value::matrix(N, N == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (N == 0) return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
    double *od = out.doubleDataMut();
    for (size_t i = 0; i < N; ++i) {
        int64_t s = roiStart(roi, i) - Lpre;
        int64_t e = roiEnd(roi, i)   + Lpost;
        if (s < 1) s = 1;
        od[i + 0 * N] = static_cast<double>(s);
        od[i + 1 * N] = static_cast<double>(e);
    }
    return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
}

// ── shortensigroi ─────────────────────────────────────────────────────
// Inverse of extend: shrink each ROI by Lpre on the left, Lpost on right.
// Drop ROIs that collapse (start > end).
Value shortensigroi(const Value &roi, int64_t Lpre, int64_t Lpost,
                    std::pmr::memory_resource *mr)
{
    sigroiRoiArgcheck(roi, "shortensigroi");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const size_t N = roi.dims().rows();
    ScratchArena scratch(mr);
    ScratchVec<int64_t> starts(0, &scratch), ends(0, &scratch);
    for (size_t i = 0; i < N; ++i) {
        int64_t s = roiStart(roi, i) + Lpre;
        int64_t e = roiEnd(roi, i)   - Lpost;
        if (s <= e) { starts.push_back(s); ends.push_back(e); }
    }
    const size_t M = starts.size();
    Value out = Value::matrix(M, M == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (M == 0) return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
    double *od = out.doubleDataMut();
    for (size_t i = 0; i < M; ++i) {
        od[i + 0 * M] = static_cast<double>(starts[i]);
        od[i + 1 * M] = static_cast<double>(ends[i]);
    }
    return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
}

// ── mergesigroi ───────────────────────────────────────────────────────
// Sort ROIs by start, merge those with gap ≤ sep.
//   sep == 0 → only overlapping ROIs merge
//   sep ≥ 1 → adjacent or near-adjacent merge
Value mergesigroi(const Value &roi, int64_t sep, std::pmr::memory_resource *mr)
{
    sigroiRoiArgcheck(roi, "mergesigroi");
    if (sep < 0)
        throw Error("mergesigroi: S must be nonnegative",
                    0, 0, "mergesigroi", "", "numkit:mergesigroi:nonnegative");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const size_t N = roi.dims().rows();
    ScratchArena scratch(mr);
    ScratchVec<size_t> idx(N, &scratch);
    for (size_t i = 0; i < N; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return roiStart(roi, a) < roiStart(roi, b); });

    ScratchVec<int64_t> starts(0, &scratch), ends(0, &scratch);
    for (size_t k = 0; k < N; ++k) {
        const int64_t s = roiStart(roi, idx[k]);
        const int64_t e = roiEnd(roi, idx[k]);
        if (!starts.empty() && s - ends.back() <= sep + 1) {
            // Merge: extend the previous end if e is larger.
            if (e > ends.back()) ends.back() = e;
        } else {
            starts.push_back(s);
            ends.push_back(e);
        }
    }
    const size_t M = starts.size();
    Value out = Value::matrix(M, M == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (M == 0) return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
    double *od = out.doubleDataMut();
    for (size_t i = 0; i < M; ++i) {
        od[i + 0 * M] = static_cast<double>(starts[i]);
        od[i + 1 * M] = static_cast<double>(ends[i]);
    }
    return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
}

// ── removesigroi ──────────────────────────────────────────────────────
// Drop ROIs whose length (end - start + 1) is ≤ maxLen. Per MATLAB doc:
// "removes signal regions of interest specified in roilims that have a
// length of s samples or less."
Value removesigroi(const Value &roi, int64_t maxLen,
                   std::pmr::memory_resource *mr)
{
    sigroiRoiArgcheck(roi, "removesigroi");
    const bool roiSingle = (roi.type() == ValueType::SINGLE);
    const size_t N = roi.dims().rows();
    ScratchArena scratch(mr);
    ScratchVec<int64_t> starts(0, &scratch), ends(0, &scratch);
    for (size_t i = 0; i < N; ++i) {
        const int64_t s = roiStart(roi, i);
        const int64_t e = roiEnd(roi, i);
        const int64_t len = e - s + 1;
        if (len > maxLen) {
            starts.push_back(s);
            ends.push_back(e);
        }
    }
    const size_t M = starts.size();
    Value out = Value::matrix(M, M == 0 ? 0 : 2, ValueType::DOUBLE, mr);
    if (M == 0) return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
    double *od = out.doubleDataMut();
    for (size_t i = 0; i < M; ++i) {
        od[i + 0 * M] = static_cast<double>(starts[i]);
        od[i + 1 * M] = static_cast<double>(ends[i]);
    }
    return detail::narrowDoubleToSingleIf(std::move(out), roiSingle, mr);
}

// ── extractsigroi ─────────────────────────────────────────────────────
// Default: cell array (one cell per ROI, each a column vector slice).
// concat=true: concatenate all slices into a single column vector.
// Contract (probed R2025b): x must be a non-empty double/single/complex
// VECTOR (logical rejected, matrices rejected); complex slices keep the
// imaginary part.
Value extractsigroi(const Value &x, const Value &roi, bool concat,
                    std::pmr::memory_resource *mr)
{
    sigroiRoiArgcheck(roi, "extractsigroi");
    detail::requireFloatInput(x, "extractsigroi", "x", 1);
    detail::requireVectorInput(x, "extractsigroi", "x", 1);
    const bool xComplex = x.isComplex();
    const bool xSingle = (x.type() == ValueType::SINGLE);
    const size_t N = roi.dims().rows();
    const int64_t L = static_cast<int64_t>(x.numel());
    if (concat) {
        // Compute total length first.
        int64_t totalLen = 0;
        for (size_t i = 0; i < N; ++i) {
            const int64_t s = std::max<int64_t>(1, roiStart(roi, i));
            const int64_t e = std::min<int64_t>(L, roiEnd(roi, i));
            if (e >= s) totalLen += (e - s + 1);
        }
        if (totalLen == 0)
            return xComplex ? Value::complexMatrix(0, 1, mr)
                            : Value::matrix(0, 1, xSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        if (xComplex) {
            auto out = Value::complexMatrix(static_cast<size_t>(totalLen), 1, mr);
            Complex *od = out.complexDataMut();
            const Complex *xd = x.complexData();
            size_t pos = 0;
            for (size_t i = 0; i < N; ++i) {
                const int64_t s = std::max<int64_t>(1, roiStart(roi, i));
                const int64_t e = std::min<int64_t>(L, roiEnd(roi, i));
                for (int64_t k = s; k <= e; ++k)
                    od[pos++] = xd[k - 1];
            }
            return out;
        }
        Value out = Value::matrix(static_cast<size_t>(totalLen),
                                  totalLen == 0 ? 0 : 1,
                                  ValueType::DOUBLE, mr);
        if (totalLen == 0) return detail::narrowDoubleToSingleIf(std::move(out), xSingle, mr);
        double *od = out.doubleDataMut();
        size_t pos = 0;
        for (size_t i = 0; i < N; ++i) {
            const int64_t s = std::max<int64_t>(1, roiStart(roi, i));
            const int64_t e = std::min<int64_t>(L, roiEnd(roi, i));
            for (int64_t k = s; k <= e; ++k)
                od[pos++] = x.elemAsDouble(static_cast<size_t>(k - 1));
        }
        return detail::narrowDoubleToSingleIf(std::move(out), xSingle, mr);
    }
    // Cell-array form.
    Value out = Value::cell(N, N == 0 ? 0 : 1, mr);
    for (size_t i = 0; i < N; ++i) {
        const int64_t s = std::max<int64_t>(1, roiStart(roi, i));
        const int64_t e = std::min<int64_t>(L, roiEnd(roi, i));
        const int64_t segLen = (e >= s) ? (e - s + 1) : 0;
        if (xComplex) {
            Value seg = segLen == 0
                ? Value::complexMatrix(0, 1, mr)
                : Value::complexMatrix(static_cast<size_t>(segLen), 1, mr);
            if (segLen > 0) {
                Complex *sd = seg.complexDataMut();
                const Complex *xd = x.complexData();
                for (int64_t k = 0; k < segLen; ++k)
                    sd[k] = xd[s - 1 + k];
            }
            out.cellAt(i) = std::move(seg);
        } else {
            Value seg = Value::matrix(static_cast<size_t>(segLen),
                                      segLen == 0 ? 0 : 1,
                                      ValueType::DOUBLE, mr);
            if (segLen > 0) {
                double *sd = seg.doubleDataMut();
                for (int64_t k = 0; k < segLen; ++k)
                    sd[k] = x.elemAsDouble(static_cast<size_t>(s - 1 + k));
            }
            out.cellAt(i) = detail::narrowDoubleToSingleIf(std::move(seg), xSingle, mr);
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
Value allocLogicalLike(const Value &x, std::pmr::memory_resource *mr)
{
    const size_t L = x.numel();
    if (x.dims().rows() == 1)
        return Value::matrix(1, L, ValueType::LOGICAL, mr);
    return Value::matrix(L, L == 0 ? 0 : 1, ValueType::LOGICAL, mr);
}
} // anon

Value sigrangebinmask(const Value &x, double threshold,
                      std::pmr::memory_resource *mr)
{
    Value out = allocLogicalLike(x, mr);
    const size_t L = x.numel();
    if (L == 0) return out;
    uint8_t *od = out.logicalDataMut();
    for (size_t i = 0; i < L; ++i)
        od[i] = (x.elemAsDouble(i) > threshold) ? 1 : 0;
    return out;
}

Value sigrangebinmask(const Value &x, double lo, double hi,
                      std::pmr::memory_resource *mr)
{
    Value out = allocLogicalLike(x, mr);
    const size_t L = x.numel();
    if (L == 0) return out;
    uint8_t *od = out.logicalDataMut();
    for (size_t i = 0; i < L; ++i) {
        const double v = x.elemAsDouble(i);
        od[i] = (v >= lo && v <= hi) ? 1 : 0;
    }
    return out;
}

} // namespace numkit::signal
