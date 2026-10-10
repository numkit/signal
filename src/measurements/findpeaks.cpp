// toolboxes/signal/src/measurements/findpeaks.cpp
//
// findpeaks — strict local maxima with MATLAB-compatible Name-Value
// options (MinPeakHeight, Threshold, MinPeakDistance, NPeaks, SortStr)
// plus the location form findpeaks(Y,X) / findpeaks(Y,Fs).

#include <nk/signal/measurements/findpeaks.hpp>
#include "findpeaks_detail.hpp"

#include <nk/value/value.hpp>
#include <nk/value/scratch.hpp>
#include <nk/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>

namespace nk::signal {

namespace {

// Topographic prominence of the peak at sample `pi` (height `h`): scan
// outward until a strictly-higher sample (or the signal end) bounds the
// peak on each side; the base is the higher of the two interval minima.
// Fills lb/rb with the (inclusive) interval bounds for the width search.
double peakProminence(const double *x, std::size_t n, std::size_t pi, double h,
                      std::size_t &lb, std::size_t &rb)
{
    double leftMin = std::numeric_limits<double>::infinity();
    lb = 0;
    for (std::size_t j = pi; j-- > 0;) {
        if (x[j] > h) { lb = j; break; }
        leftMin = std::min(leftMin, x[j]);
    }
    double rightMin = std::numeric_limits<double>::infinity();
    rb = n - 1;
    for (std::size_t j = pi + 1; j < n; ++j) {
        if (x[j] > h) { rb = j; break; }
        rightMin = std::min(rightMin, x[j]);
    }
    const double base = std::max(leftMin, rightMin);
    return h - base;
}

// Peak width per 'WidthReference' — both references reverse-engineered
// against R2025b (2026-09-17; the derivation with the full probe table is
// bugs/closed/signal/findpeaks-widthreference.md):
//
//  * 'halfprom' (default): crossings of h - prominence/2 interpolated
//    over the full prominence interval [lb, rb]. The crossing always
//    exists there: ref > base >= each side's interval minimum.
//
//  * 'halfheight': crossings of h/2 searched on each side's
//    NON-INCREASING walk from the peak (stops at the first strict rise
//    or the array end — a deeper sample beyond a rise does NOT become
//    the bound). The first sample <= ref interpolates the crossing on
//    the segment; a sample exactly EQUAL to ref lands on that sample.
//    When the walk never reaches ref (reference below the whole descent,
//    e.g. a peak on a high pedestal), the bound is the LEFTMOST index of
//    the first floor plateau: on the right the first flat sample, on the
//    left the far end of the flat — the reference's asymmetric
//    no-crossing width (its pedestal width 6, not the naive span 8).
double peakWidth(const double *x, std::size_t n, std::size_t pi, double h,
                 double prom, std::size_t lb, std::size_t rb,
                 const double *locArr, int widthRef)
{
    auto pos = [&](std::size_t i) {
        return locArr ? locArr[i] : static_cast<double>(i + 1);
    };

    if (widthRef == 1) {
        const double ref = h / 2.0;
        // Left side: walk left while non-increasing. plateau tracks the
        // leftmost index of the floor run (updated on EVERY non-rise step
        // so it extends across the flat to its far end).
        double lcross = pos(pi);
        {
            std::size_t plateau = pi;
            bool crossed = false;
            for (std::size_t k = pi; k-- > 0;) {
                if (x[k] > x[k + 1])
                    break; // strict rise — walk ends
                if (x[k] <= ref) {
                    const double f = (ref - x[k]) / (x[k + 1] - x[k]); // 0 on touch
                    lcross = pos(k) + f * (pos(k + 1) - pos(k));
                    crossed = true;
                    break;
                }
                plateau = k;
            }
            if (!crossed)
                lcross = pos(plateau);
        }
        // Right side: mirrored walk; plateau tracks only strict drops, so
        // it stays at the FIRST flat sample.
        double rcross = pos(pi);
        {
            std::size_t plateau = pi;
            bool crossed = false;
            for (std::size_t k = pi + 1; k < n; ++k) {
                if (x[k] > x[k - 1])
                    break; // strict rise — walk ends
                if (x[k] <= ref) {
                    const double f = (ref - x[k]) / (x[k - 1] - x[k]); // 0 on touch
                    rcross = pos(k) - f * (pos(k) - pos(k - 1));
                    crossed = true;
                    break;
                }
                if (x[k] < x[k - 1])
                    plateau = k;
            }
            if (!crossed)
                rcross = pos(plateau);
        }
        return rcross - lcross;
    }

    const double ref = h - prom / 2.0;

    double lcross = pos(lb);
    for (std::size_t j = pi; j > lb; --j) {
        if (x[j - 1] < ref && x[j] >= ref) {
            const double f = (ref - x[j - 1]) / (x[j] - x[j - 1]);
            lcross = pos(j - 1) + f * (pos(j) - pos(j - 1));
            break;
        }
    }
    double rcross = pos(rb);
    for (std::size_t j = pi; j < rb; ++j) {
        if (x[j] >= ref && x[j + 1] < ref) {
            const double f = (x[j] - ref) / (x[j] - x[j + 1]);
            rcross = pos(j) + f * (pos(j + 1) - pos(j));
            break;
        }
    }
    return rcross - lcross;
}

} // namespace

// Core detection + filtering. locArr == nullptr => locations are 1-based
// sample indices; otherwise locArr[i] is the location of sample i.
// Returns (heights, locations, widths, prominences). External linkage
// (declared in findpeaks_detail.hpp) so findpeaks_reg can run the core
// after parsing Name-Value options into a PeakOpts.
std::tuple<Value, Value, Value, Value>
findpeaksImpl(const Value &x, const double *locArr, const PeakOpts &opt,
              std::pmr::memory_resource *mr)
{
    ScratchArena scratch(mr);
    auto val  = ScratchVec<double>(&scratch);      // peak heights
    auto loc  = ScratchVec<double>(&scratch);      // peak locations
    auto sidx = ScratchVec<std::size_t>(&scratch); // peak sample indices

    const std::size_t n = x.numel();
    const double *p = (n > 0) ? x.doubleData() : nullptr;
    if (n >= 3) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double v = p[i];
            if (std::isnan(v) || std::isnan(p[i - 1]) || std::isnan(p[i + 1]))
                continue;
            if (!(v > p[i - 1] && v > p[i + 1]))   // strict local maximum
                continue;
            if (!(v > opt.minHeight))              // MinPeakHeight (strictly greater)
                continue;
            const double drop = std::min(v - p[i - 1], v - p[i + 1]);
            if (drop < opt.threshold)              // Threshold (>= vertical drop)
                continue;
            val.push_back(v);
            loc.push_back(locArr ? locArr[i] : static_cast<double>(i + 1));
            sidx.push_back(i);
        }
    }

    std::size_t m = val.size();

    // Prominence (and the interval bounds used for width) per raw peak.
    auto prom = ScratchVec<double>(&scratch);      prom.assign(m, 0.0);
    auto lbnd = ScratchVec<std::size_t>(&scratch); lbnd.assign(m, 0);
    auto rbnd = ScratchVec<std::size_t>(&scratch); rbnd.assign(m, 0);
    for (std::size_t k = 0; k < m; ++k) {
        std::size_t lb = 0, rb = 0;
        prom[k] = peakProminence(p, n, sidx[k], val[k], lb, rb);
        lbnd[k] = lb;
        rbnd[k] = rb;
    }

    auto keep = ScratchVec<unsigned char>(&scratch);
    keep.assign(m, 1u);

    // MinPeakProminence: drop peaks below the prominence threshold.
    if (opt.useMinProm)
        for (std::size_t k = 0; k < m; ++k)
            if (!(prom[k] >= opt.minProm)) keep[k] = 0u;

    // MinPeakDistance: greedy, tallest-first, remove peaks within the distance.
    if (opt.useMinDist && opt.minDist > 0.0 && m > 1) {
        auto order = ScratchVec<std::size_t>(&scratch);
        for (std::size_t k = 0; k < m; ++k) order.push_back(k);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            if (val[a] < val[b]) return false;
            if (val[b] < val[a]) return true;
            return loc[a] < loc[b];               // ties: lower location first
        });
        for (std::size_t oi = 0; oi < m; ++oi) {
            const std::size_t a = order[oi];
            if (!keep[a]) continue;
            for (std::size_t oj = oi + 1; oj < m; ++oj) {
                const std::size_t b = order[oj];
                if (keep[b] && std::fabs(loc[a] - loc[b]) < opt.minDist)
                    keep[b] = 0u;
            }
        }
    }

    // Survivors, still in ascending-location order.
    auto surv = ScratchVec<std::size_t>(&scratch);
    for (std::size_t k = 0; k < m; ++k)
        if (keep[k]) surv.push_back(k);

    // SortStr: reorder by height (ties keep ascending location).
    if (opt.sortMode == 1) {
        std::stable_sort(surv.begin(), surv.end(), [&](std::size_t a, std::size_t b) {
            if (val[a] < val[b]) return true;
            if (val[b] < val[a]) return false;
            return loc[a] < loc[b];
        });
    } else if (opt.sortMode == 2) {
        std::stable_sort(surv.begin(), surv.end(), [&](std::size_t a, std::size_t b) {
            if (val[a] < val[b]) return false;
            if (val[b] < val[a]) return true;
            return loc[a] < loc[b];
        });
    }

    // NPeaks truncation (after sorting).
    std::size_t keepN = surv.size();
    if (opt.nPeaks >= 0 && static_cast<std::size_t>(opt.nPeaks) < keepN)
        keepN = static_cast<std::size_t>(opt.nPeaks);

    // Output orientation matches input column-vector orientation.
    const bool col = (x.dims().ndim() == 2 && x.dims().dim(0) > 1 && x.dims().dim(1) == 1);
    auto makeRowOrCol = [&](std::size_t len) {
        return col ? Value::matrix(len, 1, ValueType::DOUBLE, mr)
                   : Value::matrix(1, len, ValueType::DOUBLE, mr);
    };
    Value vals  = makeRowOrCol(keepN);
    Value locs  = makeRowOrCol(keepN);
    Value wids  = makeRowOrCol(keepN);
    Value proms = makeRowOrCol(keepN);
    double *vp = vals.doubleDataMut();
    double *lp = locs.doubleDataMut();
    double *wp = wids.doubleDataMut();
    double *pp = proms.doubleDataMut();
    for (std::size_t k = 0; k < keepN; ++k) {
        const std::size_t s = surv[k];
        vp[k] = val[s];
        lp[k] = loc[s];
        pp[k] = prom[s];
        wp[k] = peakWidth(p, n, sidx[s], val[s], prom[s], lbnd[s], rbnd[s],
                          locArr, opt.widthRef);
    }
    return std::make_tuple(std::move(vals), std::move(locs),
                           std::move(wids), std::move(proms));
}

// Public no-option entry point (strict local maxima, 1-based indices).
std::tuple<Value, Value>
findpeaks(const Value &x, std::pmr::memory_resource *mr)
{
    auto [v, l, w, p] = findpeaksImpl(x, nullptr, PeakOpts{}, mr);
    (void)w;
    (void)p;
    return std::make_tuple(std::move(v), std::move(l));
}

} // namespace nk::signal
