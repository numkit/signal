// toolboxes/signal/src/measurements/findpeaks.cpp
//
// findpeaks delegating peak detection, prominence, width, and filtering to keel::signal::findpeaks.

#include <keel/arena.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/measurements/findpeaks.hpp>
#include "findpeaks_detail.hpp"

#include <numkit/value/value.hpp>

#include <algorithm>
#include <keel/tuple.h>

namespace numkit::signal {

keel::Tuple<Value, Value, Value, Value>
findpeaksImpl(const Value &x, const keel::f64 *locArr, const PeakOpts &opt,
              keel::Allocator *mr) {
    keel::signal::PeakOptions kopt;
    kopt.min_height       = opt.minHeight;
    kopt.threshold        = opt.threshold;
    kopt.use_min_prom     = opt.useMinProm;
    kopt.min_prom         = opt.minProm;
    kopt.use_width_filter = opt.useWidthFilter;
    kopt.min_width        = opt.minWidth;
    kopt.max_width        = opt.maxWidth;
    kopt.use_min_dist     = opt.useMinDist;
    kopt.min_dist         = opt.minDist;
    kopt.max_peaks        = opt.nPeaks;
    kopt.sort_mode        = (opt.sortMode == 1) ? keel::signal::PeakSort::kAscend
                          : (opt.sortMode == 2) ? keel::signal::PeakSort::kDescend
                                                : keel::signal::PeakSort::kNone;
    kopt.width_ref        = (opt.widthRef == 1) ? keel::signal::PeakWidthRef::kHalfHeight
                                                : keel::signal::PeakWidthRef::kHalfProm;

    const keel::usize n = x.numel();
    keel::ArrayRef<keel::f64> xRef(n > 0 ? x.doubleData() : nullptr, n);
    keel::ArrayRef<keel::f64> locRef(locArr ? locArr : nullptr, locArr ? n : 0);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::signal::PeakResult res;
    (void)keel::signal::findpeaks(xRef, locRef, kopt, res, scratch.allocator());

    const keel::usize keepN = res.values.size();
    const bool col = (x.dims().ndim() == 2 && x.dims().dim(0) > 1 && x.dims().dim(1) == 1);
    auto makeRowOrCol = [&](keel::usize len) {
        return col ? Value::matrix(len, 1, ValueType::DOUBLE, mr)
                   : Value::matrix(1, len, ValueType::DOUBLE, mr);
    };
    Value vals  = makeRowOrCol(keepN);
    Value locs  = makeRowOrCol(keepN);
    Value wids  = makeRowOrCol(keepN);
    Value proms = makeRowOrCol(keepN);
    if (keepN > 0) {
        std::copy(res.values.begin(), res.values.end(), vals.doubleDataMut());
        std::copy(res.locations.begin(), res.locations.end(), locs.doubleDataMut());
        std::copy(res.widths.begin(), res.widths.end(), wids.doubleDataMut());
        std::copy(res.prominences.begin(), res.prominences.end(), proms.doubleDataMut());
    }
    return keel::make_tuple(keel::move(vals), keel::move(locs),
                           keel::move(wids), keel::move(proms));
}

keel::Tuple<Value, Value> findpeaks(const Value &x, keel::Allocator *mr) {
    auto [v, l, w, p] = findpeaksImpl(x, nullptr, PeakOpts{}, mr);
    (void)w;
    (void)p;
    return keel::make_tuple(keel::move(v), keel::move(l));
}

} // namespace numkit::signal
