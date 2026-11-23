// toolboxes/signal/src/measurements/findpeaks_detail.hpp
//
// Private (src-only) helpers shared between the engine-free compute in
// findpeaks.cpp and its CallContext register half in findpeaks_reg.cpp. The
// register wrapper parses the Name-Value options into a PeakOpts and calls
// findpeaksImpl (a raw-buffer core library_api.md keeps off the public
// surface — the public entry is findpeaks()). NOT part of the public API.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/memory.h>
#include <keel/core.h>
#include <numkit/value/value.hpp>

#include <keel/tuple.h>

namespace numkit::signal {

// Parsed findpeaks options (filled by findpeaks_reg from Name-Value pairs).
struct PeakOpts {
    keel::f64 minHeight  = -keel::Limits<keel::f64>::infinity(); // keep if v > minHeight
    keel::f64 threshold  = 0.0;     // keep if min(v-left, v-right) >= threshold
    keel::f64 minDist    = 0.0;     // in location units; <=0 => disabled
    bool   useMinDist = false;
    keel::f64 minProm    = 0.0;     // MinPeakProminence: keep if prominence >= minProm
    bool   useMinProm = false;
    keel::i64   nPeaks     = -1;      // -1 => unlimited
    keel::i32    sortMode   = 0;       // 0 none(location), 1 ascend(height), 2 descend(height)
    keel::i32    widthRef   = 0;       // WidthReference: 0 'halfprom' (default), 1 'halfheight'
    keel::f64 minWidth   = 0.0;     // MinPeakWidth: keep if width >= minWidth
    keel::f64 maxWidth   = keel::Limits<keel::f64>::infinity(); // MaxPeakWidth: keep if width <= maxWidth
    bool   useWidthFilter = false;
};

// Core detection + filtering. locArr == nullptr => locations are 1-based
// sample indices; otherwise locArr[i] is the location of sample i.
// Returns (heights, locations, widths, prominences).
keel::Tuple<Value, Value, Value, Value>
findpeaksImpl(const Value &x, const keel::f64 *locArr, const PeakOpts &opt,
              keel::Allocator *mr);

} // namespace numkit::signal
