// toolboxes/signal/src/measurements/pulse_metrics_detail.hpp
//
// Private (src-only) helper shared between the engine-free compute in
// pulse_metrics.cpp and its CallContext register half in pulse_metrics_reg.cpp.
// pulseRiseFall is the multi-output worker behind risetime()/falltime(); it
// lives on the compute side (it only needs the file-local state-level helpers)
// but is declared here so the thin risetime_reg/falltime_reg adapters can
// forward to it. NOT part of the public signal API.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>


namespace numkit::signal {

// risetime / falltime worker. Emits up to 5 outputs into `outs`:
//   [R, LT, UT, LL, UL] — duration, lower(10%) crossing time, upper(90%)
//   crossing time, lower/upper reference levels. `rising` picks rise vs fall.
void pulseRiseFall(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                   bool rising, const char *fnname,
                   keel::Allocator *mr);

} // namespace numkit::signal
