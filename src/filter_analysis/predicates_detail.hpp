// toolboxes/signal/src/filter_analysis/predicates_detail.hpp
//
// Private (src-only) helpers shared between the engine-free compute in
// predicates.cpp and its CallContext register half in predicates_reg.cpp. NOT
// part of the public signal API — small polynomial-symmetry helpers the
// register wrappers (islinphase/isfir/…) reuse.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/array.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit::signal {

// Strip trailing (near-)zero coefficients from a polynomial Value → array.
keel::Array<keel::f64> trimTrailingZeros(const Value &p);

// True if v is (anti)symmetric: v[i] == scale * v[n-1-i] within tol.
bool isSymmetric(keel::ArrayRef<keel::f64> v, keel::f64 scale = 1.0,
                 keel::f64 tol = 1e-9);

} // namespace numkit::signal
