// toolboxes/signal/src/filter_analysis/unwrap.cpp
//
// unwrap — delegated to keel::signal::unwrap_phase.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/filter_analysis/unwrap.hpp>

#include <numkit/value/value.hpp>
#include <numkit/ops/helpers.hpp>

namespace numkit::signal {

namespace {

void unwrap_argcheck(const Value &phase)
{
    (void)phase;
}

} // namespace

Value unwrap(const Value &phase, keel::Allocator *mr)
{
    unwrap_argcheck(phase);
    const keel::usize n = phase.numel();
    auto r = createLike(phase, ValueType::DOUBLE, mr);
    if (n == 0)
        return r;
    (void)keel::signal::unwrap_phase({phase.doubleData(), n}, {r.doubleDataMut(), n});
    return r;
}

} // namespace numkit::signal
