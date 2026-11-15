// toolboxes/signal/src/filter_analysis/unwrap.cpp
//
// unwrap — split out from transforms/.cpp. The hilbert/envelope
// pair lives in transforms/hilbert.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/signal/filter_analysis/unwrap.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    const double *p = phase.doubleData();

    auto r = createLike(phase, ValueType::DOUBLE, mr);
    double *out = r.doubleDataMut();
    if (n == 0)
        return r;
    out[0] = p[0];
    for (keel::usize i = 1; i < n; ++i) {
        double d = p[i] - p[i - 1];
        d = d - 2.0 * M_PI * std::round(d / (2.0 * M_PI));
        out[i] = out[i - 1] + d;
    }
    return r;
}

} // namespace numkit::signal
