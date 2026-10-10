// toolboxes/signal/src/filter_analysis/unwrap_reg.cpp
//
// Register half of the signal unwrap builtins: the CallContext wrappers
// delegating to the engine-free compute in unwrap.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <nk/signal/filter_analysis/unwrap.hpp>

#include <nk/core/engine.hpp>   // CallContext, Span, ctx.engine->resource()
#include <nk/value/value.hpp>
#include <nk/value/error.hpp>
#include <nk/value/scratch.hpp>
#include <nk/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <nk/value/argcheck.hpp>

namespace nk::signal {

namespace detail {

void unwrap_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unwrap");
    outs[0] = unwrap(args[0], ctx.engine->resource());
}

} // namespace detail

} // namespace nk::signal
