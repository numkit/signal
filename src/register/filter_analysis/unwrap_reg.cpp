// toolboxes/signal/src/filter_analysis/unwrap_reg.cpp
//
// Register half of the signal unwrap builtins: the CallContext wrappers
// delegating to the engine-free compute in unwrap.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/signal/filter_analysis/unwrap.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <tuple>
#include <utility>
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void unwrap_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "unwrap");
    outs[0] = unwrap(args[0], ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
