// toolboxes/signal/src/digital_filtering/poly_utils_reg.cpp
//
// Register half of the signal poly_utils builtins: the CallContext wrappers
// delegating to the engine-free compute in poly_utils.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/signal/digital_filtering/poly_utils.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void polyscale_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "polyscale");
    outs[0] = polyscale(args[0], args[1], ctx.engine->allocator());
}

void polystab_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "polystab");
    outs[0] = polystab(args[0], ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
