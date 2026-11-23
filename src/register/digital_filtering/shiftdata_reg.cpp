// toolboxes/signal/src/digital_filtering/shiftdata_reg.cpp
//
// Register half of the signal shiftdata builtins: the CallContext wrappers
// delegating to the engine-free compute in shiftdata.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/signal/digital_filtering/shiftdata.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>


namespace numkit::signal {

namespace detail {

void shiftdata_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "shiftdata");
    keel::i32 dim = 0;
    if (args.size() >= 2 && !args[1].isEmpty())
        dim = argcheck::dimArg(args[1], "shiftdata");
    auto [shifted, permV, nshV] = shiftdata(args[0], dim, ctx.engine->allocator());
    outs[0] = shifted;
    if (nargout >= 2 && outs.size() >= 2) outs[1] = permV;
    if (nargout >= 3 && outs.size() >= 3) outs[2] = nshV;
}

void unshiftdata_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "unshiftdata");
    outs[0] = unshiftdata(args[0], args[1], args[2], ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
