// toolboxes/signal/src/digital_filtering/shiftdata_reg.cpp
//
// Register half of the signal shiftdata builtins: the CallContext wrappers
// delegating to the engine-free compute in shiftdata.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <nk/signal/digital_filtering/shiftdata.hpp>

#include <nk/core/engine.hpp>   // CallContext, Span, ctx.engine->resource()
#include <nk/value/value.hpp>
#include <nk/value/argcheck.hpp>
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

namespace nk::signal {

namespace detail {

void shiftdata_reg(Span<const Value> args, size_t nargout,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "shiftdata");
    int dim = 0;
    if (args.size() >= 2 && !args[1].isEmpty())
        dim = argcheck::dimArg(args[1], "shiftdata");
    auto [shifted, permV, nshV] = shiftdata(args[0], dim, ctx.engine->resource());
    outs[0] = shifted;
    if (nargout >= 2 && outs.size() >= 2) outs[1] = permV;
    if (nargout >= 3 && outs.size() >= 3) outs[2] = nshV;
}

void unshiftdata_reg(Span<const Value> args, size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "unshiftdata");
    outs[0] = unshiftdata(args[0], args[1], args[2], ctx.engine->resource());
}

} // namespace detail

} // namespace nk::signal
