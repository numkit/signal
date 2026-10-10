// toolboxes/signal/src/digital_filtering/poly_utils_reg.cpp
//
// Register half of the signal poly_utils builtins: the CallContext wrappers
// delegating to the engine-free compute in poly_utils.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <nk/signal/digital_filtering/poly_utils.hpp>

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

void polyscale_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "polyscale");
    outs[0] = polyscale(args[0], args[1], ctx.engine->resource());
}

void polystab_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "polystab");
    outs[0] = polystab(args[0], ctx.engine->resource());
}

} // namespace detail

} // namespace nk::signal
