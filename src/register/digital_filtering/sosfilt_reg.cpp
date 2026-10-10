// toolboxes/signal/src/digital_filtering/sosfilt_reg.cpp
//
// Register half of the signal sosfilt builtins: the CallContext wrappers
// delegating to the engine-free compute in sosfilt.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <nk/signal/digital_filtering/sosfilt.hpp>

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

void sosfilt_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "sosfilt");
    outs[0] = sosfilt(args[0], args[1], ctx.engine->resource());
}

void sosfiltfilt_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "sosfiltfilt");  // absent from the R2025b reference build; historical signature
    outs[0] = sosfiltfilt(args[0], args[1], ctx.engine->resource());
}

} // namespace detail

} // namespace nk::signal
