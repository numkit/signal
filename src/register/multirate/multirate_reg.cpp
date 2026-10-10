// toolboxes/signal/src/multirate/multirate_reg.cpp
//
// Register half of the signal multirate builtins: the CallContext wrappers
// delegating to the engine-free compute in multirate.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <nk/signal/multirate/multirate.hpp>

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

void downsample_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "downsample");
    const size_t n = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "downsample"), 2, "n", "downsample"));
    size_t phase = 0;
    if (args.size() >= 3)
        phase = static_cast<size_t>(argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "phase", "downsample"), 3, "phase", "downsample"));
    // phase < n contract lives in the compute argcheck.
    outs[0] = downsample(args[0], n, phase, ctx.engine->resource());
}

void upsample_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "upsample");
    const size_t n = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "upsample"), 2, "n", "upsample"));
    size_t phase = 0;
    if (args.size() >= 3)
        phase = static_cast<size_t>(argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "phase", "upsample"), 3, "phase", "upsample"));
    // phase < n contract lives in the compute argcheck.
    outs[0] = upsample(args[0], n, phase, ctx.engine->resource());
}

void decimate_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "decimate");
    outs[0] = decimate(args[0], static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "q", "decimate"), 2, "q", "decimate")),
        ctx.engine->resource());
}

void resample_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "resample");  // reference also accepts n/beta/method + name-value (unimplemented)
    outs[0] = resample(args[0],
        static_cast<size_t>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "p", "resample"), 2, "p", "resample")),
        static_cast<size_t>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[2], 3, "q", "resample"), 3, "q", "resample")),
        ctx.engine->resource());
}

} // namespace detail

} // namespace nk::signal
