// toolboxes/signal/src/multirate/multirate_reg.cpp
//
// Register half of the signal multirate builtins: the CallContext wrappers
// delegating to the engine-free compute in multirate.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/signal/multirate/multirate.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

void downsample_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "downsample");
    const keel::usize n = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "downsample"), 2, "n", "downsample"));
    keel::usize phase = 0;
    if (args.size() >= 3)
        phase = keel::as<keel::usize>(argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "phase", "downsample"), 3, "phase", "downsample"));
    // phase < n contract lives in the compute argcheck.
    outs[0] = downsample(args[0], n, phase, ctx.engine->allocator());
}

void upsample_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "upsample");
    const keel::usize n = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "upsample"), 2, "n", "upsample"));
    keel::usize phase = 0;
    if (args.size() >= 3)
        phase = keel::as<keel::usize>(argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "phase", "upsample"), 3, "phase", "upsample"));
    // phase < n contract lives in the compute argcheck.
    outs[0] = upsample(args[0], n, phase, ctx.engine->allocator());
}

void decimate_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "decimate");
    outs[0] = decimate(args[0], keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "q", "decimate"), 2, "q", "decimate")),
        ctx.engine->allocator());
}

void resample_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "resample");
    const keel::usize p = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "p", "resample"), 2, "p", "resample"));
    const keel::usize q = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "q", "resample"), 3, "q", "resample"));
    int n_order = 10;
    keel::f64 beta = 5.0;
    if (args.size() >= 4 && args[3].isNumeric() && args[3].numel() == 1)
        n_order = argcheck::expectedInteger(
            argcheck::expectedReal(args[3], 4, "n", "resample"), 4, "n", "resample");
    if (args.size() >= 5 && args[4].isNumeric() && args[4].numel() == 1)
        beta = argcheck::expectedReal(args[4], 5, "beta", "resample");
    outs[0] = resample(args[0], p, q, n_order, beta, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
