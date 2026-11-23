// toolboxes/signal/src/multirate/extras_reg.cpp
//
// Register half of the signal extras builtins: the CallContext wrappers
// delegating to the engine-free compute in extras.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/signal/multirate/extras.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::signal {

namespace detail {

void upfirdn_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "upfirdn");
    const keel::usize p = (args.size() >= 3 && !args[2].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[2], 3, "p", "upfirdn"), 3, "p", "upfirdn"))
        : 1;
    const keel::usize q = (args.size() >= 4 && !args[3].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[3], 4, "q", "upfirdn"), 4, "q", "upfirdn"))
        : 1;
    outs[0] = upfirdn(args[0], args[1], p, q, ctx.engine->allocator());
}

void interp_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "interp");
    const keel::usize r = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "r", "interp"), 2, "r", "interp"));
    const keel::usize n = (args.size() >= 3)
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[2], 3, "n", "interp"), 3, "n", "interp"))
        : 4;
    const keel::f64 alpha = (args.size() >= 4) ? args[3].toScalar() : 0.5;
    outs[0] = interp(args[0], r, n, alpha, ctx.engine->allocator());
}

void intfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "intfilt");
    const keel::usize r = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[0], 1, "r", "intfilt"), 1, "r", "intfilt"));
    const keel::usize n = (args.size() >= 2)
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[1], 2, "n", "intfilt"), 2, "n", "intfilt"))
        : 4;
    const keel::f64 alpha = (args.size() >= 3) ? args[2].toScalar() : 0.5;
    outs[0] = intfilt(r, n, alpha, ctx.engine->allocator());
}

void fftfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "fftfilt");
    const keel::usize nfft = (args.size() >= 3)
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[2], 3, "nfft", "fftfilt"), 3, "nfft", "fftfilt"))
        : 0;
    outs[0] = fftfilt(args[0], args[1], nfft, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
