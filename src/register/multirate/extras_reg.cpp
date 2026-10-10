// toolboxes/signal/src/multirate/extras_reg.cpp
//
// Register half of the signal extras builtins: the CallContext wrappers
// delegating to the engine-free compute in extras.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/signal/multirate/extras.hpp>

#include <numkit/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

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

namespace numkit::signal {

namespace detail {

void upfirdn_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "upfirdn");
    const size_t p = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "p", "upfirdn"), 3, "p", "upfirdn"));
    const size_t q = (args.size() >= 4)
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[3], 4, "q", "upfirdn"), 4, "q", "upfirdn"))
        : 1;
    outs[0] = upfirdn(args[0], args[1], p, q, ctx.engine->resource());
}

void interp_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "interp");
    const size_t r = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "r", "interp"), 2, "r", "interp"));
    const size_t n = (args.size() >= 3)
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[2], 3, "n", "interp"), 3, "n", "interp"))
        : 4;
    const double alpha = (args.size() >= 4) ? args[3].toScalar() : 0.5;
    outs[0] = interp(args[0], r, n, alpha, ctx.engine->resource());
}

void intfilt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "intfilt");
    const size_t r = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[0], 1, "r", "intfilt"), 1, "r", "intfilt"));
    const size_t n = (args.size() >= 2)
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[1], 2, "n", "intfilt"), 2, "n", "intfilt"))
        : 4;
    const double alpha = (args.size() >= 3) ? args[2].toScalar() : 0.5;
    outs[0] = intfilt(r, n, alpha, ctx.engine->resource());
}

void fftfilt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "fftfilt");
    const size_t nfft = (args.size() >= 3)
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[2], 3, "nfft", "fftfilt"), 3, "nfft", "fftfilt"))
        : 0;
    outs[0] = fftfilt(args[0], args[1], nfft, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
