// toolboxes/signal/src/convolution/extras_reg.cpp
//
// Register half of the signal extras builtins: the CallContext wrappers
// delegating to the engine-free compute in extras.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/signal/convolution/extras.hpp>

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

void cconv_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "cconv");
    const size_t n = (args.size() >= 3)
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[2], 3, "n", "cconv"), 3, "n", "cconv"))
        : 0;
    outs[0] = cconv(args[0], args[1], n, ctx.engine->resource());
}

void convmtx_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "convmtx");
    outs[0] = convmtx(args[0], static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "convmtx"), 2, "n", "convmtx")),
        ctx.engine->resource());
}

void xcorr2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "xcorr2");
    const Value &A = args[0];
    const Value &B = (args.size() >= 2) ? args[1] : args[0];
    outs[0] = xcorr2(A, B, ctx.engine->resource());
}

void finddelay_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "finddelay");
    const long max_lag = (args.size() >= 3)
        ? argcheck::expectedInteger(
              argcheck::expectedReal(args[2], 3, "maxlag", "finddelay"), 3, "maxlag", "finddelay")
        : 0;
    const long d = finddelay(args[0], args[1], max_lag, ctx.engine->resource());
    outs[0] = Value::scalar(static_cast<double>(d), ctx.engine->resource());
}

void alignsignals_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "alignsignals");  // no max: name-value tail in the reference
    const long max_lag = (args.size() >= 3)
        ? argcheck::expectedInteger(
              argcheck::expectedReal(args[2], 3, "maxlag", "alignsignals"), 3, "maxlag", "alignsignals")
        : 0;
    auto [xa, ya] = alignsignals(args[0], args[1], max_lag, ctx.engine->resource());
    outs[0] = std::move(xa);
    if (nargout > 1) outs[1] = std::move(ya);
}

} // namespace detail

} // namespace numkit::signal
