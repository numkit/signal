// toolboxes/signal/src/digital_filtering/spec_driven_reg.cpp
//
// Register half of the signal spec_driven builtins: the CallContext wrappers
// delegating to the engine-free compute in spec_driven.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <numkit/signal/digital_filtering/spec_driven.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
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

// MATLAB lowpass/highpass/bandpass/bandstop: when fs is omitted, the
// cutoffs are interpreted as already normalized to Nyquist, equivalent
// to fs = 2 (so fpass in [0, 1] maps to [0, pi] in normalized rad/sample).
void lowpass_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 9, "lowpass");
    const double fs = (args.size() >= 3) ? args[2].toScalar() : 2.0;
    const int order = (args.size() >= 4)
        ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "order", "lowpass"), 4, "order", "lowpass") : 8;
    outs[0] = lowpass(args[0], args[1].toScalar(), fs, order, ctx.engine->resource());
}

void highpass_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 9, "highpass");
    const double fs = (args.size() >= 3) ? args[2].toScalar() : 2.0;
    const int order = (args.size() >= 4)
        ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "order", "highpass"), 4, "order", "highpass") : 8;
    outs[0] = highpass(args[0], args[1].toScalar(), fs, order, ctx.engine->resource());
}

void bandpass_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 9, "bandpass");
    if (args[1].numel() != 2)
        throw Error("bandpass: cutoff must be a 2-element [flo fhi]",
                     0, 0, "bandpass", "", "numkit:bandpass:badCutoff");
    const double fs = (args.size() >= 3) ? args[2].toScalar() : 2.0;
    const int order = (args.size() >= 4)
        ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "order", "bandpass"), 4, "order", "bandpass") : 8;
    outs[0] = bandpass(args[0], args[1].elemAsDouble(0), args[1].elemAsDouble(1), fs, order, ctx.engine->resource());
}

void bandstop_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 9, "bandstop");
    if (args[1].numel() != 2)
        throw Error("bandstop: cutoff must be a 2-element [flo fhi]",
                     0, 0, "bandstop", "", "numkit:bandstop:badCutoff");
    const double fs = (args.size() >= 3) ? args[2].toScalar() : 2.0;
    const int order = (args.size() >= 4)
        ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "order", "bandstop"), 4, "order", "bandstop") : 8;
    outs[0] = bandstop(args[0], args[1].elemAsDouble(0), args[1].elemAsDouble(1), fs, order, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
