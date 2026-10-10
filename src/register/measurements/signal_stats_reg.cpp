// toolboxes/signal/src/measurements/signal_stats_reg.cpp
//
// CallContext register half of measurements/signal_stats.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/plugin.hpp>
#include <numkit/signal/measurements/signal_stats.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::signal {

namespace detail {

static int dimFromArg(Span<const Value> args, const char *fn)
{
    return (args.size() >= 2) ? argcheck::dimArg(args[1], fn) : 0;
}

void rms_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "rms");
    outs[0] = rms(args[0], dimFromArg(args, "rms"), ctx.engine->resource());
}

void rssq_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "rssq");
    outs[0] = rssq(args[0], dimFromArg(args, "rssq"), ctx.engine->resource());
}

void peak2peak_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "peak2peak");
    outs[0] = peak2peak(args[0], dimFromArg(args, "peak2peak"), ctx.engine->resource());
}

void peak2rms_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "peak2rms");
    outs[0] = peak2rms(args[0], dimFromArg(args, "peak2rms"), ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
