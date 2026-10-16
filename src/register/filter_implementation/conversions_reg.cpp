// toolboxes/signal/src/filter_implementation/conversions_reg.cpp
//
// Register half of the signal conversions builtins: the CallContext wrappers
// delegating to the engine-free compute in conversions.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <numkit/signal/filter_implementation/conversions.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
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
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void zp2sos_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 6, "zp2sos");
    const double gain = (args.size() >= 3 && !args[2].isEmpty())
                            ? args[2].toScalar()
                            : 1.0;
    auto *mr = ctx.engine->resource();
    if (nargout >= 2) {
        auto [sos, g] = zp2sosWithGain(args[0], args[1], gain, /*surplusAtOrigin=*/true, mr);
        outs[0] = std::move(sos);
        outs[1] = Value::scalar(g, mr);
    } else {
        outs[0] = zp2sos(args[0], args[1], gain, mr);
    }
}

void tf2sos_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "tf2sos");
    auto *mr = ctx.engine->resource();
    if (nargout >= 2) {
        auto [sos, g] = tf2sosWithGain(args[0], args[1], mr);
        outs[0] = std::move(sos);
        outs[1] = Value::scalar(g, mr);
    } else {
        outs[0] = tf2sos(args[0], args[1], mr);
    }
}

} // namespace detail

} // namespace numkit::signal
