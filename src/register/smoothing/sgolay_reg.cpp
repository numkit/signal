// toolboxes/signal/src/smoothing/sgolay_reg.cpp
//
// Register half of the signal sgolay builtins: the CallContext wrappers
// delegating to the engine-free compute in sgolay.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <numkit/signal/smoothing/sgolay.hpp>

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

void sgolay_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "sgolay");
    auto *mr = ctx.engine->resource();
    const int order    = argcheck::expectedInteger(argcheck::expectedReal(args[0], 1, "order", "sgolay"), 1, "order", "sgolay");
    const int framelen = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "framelen", "sgolay"), 2, "framelen", "sgolay");
    outs[0] = sgolay(order, framelen, mr);
    // [B,G] = sgolay(...): G is the framelen Ã— (order+1) differentiation-
    // filter matrix (MATLAB's second output).
    if (nargout >= 2)
        outs[1] = sgolayDiff(order, framelen, mr);
}

void sgolayfilt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "sgolayfilt");
    auto *res = ctx.engine->resource();
    const int order    = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "order", "sgolayfilt"), 2, "order", "sgolayfilt");
    const int framelen = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "framelen", "sgolayfilt"), 3, "framelen", "sgolayfilt");
    // sgolayfilt(x, order, framelen, weights, dim): weights (4th) and dim
    // (5th) are optional; an empty [] in either slot selects the default.
    const int dim = (args.size() >= 5 && !args[4].isEmpty())
                        ? argcheck::dimArg(args[4], "sgolayfilt") : 0;
    const Value weights = (args.size() >= 4)
                              ? args[3]
                              : Value::matrix(0, 0, ValueType::DOUBLE, res);
    outs[0] = sgolayfilt(args[0], order, framelen, weights, dim, res);
}

} // namespace detail

} // namespace numkit::signal
