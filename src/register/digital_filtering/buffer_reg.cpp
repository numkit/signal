// toolboxes/signal/src/digital_filtering/buffer_reg.cpp
//
// Register half of the signal buffer builtins: the CallContext wrappers
// delegating to the engine-free compute in buffer.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <numkit/signal/digital_filtering/buffer.hpp>

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

void buffer_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "buffer");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "buffer"), 2, "n", "buffer");
    int p = 0;
    if (args.size() >= 3 && !args[2].isEmpty())
        p = argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "p", "buffer"), 3, "p", "buffer");
    const Value &opt = (args.size() >= 4) ? args[3] : Value::Empty;
    if (nargout >= 2 && outs.size() >= 2) {
        auto [Y, Z] = buffer2(args[0], n, p, opt, ctx.engine->resource());
        outs[0] = Y;
        outs[1] = Z;
    } else {
        outs[0] = buffer(args[0], n, p, opt, ctx.engine->resource());
    }
}

} // namespace detail

} // namespace numkit::signal
