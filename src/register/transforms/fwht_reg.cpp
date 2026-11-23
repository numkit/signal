// toolboxes/signal/src/transforms/fwht_reg.cpp
//
// CallContext register half of transforms/fwht.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/transforms/fwht.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

static void parseFwhtArgs(keel::ArrayRef<Value> args,
                           keel::usize &n, keel::String &ordering)
{
    n = 0;
    ordering.assign("sequency");
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString()) {
            // (x, ordering) â€” n omitted.
            ordering = args[1].toString();
            return;
        }
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "n", "fwht"), 2, "n", "fwht");
    }
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (!(args[2].isChar() || args[2].isString()))
            throw Error("fwht: ordering must be a string",
                         0, 0, "fwht", "", "numkit:fwht:badOrdering");
        ordering = args[2].toString();
    }
}

void fwht_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "fwht");
    keel::usize n; keel::String ordering;
    parseFwhtArgs(args, n, ordering);
    outs[0] = fwht(args[0], n, ordering, ctx.engine->allocator());
}

void ifwht_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "ifwht");
    keel::usize n; keel::String ordering;
    parseFwhtArgs(args, n, ordering);
    outs[0] = ifwht(args[0], n, ordering, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
