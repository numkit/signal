// toolboxes/signal/src/digital_filtering/filter_reg.cpp
//
// Register half of the signal filter builtins: the CallContext wrappers
// delegating to the engine-free compute in filter.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory;
// the [y, zf] tail used to re-implement the DF2T pipeline here and now
// lives in compute behind filter_argcheck/runFilter (argcheck-separation
// pattern, dev-docs/todo/argcheck_separation.md).

#include <nk/signal/digital_filtering/filter.hpp>

#include <nk/core/engine.hpp>   // CallContext, Span, ctx.engine->resource()
#include <nk/value/argcheck.hpp>
#include <nk/value/error.hpp>
#include <nk/value/value.hpp>

namespace nk::signal {

namespace detail {

void filter_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "filter");
    // zi is the 4th argument when present and value-like; char/string
    // fourth arguments belong to the dim-argument form nk does not ship,
    // so they are treated as absent (previous behaviour).
    const bool hasZi = (args.size() >= 4 && !args[3].isEmpty()
                        && !args[3].isChar() && !args[3].isString());
    auto *mr = ctx.engine->resource();

    // Fast path: plain filter(b,a,x) with one output.
    if (!hasZi && nargout <= 1) {
        outs[0] = filter(args[0], args[1], args[2], mr);
        return;
    }
    // [y, zf] = filter(b,a,x[,zi]): thread initial conditions through the
    // DF2T state and return the final state (one column per signal).
    const Value zi = hasZi ? args[3] : Value();
    FilterState st = filter(args[0], args[1], args[2], zi, mr);
    outs[0] = std::move(st.y);
    if (nargout > 1)
        outs[1] = std::move(st.zf);
}

void filtfilt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "filtfilt");
    outs[0] = filtfilt(args[0], args[1], args[2], ctx.engine->resource());
}

} // namespace detail

} // namespace nk::signal
