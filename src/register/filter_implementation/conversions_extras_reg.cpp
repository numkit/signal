// toolboxes/signal/src/filter_implementation/conversions_extras_reg.cpp
//
// Register half of the signal conversions_extras builtins: the CallContext wrappers
// delegating to the engine-free compute in conversions_extras.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <nk/signal/filter_implementation/conversions_extras.hpp>

#include <nk/core/engine.hpp>   // CallContext, Span, ctx.engine->resource()
#include <nk/value/value.hpp>
#include <nk/value/error.hpp>
#include <nk/value/scratch.hpp>
#include <nk/value/span.hpp>

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
#include <nk/value/argcheck.hpp>

namespace nk::signal {

namespace detail {

void sos2tf_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sos2tf");
    const double g = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    auto [b, a] = sos2tf(args[0], g, ctx.engine->resource());
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(a);
}

void sos2zp_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sos2zp");
    const double g = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    auto [z, p, gain] = sos2zp(args[0], g, ctx.engine->resource());
    outs[0] = std::move(z);
    if (nargout > 1) outs[1] = std::move(p);
    if (nargout > 2) outs[2] = Value::scalar(gain, ctx.engine->resource());
}

void tf2zpk_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "tf2zpk");
    auto [z, p, gain] = tf2zpk(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(z);
    if (nargout > 1) outs[1] = std::move(p);
    if (nargout > 2) outs[2] = Value::scalar(gain, ctx.engine->resource());
}

void tf2ss_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "tf2ss");
    auto [A, B, C, D] = tf2ss(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(A);
    if (nargout > 1) outs[1] = std::move(B);
    if (nargout > 2) outs[2] = std::move(C);
    if (nargout > 3) outs[3] = std::move(D);
}

void ss2tf_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "ss2tf");
    const double D = (args[3].numel() == 0) ? 0.0 : args[3].elemAsDouble(0);
    auto [b, a] = ss2tf(args[0], args[1], args[2], D, ctx.engine->resource());
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(a);
}

void ss2zp_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "ss2zp");
    const double D = (args[3].numel() == 0) ? 0.0 : args[3].elemAsDouble(0);
    auto [z, p, gain] = ss2zp(args[0], args[1], args[2], D, ctx.engine->resource());
    outs[0] = std::move(z);
    if (nargout > 1) outs[1] = std::move(p);
    if (nargout > 2) outs[2] = Value::scalar(gain, ctx.engine->resource());
}

void zp2ss_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "zp2ss");
    auto [A, B, C, D] = zp2ss(args[0], args[1], args[2].toScalar(), ctx.engine->resource());
    outs[0] = std::move(A);
    if (nargout > 1) outs[1] = std::move(B);
    if (nargout > 2) outs[2] = std::move(C);
    if (nargout > 3) outs[3] = std::move(D);
}

void sos2ss_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sos2ss");
    const double g = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    auto [A, B, C, D] = sos2ss(args[0], g, ctx.engine->resource());
    outs[0] = std::move(A);
    if (nargout > 1) outs[1] = std::move(B);
    if (nargout > 2) outs[2] = std::move(C);
    if (nargout > 3) outs[3] = std::move(D);
}

void ss2sos_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 7, "ss2sos");
    const double D = (args[3].numel() == 0) ? 0.0 : args[3].elemAsDouble(0);
    outs[0] = ss2sos(args[0], args[1], args[2], D, ctx.engine->resource());
}

void ctf2zp_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "ctf2zp");
    auto *mr = ctx.engine->resource();
    Value DEN = (args.size() >= 2) ? args[1] : Value::scalar(1.0, mr);
    const Value &SV = (args.size() >= 3) ? args[2] : Value::Empty;
    auto [Z, P, k] = ctf2zp(args[0], DEN, SV, mr);
    outs[0] = std::move(Z);
    if (nargout >= 2 && outs.size() >= 2) outs[1] = std::move(P);
    if (nargout >= 3 && outs.size() >= 3) outs[2] = Value::scalar(k, mr);
}

void scaleFilterSections_reg(Span<const Value> args, size_t /*nargout*/,
                              Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "scaleFilterSections");
    outs[0] = scaleFilterSections(args[0], args[1], ctx.engine->resource());
}

} // namespace detail

} // namespace nk::signal
