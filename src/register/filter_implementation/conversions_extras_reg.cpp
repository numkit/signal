// toolboxes/signal/src/filter_implementation/conversions_extras_reg.cpp
//
// Register half of the signal conversions_extras builtins: the CallContext wrappers
// delegating to the engine-free compute in conversions_extras.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/signal/filter_implementation/conversions_extras.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void sos2tf_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sos2tf");
    const keel::f64 g = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    auto [b, a] = sos2tf(args[0], g, ctx.engine->allocator());
    outs[0] = keel::move(b);
    if (nargout > 1) outs[1] = keel::move(a);
}

void sos2zp_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sos2zp");
    const keel::f64 g = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    auto [z, p, gain] = sos2zp(args[0], g, ctx.engine->allocator());
    outs[0] = keel::move(z);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = Value::scalar(gain, ctx.engine->allocator());
}

void tf2zpk_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "tf2zpk");
    auto *mr = ctx.engine->allocator();
    Value a = (args.size() >= 2 && !args[1].isEmpty()) ? args[1] : Value::scalar(1.0, mr);
    auto [z, p, gain] = tf2zpk(args[0], a, mr);
    outs[0] = keel::move(z);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = Value::scalar(gain, mr);
}

void tf2ss_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "tf2ss");
    auto [A, B, C, D] = tf2ss(args[0], args[1], ctx.engine->allocator());
    outs[0] = keel::move(A);
    if (nargout > 1) outs[1] = keel::move(B);
    if (nargout > 2) outs[2] = keel::move(C);
    if (nargout > 3) outs[3] = keel::move(D);
}

void ss2tf_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "ss2tf");
    const keel::f64 D = (args[3].numel() == 0) ? 0.0 : args[3].elemAsDouble(0);
    auto [b, a] = ss2tf(args[0], args[1], args[2], D, ctx.engine->allocator());
    outs[0] = keel::move(b);
    if (nargout > 1) outs[1] = keel::move(a);
}

void ss2zp_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "ss2zp");
    const keel::f64 D = (args[3].numel() == 0) ? 0.0 : args[3].elemAsDouble(0);
    auto [z, p, gain] = ss2zp(args[0], args[1], args[2], D, ctx.engine->allocator());
    outs[0] = keel::move(z);
    if (nargout > 1) outs[1] = keel::move(p);
    if (nargout > 2) outs[2] = Value::scalar(gain, ctx.engine->allocator());
}

void zp2ss_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "zp2ss");
    auto [A, B, C, D] = zp2ss(args[0], args[1], args[2].toScalar(), ctx.engine->allocator());
    outs[0] = keel::move(A);
    if (nargout > 1) outs[1] = keel::move(B);
    if (nargout > 2) outs[2] = keel::move(C);
    if (nargout > 3) outs[3] = keel::move(D);
}

void sos2ss_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sos2ss");
    const keel::f64 g = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    auto [A, B, C, D] = sos2ss(args[0], g, ctx.engine->allocator());
    outs[0] = keel::move(A);
    if (nargout > 1) outs[1] = keel::move(B);
    if (nargout > 2) outs[2] = keel::move(C);
    if (nargout > 3) outs[3] = keel::move(D);
}

void ss2sos_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 7, "ss2sos");
    const keel::f64 D = (args[3].numel() == 0) ? 0.0 : args[3].elemAsDouble(0);
    outs[0] = ss2sos(args[0], args[1], args[2], D, ctx.engine->allocator());
}

void ctf2zp_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "ctf2zp");
    auto *mr = ctx.engine->allocator();
    Value DEN = (args.size() >= 2) ? args[1] : Value::scalar(1.0, mr);
    const Value &SV = (args.size() >= 3) ? args[2] : Value::Empty;
    auto [Z, P, k] = ctf2zp(args[0], DEN, SV, mr);
    outs[0] = keel::move(Z);
    if (nargout >= 2 && outs.size() >= 2) outs[1] = keel::move(P);
    if (nargout >= 3 && outs.size() >= 3) outs[2] = Value::scalar(k, mr);
}

void scaleFilterSections_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "scaleFilterSections");
    outs[0] = scaleFilterSections(args[0], args[1], ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
