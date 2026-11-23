// toolboxes/signal/src/spectral_analysis/signal_modeling_reg.cpp
//
// CallContext register half of spectral_analysis/signal_modeling.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::signal {

namespace detail {

void levinson_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "levinson");
    keel::i32 n = -1;
    if (args.size() >= 2 && !args[1].isEmpty())
        n = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "n", "levinson"), 2, "n", "levinson");
    auto [a, e, k] = levinson(args[0], n, ctx.engine->allocator());
    outs[0] = keel::move(a);
    if (nargout > 1) outs[1] = keel::move(e);
    if (nargout > 2) outs[2] = keel::move(k);
}

void rlevinson_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "rlevinson");
    auto [R, k] = rlevinson(args[0], args[1].toScalar(), ctx.engine->allocator());
    outs[0] = keel::move(R);
    if (nargout > 1) outs[1] = keel::move(k);
}

#define NUMKIT_AR_REG(name, fn)                                                     \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize nargout,                    \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 2, 2, #name);                          \
        const keel::i32 p = argcheck::expectedInteger(                                  \
        argcheck::expectedReal(args[1], 2, "p", #name), 2, "p", #name);      \
        auto [a, e, k] = fn(args[0], p, ctx.engine->allocator());                \
        outs[0] = keel::move(a);                                                  \
        if (nargout > 1) outs[1] = keel::move(e);                                 \
        if (nargout > 2) outs[2] = keel::move(k);                                 \
    }

NUMKIT_AR_REG(aryule, aryule)
NUMKIT_AR_REG(arburg, arburg)

#undef NUMKIT_AR_REG

void lpc_reg(keel::ArrayRef<Value> args, keel::usize nargout,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "lpc");
    const keel::i32 p = (args.size() >= 2 && !args[1].isEmpty())
        ? keel::as<keel::i32>(argcheck::roundLength(args[1], 2, "p", "lpc"))
        : keel::as<keel::i32>(args[0].numel() > 0 ? args[0].numel() - 1 : 0);
    auto [a, g] = lpc(args[0], p, ctx.engine->allocator());
    outs[0] = keel::move(a);
    if (nargout > 1) outs[1] = keel::move(g);
}

void ac2poly_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "ac2poly");
    auto [a, e] = ac2poly(args[0], ctx.engine->allocator());
    outs[0] = keel::move(a);
    if (nargout > 1) outs[1] = keel::move(e);
}

void poly2ac_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "poly2ac");
    const keel::f64 e = (args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : 1.0;
    outs[0] = poly2ac(args[0], e, ctx.engine->allocator());
}

void ac2rc_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "ac2rc");
    auto [k, r0] = ac2rc(args[0], ctx.engine->allocator());
    outs[0] = keel::move(k);
    if (nargout > 1) outs[1] = keel::move(r0);
}

void schurrc_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "schurrc");
    outs[0] = schurrc(args[0], ctx.engine->allocator());
}

void rc2ac_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "rc2ac");
    const keel::f64 r0 = (args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : 1.0;
    outs[0] = rc2ac(args[0], r0, ctx.engine->allocator());
}

void poly2rc_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "poly2rc");
    // [k, r0] = poly2rc(a, efinal). efinal (final prediction error) is
    // only needed for the second output; defaults to 0.
    const keel::f64 efinal = (args.size() >= 2 && !args[1].isEmpty())
                              ? args[1].toScalar() : 0.0;
    auto [k, r0] = poly2rc(args[0], efinal, ctx.engine->allocator());
    outs[0] = keel::move(k);
    if (nargout > 1) outs[1] = keel::move(r0);
}

void rc2poly_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "rc2poly");
    // [a, efinal] = rc2poly(k, r0). r0 (zero-lag autocorrelation) is only
    // needed for the second output efinal = r0*prod(1-k.^2); defaults to 1.
    const keel::f64 r0 = (args.size() >= 2 && !args[1].isEmpty())
                          ? args[1].toScalar() : 1.0;
    auto [a, efinal] = rc2poly(args[0], r0, ctx.engine->allocator());
    outs[0] = keel::move(a);
    if (nargout > 1) outs[1] = keel::move(efinal);
}

#define NUMKIT_UNARY_CONV_REG(name)                                                 \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 1, 1, #name);                          \
        outs[0] = name(args[0], ctx.engine->allocator());                         \
    }

NUMKIT_UNARY_CONV_REG(is2rc)
NUMKIT_UNARY_CONV_REG(rc2is)
NUMKIT_UNARY_CONV_REG(lar2rc)
NUMKIT_UNARY_CONV_REG(rc2lar)
NUMKIT_UNARY_CONV_REG(poly2lsf)
NUMKIT_UNARY_CONV_REG(lsf2poly)

#undef NUMKIT_UNARY_CONV_REG

#define NUMKIT_AR2_REG(name, fn)                                                    \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize nargout,                    \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 2, 2, #name);                          \
        const keel::i32 p = argcheck::expectedInteger(                                  \
        argcheck::expectedReal(args[1], 2, "p", #name), 2, "p", #name);      \
        auto [a, e] = fn(args[0], p, ctx.engine->allocator());                   \
        outs[0] = keel::move(a);                                                  \
        if (nargout > 1) outs[1] = keel::move(e);                                 \
    }

NUMKIT_AR2_REG(arcov,  arcov)
NUMKIT_AR2_REG(armcov, armcov)

#undef NUMKIT_AR2_REG

void prony_reg(keel::ArrayRef<Value> args, keel::usize nargout,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "prony");
    const keel::i32 nb = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "nb", "prony"), 2, "nb", "prony");
    const keel::i32 na = (args.size() >= 3 && !args[2].isEmpty())
        ? argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "na", "prony"), 3, "na", "prony")
        : nb;
    auto [b, a] = prony(args[0], nb, na, ctx.engine->allocator());
    outs[0] = keel::move(b);
    if (nargout > 1) outs[1] = keel::move(a);
}

void stmcb_reg(keel::ArrayRef<Value> args, keel::usize nargout,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "stmcb");
    if (args.size() >= 4 && ((!args[1].isScalar() && !args[1].isEmpty()) || args.size() == 6)) {
        // Two-signal form stmcb(y, x, nb, na [, niter [, ai]])
        const keel::i32 nb    = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "nb", "stmcb"), 3, "nb", "stmcb");
        const keel::i32 na    = argcheck::expectedInteger(argcheck::expectedReal(args[3], 4, "na", "stmcb"), 4, "na", "stmcb");
        const keel::i32 niter = (args.size() >= 5 && !args[4].isEmpty())
                              ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[4], 5, "niter", "stmcb"), 5, "niter", "stmcb") : 5;
        const Value ai  = (args.size() >= 6) ? args[5] : Value();
        auto [b, a] = stmcb(args[0], args[1], nb, na, niter, ai, ctx.engine->allocator());
        outs[0] = keel::move(b);
        if (nargout > 1) outs[1] = keel::move(a);
        return;
    }
    const keel::i32 nb    = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "nb", "stmcb"), 2, "nb", "stmcb");
    const keel::i32 na    = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "na", "stmcb"), 3, "na", "stmcb");
    const keel::i32 niter = (args.size() >= 4 && !args[3].isEmpty())
                          ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "niter", "stmcb"), 4, "niter", "stmcb") : 5;
    const Value ai  = (args.size() >= 5) ? args[4] : Value();
    auto [b, a] = stmcb(args[0], Value(), nb, na, niter, ai, ctx.engine->allocator());
    outs[0] = keel::move(b);
    if (nargout > 1) outs[1] = keel::move(a);
}

void corrmtx_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "corrmtx");
    const keel::i32 m = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "m", "corrmtx"), 2, "m", "corrmtx");
    outs[0] = corrmtx(args[0], m, ctx.engine->allocator());
}

void invfreqs_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 9, "invfreqs");
    const keel::i32 nb = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "nb", "invfreqs"), 3, "nb", "invfreqs");
    const keel::i32 na = argcheck::expectedInteger(argcheck::expectedReal(args[3], 4, "na", "invfreqs"), 4, "na", "invfreqs");
    auto [b, a] = invfreqs(args[0], args[1], nb, na, ctx.engine->allocator());
    outs[0] = keel::move(b);
    if (nargout > 1) outs[1] = keel::move(a);
}

void invfreqz_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 9, "invfreqz");
    const keel::i32 nb = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "nb", "invfreqz"), 3, "nb", "invfreqz");
    const keel::i32 na = argcheck::expectedInteger(argcheck::expectedReal(args[3], 4, "na", "invfreqz"), 4, "na", "invfreqz");
    auto [b, a] = invfreqz(args[0], args[1], nb, na, ctx.engine->allocator());
    outs[0] = keel::move(b);
    if (nargout > 1) outs[1] = keel::move(a);
}

} // namespace detail

} // namespace numkit::signal
