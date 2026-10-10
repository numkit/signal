// toolboxes/signal/src/spectral_analysis/signal_modeling_reg.cpp
//
// CallContext register half of spectral_analysis/signal_modeling.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/plugin.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>
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

void levinson_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "levinson");
    int n = -1;
    if (args.size() >= 2 && !args[1].isEmpty())
        n = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "n", "levinson"), 2, "n", "levinson");
    auto [a, e, k] = levinson(args[0], n, ctx.engine->resource());
    outs[0] = std::move(a);
    if (nargout > 1) outs[1] = std::move(e);
    if (nargout > 2) outs[2] = std::move(k);
}

void rlevinson_reg(Span<const Value> args, size_t nargout,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "rlevinson");
    auto [R, k] = rlevinson(args[0], args[1].toScalar(), ctx.engine->resource());
    outs[0] = std::move(R);
    if (nargout > 1) outs[1] = std::move(k);
}

#define NUMKIT_AR_REG(name, fn)                                                     \
    void name##_reg(Span<const Value> args, size_t nargout,                    \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 2, 2, #name);                          \
        const int p = argcheck::expectedInteger(                                  \
        argcheck::expectedReal(args[1], 2, "p", #name), 2, "p", #name);      \
        auto [a, e, k] = fn(args[0], p, ctx.engine->resource());                \
        outs[0] = std::move(a);                                                  \
        if (nargout > 1) outs[1] = std::move(e);                                 \
        if (nargout > 2) outs[2] = std::move(k);                                 \
    }

NUMKIT_AR_REG(aryule, aryule)
NUMKIT_AR_REG(arburg, arburg)

#undef NUMKIT_AR_REG

void lpc_reg(Span<const Value> args, size_t nargout,
             Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "lpc");
    const int p = static_cast<int>(argcheck::roundLength(args[1], 2, "p", "lpc"));
    auto [a, g] = lpc(args[0], p, ctx.engine->resource());
    outs[0] = std::move(a);
    if (nargout > 1) outs[1] = std::move(g);
}

void ac2poly_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "ac2poly");
    auto [a, e] = ac2poly(args[0], ctx.engine->resource());
    outs[0] = std::move(a);
    if (nargout > 1) outs[1] = std::move(e);
}

void poly2ac_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "poly2ac");
    const double e = (args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : 1.0;
    outs[0] = poly2ac(args[0], e, ctx.engine->resource());
}

void ac2rc_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "ac2rc");
    auto [k, r0] = ac2rc(args[0], ctx.engine->resource());
    outs[0] = std::move(k);
    if (nargout > 1) outs[1] = std::move(r0);
}

void schurrc_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "schurrc");
    outs[0] = schurrc(args[0], ctx.engine->resource());
}

void rc2ac_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "rc2ac");
    outs[0] = rc2ac(args[0], args[1].toScalar(), ctx.engine->resource());
}

void poly2rc_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "poly2rc");
    // [k, r0] = poly2rc(a, efinal). efinal (final prediction error) is
    // only needed for the second output; defaults to 0.
    const double efinal = (args.size() >= 2 && !args[1].isEmpty())
                              ? args[1].toScalar() : 0.0;
    auto [k, r0] = poly2rc(args[0], efinal, ctx.engine->resource());
    outs[0] = std::move(k);
    if (nargout > 1) outs[1] = std::move(r0);
}

void rc2poly_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "rc2poly");
    // [a, efinal] = rc2poly(k, r0). r0 (zero-lag autocorrelation) is only
    // needed for the second output efinal = r0*prod(1-k.^2); defaults to 1.
    const double r0 = (args.size() >= 2 && !args[1].isEmpty())
                          ? args[1].toScalar() : 1.0;
    auto [a, efinal] = rc2poly(args[0], r0, ctx.engine->resource());
    outs[0] = std::move(a);
    if (nargout > 1) outs[1] = std::move(efinal);
}

#define NUMKIT_UNARY_CONV_REG(name)                                                 \
    void name##_reg(Span<const Value> args, size_t /*nargout*/,                \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 1, 1, #name);                          \
        outs[0] = name(args[0], ctx.engine->resource());                         \
    }

NUMKIT_UNARY_CONV_REG(is2rc)
NUMKIT_UNARY_CONV_REG(rc2is)
NUMKIT_UNARY_CONV_REG(lar2rc)
NUMKIT_UNARY_CONV_REG(rc2lar)
NUMKIT_UNARY_CONV_REG(poly2lsf)
NUMKIT_UNARY_CONV_REG(lsf2poly)

#undef NUMKIT_UNARY_CONV_REG

#define NUMKIT_AR2_REG(name, fn)                                                    \
    void name##_reg(Span<const Value> args, size_t nargout,                    \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 2, 2, #name);                          \
        const int p = argcheck::expectedInteger(                                  \
        argcheck::expectedReal(args[1], 2, "p", #name), 2, "p", #name);      \
        auto [a, e] = fn(args[0], p, ctx.engine->resource());                   \
        outs[0] = std::move(a);                                                  \
        if (nargout > 1) outs[1] = std::move(e);                                 \
    }

NUMKIT_AR2_REG(arcov,  arcov)
NUMKIT_AR2_REG(armcov, armcov)

#undef NUMKIT_AR2_REG

void prony_reg(Span<const Value> args, size_t nargout,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "prony");
    const int nb = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "nb", "prony"), 2, "nb", "prony");
    const int na = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "na", "prony"), 3, "na", "prony");
    auto [b, a] = prony(args[0], nb, na, ctx.engine->resource());
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(a);
}

void stmcb_reg(Span<const Value> args, size_t nargout,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "stmcb");
    // Two-signal form stmcb(y, x, nb, na) — x is a vector, not a scalar order.
    if (!args[1].isScalar())
        throw Error("stmcb: the two-signal form stmcb(y, x, ...) is not yet supported",
                    0, 0, "stmcb", "", "nk:stmcb:twosignal");
    const int nb    = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "nb", "stmcb"), 2, "nb", "stmcb");
    const int na    = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "na", "stmcb"), 3, "na", "stmcb");
    const int niter = (args.size() >= 4 && !args[3].isEmpty())
                          ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "niter", "stmcb"), 4, "niter", "stmcb") : 5;
    if (args.size() >= 5)
        throw Error("stmcb: an explicit ai initial estimate is not yet supported",
                    0, 0, "stmcb", "", "nk:stmcb:ai");
    auto [b, a] = stmcb(args[0], nb, na, niter, ctx.engine->resource());
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(a);
}

void corrmtx_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "corrmtx");
    const int m = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "m", "corrmtx"), 2, "m", "corrmtx");
    outs[0] = corrmtx(args[0], m, ctx.engine->resource());
}

void invfreqs_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 9, "invfreqs");
    const int nb = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "nb", "invfreqs"), 3, "nb", "invfreqs");
    const int na = argcheck::expectedInteger(argcheck::expectedReal(args[3], 4, "na", "invfreqs"), 4, "na", "invfreqs");
    auto [b, a] = invfreqs(args[0], args[1], nb, na, ctx.engine->resource());
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(a);
}

void invfreqz_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 9, "invfreqz");
    const int nb = argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "nb", "invfreqz"), 3, "nb", "invfreqz");
    const int na = argcheck::expectedInteger(argcheck::expectedReal(args[3], 4, "na", "invfreqz"), 4, "na", "invfreqz");
    auto [b, a] = invfreqz(args[0], args[1], nb, na, ctx.engine->resource());
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(a);
}

} // namespace detail

} // namespace numkit::signal
