// toolboxes/signal/src/filter_design/analog_filters_reg.cpp
//
// CallContext register half of filter_design/analog_filters.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/ops/poly_helpers.hpp>
#include <numkit/signal/filter_design/analog_filters.hpp>
#include <numkit/signal/filter_implementation/conversions_extras.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

#define NUMKIT_PROTO0_REG(name)                                                     \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize nargout,                    \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 1, 1, #name);                          \
        const keel::i32 N = argcheck::expectedPositiveInteger(                     \
        argcheck::expectedReal(args[0], 1, "N", #name), 1, "N", #name);        \
        auto [z, p, k] = name(N, ctx.engine->allocator());                       \
        outs[0] = std::move(z);                                                  \
        if (nargout > 1) outs[1] = std::move(p);                                 \
        if (nargout > 2) outs[2] = std::move(k);                                 \
    }

#define NUMKIT_PROTO1_REG(name)                                                     \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize nargout,                    \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 2, 2, #name);                          \
        const keel::i32 N = argcheck::expectedPositiveInteger(                     \
        argcheck::expectedReal(args[0], 1, "N", #name), 1, "N", #name);        \
        const keel::f64 r = args[1].toScalar();                                    \
        auto [z, p, k] = name(N, r, ctx.engine->allocator());                    \
        outs[0] = std::move(z);                                                  \
        if (nargout > 1) outs[1] = std::move(p);                                 \
        if (nargout > 2) outs[2] = std::move(k);                                 \
    }

NUMKIT_PROTO0_REG(buttap)
NUMKIT_PROTO0_REG(besselap)
NUMKIT_PROTO1_REG(cheb1ap)
NUMKIT_PROTO1_REG(cheb2ap)

// ellipap takes 3 args (N, Rp, Rs).
void ellipap_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "ellipap");
    const keel::i32 N     = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[0], 1, "N", "ellipap"), 1, "N", "ellipap");
    const keel::f64 Rp = args[1].toScalar();
    const keel::f64 Rs = args[2].toScalar();
    auto [z, p, k] = ellipap(N, Rp, Rs, ctx.engine->allocator());
    outs[0] = std::move(z);
    if (nargout > 1) outs[1] = std::move(p);
    if (nargout > 2) outs[2] = std::move(k);
}

#undef NUMKIT_PROTO0_REG
#undef NUMKIT_PROTO1_REG

// MATLAB's lp2* TF form returns the numerator at its true polynomial degree
// (length = #zeros + 1), NOT zero-padded to the denominator length the way
// zp2tf does. e.g. lp2lp of a Butterworth prototype (no zeros) returns
// bt = [Wo^N] (length 1), while zp2tf would give [0 â€¦ 0 Wo^N]. Strip the
// leading exact-zeros that zp2tf inserted so the TF form matches MATLAB.
// (No-op for lp2hp/lp2bs, whose numerators are already full length.)
static Value lp2StripLeadingZeros(const Value &b, keel::Allocator *mr)
{
    const keel::usize n = b.numel();
    keel::usize s = 0;
    while (s + 1 < n && b.elemAsDouble(s) == 0.0) ++s;
    if (s == 0) return b;
    auto r = Value::matrix(1, n - s, ValueType::DOUBLE, mr);
    keel::f64 *d = r.doubleDataMut();
    for (keel::usize i = s; i < n; ++i) d[i - s] = b.elemAsDouble(i);
    return r;
}

// MATLAB lp2lp/lp2hp accept BOTH:
//   [zt, pt, kt] = lp2lp(z, p, k, Wo)   -- ZPK form (4 args)
//   [bt, at]     = lp2lp(b, a, Wo)      -- TF form  (3 args)
// Same for lp2hp. The TF form is identified by the 3-arg call and
// dispatches via tf2zpk -> ZPK lp2lp -> zp2tf.
#define NUMKIT_LP2X1_REG(name, fn)                                                  \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize nargout,                    \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 3, 5, #name);                          \
        auto *mr = ctx.engine->allocator();                                       \
        if (args.size() == 3) {                                                  \
            /* TF form: (b, a, Wo) -> (bt, at). */                               \
            const keel::f64 Wo = args[2].toScalar();                                \
            auto [z0, p0, k0] = tf2zpk(args[0], args[1], mr);                    \
            auto [zt, pt, kt] = fn(z0, p0, k0, Wo, mr);                          \
            auto [bt, at] = numkit::builtin::zp2tf(zt, pt, kt.toScalar(), mr);           \
            bt = lp2StripLeadingZeros(bt, mr);                                   \
            outs[0] = std::move(bt);                                             \
            if (nargout > 1) outs[1] = std::move(at);                            \
            return;                                                              \
        }                                                                        \
        const keel::f64 Wo = args[3].toScalar();                                   \
        auto [z, p, k] = fn(args[0], args[1], args[2].toScalar(), Wo, mr);      \
        outs[0] = std::move(z);                                                  \
        if (nargout > 1) outs[1] = std::move(p);                                 \
        if (nargout > 2) outs[2] = std::move(k);                                 \
    }

NUMKIT_LP2X1_REG(lp2lp, lp2lp)
NUMKIT_LP2X1_REG(lp2hp, lp2hp)

#undef NUMKIT_LP2X1_REG

// MATLAB lp2bp/lp2bs accept BOTH:
//   [zt, pt, kt] = lp2bp(z, p, k, Wo, Bw)   -- ZPK form (5 args)
//   [bt, at]     = lp2bp(b, a, Wo, Bw)      -- TF form  (4 args)
#define NUMKIT_LP2X2_REG(name, fn)                                                  \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize nargout,                    \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 4, 6, #name);                          \
        auto *mr = ctx.engine->allocator();                                       \
        if (args.size() == 4) {                                                  \
            /* TF form: (b, a, Wo, Bw) -> (bt, at). */                           \
            const keel::f64 Wo = args[2].toScalar();                                \
            const keel::f64 Bw = args[3].toScalar();                                \
            auto [z0, p0, k0] = tf2zpk(args[0], args[1], mr);                    \
            auto [zt, pt, kt] = fn(z0, p0, k0, Wo, Bw, mr);                      \
            auto [bt, at] = numkit::builtin::zp2tf(zt, pt, kt.toScalar(), mr);           \
            bt = lp2StripLeadingZeros(bt, mr);                                   \
            outs[0] = std::move(bt);                                             \
            if (nargout > 1) outs[1] = std::move(at);                            \
            return;                                                              \
        }                                                                        \
        if (args.size() < 5)                                                     \
            throw Error(#name ": requires (z, p, k, Wo, Bw) or (b, a, Wo, Bw)", \
                         0, 0, #name, "", "numkit:" #name ":nargin");                 \
        const keel::f64 Wo = args[3].toScalar();                                   \
        const keel::f64 Bw = args[4].toScalar();                                   \
        auto [z, p, k] = fn(args[0], args[1], args[2].toScalar(), Wo, Bw, mr);  \
        outs[0] = std::move(z);                                                  \
        if (nargout > 1) outs[1] = std::move(p);                                 \
        if (nargout > 2) outs[2] = std::move(k);                                 \
    }

NUMKIT_LP2X2_REG(lp2bp, lp2bp)
NUMKIT_LP2X2_REG(lp2bs, lp2bs)

#undef NUMKIT_LP2X2_REG

void bilinear_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "bilinear");
    const keel::f64 fs = args[2].toScalar();
    const keel::f64 fp = (args.size() >= 4 && !args[3].isEmpty()) ? args[3].toScalar() : 0.0;
    auto [bd, ad] = bilinear(args[0], args[1], fs, fp, ctx.engine->allocator());
    outs[0] = std::move(bd);
    if (nargout > 1) outs[1] = std::move(ad);
}

void impinvar_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "impinvar");
    const keel::f64 fs  = (args.size() >= 3 && !args[2].isEmpty()) ? args[2].toScalar() : 1.0;
    const keel::f64 tol = (args.size() >= 4 && !args[3].isEmpty()) ? args[3].toScalar() : 1e-3;
    auto [bd, ad] = impinvar(args[0], args[1], fs, tol, ctx.engine->allocator());
    outs[0] = std::move(bd);
    if (nargout > 1) outs[1] = std::move(ad);
}

namespace {

// freqs(b, a) auto grid â€” the classic freqint auto-ranging algorithm
// (Andy Grace 1990, rev 1996; still live inside MATLAB's freqs â€” every
// endpoint AND interior point verified against R2025b, see
// bugs/closed/signal/freqs-two-arg-auto-w.md):
//
//   ez    = poles (imag >= 0) ++ zeros (|z| < 1e5, imag >= 0)
//           (upper half plane only; no poles -> single pole at -1000,
//           which DERIVES the documented [100, 1e4] default).
//   low   = round(log10(0.1 * min(|Re ez| + 2*Im ez)) - 0.5)   // round half away
//   high  = round(log10(max(3*|Re ez| + 1.5*Im ez)) + 0.5)
//   base  = logspace(low, high, 200 + (P - Z) + [10 if any zero has
//           |Im| < |Re|])                                     // the LONG grid
//   refine: for each oscillatory root (Im > |Re|, descending |Re|):
//           window [max(0.8*Im - 3|Re|, 10^low), 1.2*Im + 4*Re|] â€”
//           base points inside are REPLACED by a denser logspace
//           (count + npts2 points, npts2 = 2 + 8/ceil(|P-Z+eps|/10)).
//   final = the long grid linearly interpolated in log10 at 200
//           equally spaced INDEX positions (caller-side resample).
//
// Roots at the origin count as "integrators" (|ez| < 1e-10) and are
// shifted by 1 before the log so the extremes stay finite. All ops
// mirror MATLAB doubles exactly (std::round is half-away-from-zero,
// linspace/logspace constructed as 10.^(lo + i*d)).

struct FreqsRoot {
    keel::f64 re, im;
};

keel::Array<keel::f64> freqsAutoGridVec(const Value &b, const Value &a, keel::i32 npts)
{
    using numkit::ops::polyRootsDurandKerner;
    constexpr keel::f64 kEps = 2.220446049250313e-16;
    keel::Allocator *mr = keel::default_allocator();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    auto rootsOf = [&](const Value &p) -> keel::Array<FreqsRoot> {
        keel::Array<FreqsRoot> out;
        if (!p.isComplex() && p.numel() >= 1) {
            auto r = polyRootsDurandKerner(p.doubleData(), p.numel(), scratch.allocator());
            for (const auto &z : r)
                out.push_back({z.real(), z.imag()});
        }
        return out;
    };
    keel::Array<FreqsRoot> ep = rootsOf(a); // poles (full set)
    keel::Array<FreqsRoot> tz = rootsOf(b); // zeros (full set)
    if (ep.empty())
        ep.push_back({-1000.0, 0.0});

    // ez: upper-half roots; zeros filtered to |z| < 1e5.
    keel::Array<FreqsRoot> ez;
    for (auto &r : ep)
        if (r.im >= 0.0)
            ez.push_back(r);
    for (auto &r : tz)
        if (keel::abs(keel::c64(r.re, r.im)) < 1e5 && r.im >= 0.0)
            ez.push_back(r);

    auto log10v = [](keel::f64 x) { return keel::log10(x); };
    keel::f64 loNum = 1e300, hiNum = 0.0;
    for (auto &r : ez) {
        const bool integ = keel::abs(keel::c64(r.re, r.im)) < 1e-10;
        loNum = std::min(loNum, keel::abs(r.re + (integ ? 1.0 : 0.0)) + 2.0 * r.im);
        hiNum = std::max(hiNum, 3.0 * keel::abs(r.re + (integ ? 1.0 : 0.0)) + 1.5 * r.im);
    }
    const keel::i32 low = keel::as<keel::i32>(keel::round(log10v(0.1 * loNum) - 0.5));
    const keel::i32 high = keel::as<keel::i32>(keel::round(log10v(hiNum) + 0.5));

    const keel::i32 diffzp = keel::as<keel::i32>(ep.size()) - keel::as<keel::i32>(tz.size());
    bool anyRealDomZero = false;
    for (auto &r : tz)
        if (keel::abs(r.im) < keel::abs(r.re))
            anyRealDomZero = true;
    const keel::i32 nLong = 200 + diffzp + (anyRealDomZero ? 10 : 0);

    auto logspace = [](keel::f64 l, keel::f64 u, keel::i32 n) {
        keel::Array<keel::f64> v(n);
        const keel::f64 d = (u - l) / (n - 1);
        for (keel::i32 i = 0; i < n; ++i)
            v[i] = keel::pow(10.0, l + i * d);
        v[n - 1] = keel::pow(10.0, u); // linspace forces the last point exactly
        return v;
    };
    keel::Array<keel::f64> w = logspace(low, high, nLong); // the base LONG grid

    // Oscillatory refinement (Im > |Re|), descending |Re| (stable).
    keel::Array<keel::i32> osc;
    for (keel::usize k = 0; k < ez.size(); ++k)
        if (ez[k].im > keel::abs(ez[k].re))
            osc.push_back(keel::as<keel::i32>(k));
    std::stable_sort(osc.begin(), osc.end(),
                     [&](keel::i32 x, keel::i32 y) { return keel::abs(ez[x].re) > keel::abs(ez[y].re); });
    const keel::f64 npts2 = 2.0 + 8.0 / keel::ceil(keel::abs((diffzp + kEps) / 10.0));

    keel::Array<keel::f64> f = w.clone();
    keel::Array<keel::f64> z;
    for (keel::i32 k : osc) {
        const keel::f64 r1 = std::max(0.8 * ez[k].im - 3.0 * keel::abs(ez[k].re),
                                   keel::pow(10.0, keel::as<keel::f64>(low)));
        const keel::f64 r2 = 1.2 * ez[k].im + 4.0 * keel::abs(ez[k].re);
        auto outside = [&](keel::f64 x) { return x > r2 || x < r1; };
        z.resize(std::remove_if(z.begin(), z.end(), [&](keel::f64 x) { return !outside(x); }) - z.begin());
        f.resize(std::remove_if(f.begin(), f.end(), [&](keel::f64 x) { return !outside(x); }) - f.begin());
        keel::i64 cnt = 0;
        for (keel::f64 x : w)
            if (x >= r1 && x <= r2)
                ++cnt;
        auto seg = logspace(keel::log10(r1), keel::log10(r2), keel::as<keel::i32>(cnt + npts2));
        z.append(seg.as_ref());
    }
    f.append(z.as_ref());
    std::sort(f.begin(), f.end()); // w_long

    // Resample to exactly 200 points at evenly spaced INDEX positions,
    // linear in log10 (the caller-side interp1 of the classic code).
    keel::Array<keel::f64> lw(f.size());
    for (keel::usize i = 0; i < f.size(); ++i)
        lw[i] = keel::log10(f[i]);
    const keel::f64 dxi = keel::as<keel::f64>(f.size() - 1) / (npts - 1);
    keel::Array<keel::f64> out(npts);
    for (keel::i32 i = 0; i < npts; ++i) {
        // 1-based index into lw; the last sample is EXACTLY the last knot
        // (linspace-forced), where interp1 returns log10(w_long(end)) as-is
        // and the pow round-trip gives back the exact decade endpoint.
        const keel::f64 x = (i == npts - 1) ? keel::as<keel::f64>(f.size()) : 1.0 + i * dxi;
        const keel::usize j = keel::as<keel::usize>(x) - 1;
        const keel::f64 t = x - (j + 1);
        // At the last sample (x == f.size(), t == 0) the upper knot j+1
        // is one past the end â€” clamp it so the read stays in bounds
        // (the value is unchanged: t == 0 kills the difference). The
        // unclamped read was benign in Release but tripped the MSVC
        // Debug iterator assert (bugs/closed/signal/freqs-autogrid-debug-oob).
        const keel::usize j2 = std::min(j + 1, lw.size() - 1);
        const keel::f64 lv = lw[j] + t * (lw[j2] - lw[j]);
        out[i] = keel::pow(10.0, lv);
    }
    return out;
}

Value freqsAutoGrid(const Value &b, const Value &a, keel::Allocator *mr,
                    keel::i32 npts = 200)
{
    auto v = freqsAutoGridVec(b, a, npts);
    Value w = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    keel::f64 *wd = w.doubleDataMut();
    for (keel::i32 i = 0; i < npts; ++i)
        wd[i] = v[i];
    return w;
}

} // namespace

void freqs_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    if (args.size() == 2) {
        // freqs(b, a): auto grid; the no-output PLOT form is a separate
        // piece of the same filed gap.
        auto w = freqsAutoGrid(args[0], args[1], ctx.engine->allocator());
        outs[0] = freqs(args[0], args[1], w, ctx.engine->allocator());
        if (nargout > 1)
            outs[1] = std::move(w);
        return;
    }
    argcheck::checkNargin(args.size(), 3, 3, "freqs");
    // MATLAB freqs(b, a, n): a SCALAR third argument is the NUMBER of
    // auto-spaced points (same freqint grid machinery as the two-arg
    // form, resampled to n); only a VECTOR means "exactly these
    // frequencies" (bugs/closed/signal/freqs-scalar-w).
    if (args[2].numel() == 1 && !args[2].isChar() && !args[2].isString()) {
        const keel::i32 npts = std::max(2, argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "n", "freqs"), 3, "n", "freqs"));
        auto w = freqsAutoGrid(args[0], args[1], ctx.engine->allocator(), npts);
        outs[0] = freqs(args[0], args[1], w, ctx.engine->allocator());
        if (nargout > 1)
            outs[1] = std::move(w);
        return;
    }
    outs[0] = freqs(args[0], args[1], args[2], ctx.engine->allocator());
    if (nargout > 1)
        outs[1] = args[2];
}

} // namespace detail

} // namespace numkit::signal
