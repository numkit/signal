// toolboxes/signal/src/spectral_analysis/signal_modeling.cpp
//
// Parametric signal modelling: Levinson-Durbin, Yule-Walker / Burg /
// covariance AR, Prony / Steiglitz-McBride, corrmtx, LSF, invfreqs/z
// (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

keel::Array<keel::f64> readVec(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> v(n, mr);
    for (keel::usize i = 0; i < n; ++i) v[i] = x.elemAsDouble(i);
    return v;
}

keel::Array<keel::c64> readComplexVec(const Value &v, keel::Allocator *mr)
{
    const keel::usize n = v.numel();
    keel::Array<keel::c64> out(n, mr);
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

Value rowVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

Value colVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

void stmcb_argcheck(const Value &y, const Value &x, keel::i32 nb, keel::i32 na, const Value &ai)
{
    if (nb < 0)
        throw Error("stmcb: numerator order nb must be non-negative",
                    0, 0, "stmcb", "", "numkit:stmcb:InvalidNb");
    if (na < 0)
        throw Error("stmcb: denominator order na must be non-negative",
                    0, 0, "stmcb", "", "numkit:stmcb:InvalidNa");
    if (!x.isEmpty() && x.numel() != y.numel())
        throw Error("stmcb: input and output signal vectors must have the same length",
                    0, 0, "stmcb", "", "numkit:stmcb:InvalidDimensions");
    if (!ai.isEmpty()) {
        const keel::usize n_ai = ai.numel();
        if (n_ai != keel::as<keel::usize>(na + 1) && n_ai != keel::as<keel::usize>(na))
            throw Error("stmcb: initial denominator estimate ai must have length na+1 (or na)",
                        0, 0, "stmcb", "", "numkit:stmcb:InvalidAiLength");
        if (n_ai == keel::as<keel::usize>(na + 1) && ai.elemAsDouble(0) == 0.0)
            throw Error("stmcb: leading coefficient of ai must be nonzero",
                        0, 0, "stmcb", "", "numkit:stmcb:ZeroLeadingCoefficient");
    }
}

void invfreq_argcheck(const Value &H, const Value &w, keel::i32 nb, keel::i32 na, const char *fn)
{
    if (H.numel() != w.numel())
        throw numkit::Error(keel::format("{}: H and w must be the same length", fn));
    const keel::i32 K = keel::as<keel::i32>(H.numel());
    const keel::i32 nv = nb + 1 + na;
    if (K * 2 < nv)
        throw numkit::Error("invfreq: not enough frequency points to fit (nb, na)");
}

} // anonymous

keel::Tuple<Value, Value, Value>
levinson(const Value &r, keel::i32 n, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rv = readVec(r, scratch.allocator());
    keel::signal::LevinsonResult res;
    (void)keel::signal::levinson(rv.as_ref(), n, res, scratch.allocator());
    return keel::make_tuple(rowVec(res.a.as_ref(), mr),
                            Value::scalar(res.e, mr),
                            colVec(res.k.as_ref(), mr));
}

keel::Tuple<Value, Value>
rlevinson(const Value &a, keel::f64 efinal, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto av = readVec(a, scratch.allocator());
    keel::signal::RlevinsonResult res;
    (void)keel::signal::rlevinson(av.as_ref(), efinal, res, scratch.allocator());
    return keel::make_tuple(rowVec(res.r.as_ref(), mr), colVec(res.k.as_ref(), mr));
}

keel::Tuple<Value, Value, Value>
aryule(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    keel::signal::LevinsonResult res;
    (void)keel::signal::aryule(xv.as_ref(), p, res, scratch.allocator());
    return keel::make_tuple(rowVec(res.a.as_ref(), mr),
                            Value::scalar(res.e, mr),
                            colVec(res.k.as_ref(), mr));
}

keel::Tuple<Value, Value, Value>
arburg(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    keel::signal::LevinsonResult res;
    (void)keel::signal::arburg(xv.as_ref(), p, res, scratch.allocator());
    return keel::make_tuple(rowVec(res.a.as_ref(), mr),
                            Value::scalar(res.e, mr),
                            colVec(res.k.as_ref(), mr));
}

keel::Tuple<Value, Value>
lpc(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    auto [a, e, k] = aryule(x, p, mr);
    (void)k;
    return keel::make_tuple(keel::move(a), keel::move(e));
}

keel::Tuple<Value, Value>
ac2poly(const Value &R, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rv = readVec(R, scratch.allocator());
    const keel::i32 p = keel::max(keel::as<keel::i32>(rv.size()) - 1, keel::i32{0});
    keel::signal::LevinsonResult res;
    (void)keel::signal::levinson(rv.as_ref(), p, res, scratch.allocator());
    return keel::make_tuple(rowVec(res.a.as_ref(), mr), Value::scalar(res.e, mr));
}

Value poly2ac(const Value &a, keel::f64 e, keel::Allocator *mr)
{
    auto [Rv, kv] = rlevinson(a, e, mr);
    (void)kv;
    return Rv;
}

keel::Tuple<Value, Value>
ac2rc(const Value &R, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rv = readVec(R, scratch.allocator());
    const keel::i32 p = keel::max(keel::as<keel::i32>(rv.size()) - 1, keel::i32{0});
    keel::signal::LevinsonResult res;
    (void)keel::signal::levinson(rv.as_ref(), p, res, scratch.allocator());
    return keel::make_tuple(colVec(res.k.as_ref(), mr),
                            Value::scalar(rv.empty() ? 0.0 : rv[0], mr));
}

Value schurrc(const Value &R, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rv = readVec(R, scratch.allocator());
    keel::Array<keel::f64> k(scratch.allocator());
    (void)keel::signal::schurrc(rv.as_ref(), k, scratch.allocator());
    return colVec(k.as_ref(), mr);
}

Value rc2ac(const Value &k, keel::f64 r0, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto kv = readVec(k, scratch.allocator());
    keel::Array<keel::f64> r(scratch.allocator());
    (void)keel::signal::rc2ac(kv.as_ref(), r0, r, scratch.allocator());
    return rowVec(r.as_ref(), mr);
}

keel::Tuple<Value, Value>
poly2rc(const Value &a, keel::f64 efinal, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto av = readVec(a, scratch.allocator());
    keel::Array<keel::f64> k(scratch.allocator());
    keel::f64 r0 = 0.0;
    (void)keel::signal::poly2rc(av.as_ref(), efinal, k, r0, scratch.allocator());
    return keel::make_tuple(colVec(k.as_ref(), mr), Value::scalar(r0, mr));
}

keel::Tuple<Value, Value>
rc2poly(const Value &k, keel::f64 r0, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto kv = readVec(k, scratch.allocator());
    keel::Array<keel::f64> a(scratch.allocator());
    keel::f64 efinal = 0.0;
    (void)keel::signal::rc2poly(kv.as_ref(), r0, a, efinal, scratch.allocator());
    return keel::make_tuple(rowVec(a.as_ref(), mr), Value::scalar(efinal, mr));
}

Value is2rc(const Value &is, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(is, scratch.allocator());
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    (void)keel::signal::is2rc(v.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), v.size()));
    return out;
}

Value rc2is(const Value &k, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(k, scratch.allocator());
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    (void)keel::signal::rc2is(v.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), v.size()));
    return out;
}

Value lar2rc(const Value &g, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(g, scratch.allocator());
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    (void)keel::signal::lar2rc(v.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), v.size()));
    return out;
}

Value rc2lar(const Value &k, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(k, scratch.allocator());
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    (void)keel::signal::rc2lar(v.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), v.size()));
    return out;
}

keel::Tuple<Value, Value>
arcov(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    keel::Array<keel::f64> a(scratch.allocator());
    keel::f64 e = 0.0;
    (void)keel::signal::arcov(xv.as_ref(), p, a, e, scratch.allocator());
    return keel::make_tuple(rowVec(a.as_ref(), mr), Value::scalar(e, mr));
}

keel::Tuple<Value, Value>
armcov(const Value &x, keel::i32 p, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    keel::Array<keel::f64> a(scratch.allocator());
    keel::f64 e = 0.0;
    (void)keel::signal::armcov(xv.as_ref(), p, a, e, scratch.allocator());
    return keel::make_tuple(rowVec(a.as_ref(), mr), Value::scalar(e, mr));
}

keel::Tuple<Value, Value>
prony(const Value &h, keel::i32 nb, keel::i32 na, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto hv = readVec(h, scratch.allocator());
    keel::signal::TfResult tf;
    (void)keel::signal::prony(hv.as_ref(), nb, na, tf, scratch.allocator());
    return keel::make_tuple(rowVec(tf.b.as_ref(), mr), rowVec(tf.a.as_ref(), mr));
}

keel::Tuple<Value, Value>
stmcb(const Value &y, const Value &x, keel::i32 nb, keel::i32 na, keel::i32 niter,
      const Value &ai, keel::Allocator *mr)
{
    stmcb_argcheck(y, x, nb, na, ai);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto yv = readVec(y, scratch.allocator());
    auto xv = x.isEmpty() ? keel::Array<keel::f64>(scratch.allocator()) : readVec(x, scratch.allocator());
    auto aiv = ai.isEmpty() ? keel::Array<keel::f64>(scratch.allocator()) : readVec(ai, scratch.allocator());
    keel::signal::TfResult tf;
    (void)keel::signal::stmcb(yv.as_ref(), xv.as_ref(), nb, na, niter, aiv.as_ref(), tf,
                              scratch.allocator());
    return keel::make_tuple(rowVec(tf.b.as_ref(), mr), rowVec(tf.a.as_ref(), mr));
}

keel::Tuple<Value, Value>
stmcb(const Value &h, keel::i32 nb, keel::i32 na, keel::i32 niter, keel::Allocator *mr)
{
    return stmcb(h, Value(), nb, na, niter, Value(), mr);
}

Value corrmtx(const Value &x, keel::i32 m, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    const keel::usize rows = xv.size() + keel::as<keel::usize>(m);
    const keel::usize cols = keel::as<keel::usize>(m + 1);
    auto X = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    (void)keel::signal::corrmtx(xv.as_ref(), m,
                                keel::MutArrayRef<keel::f64>(X.doubleDataMut(), rows * cols));
    return X;
}

Value poly2lsf(const Value &a, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto av = readVec(a, scratch.allocator());
    keel::Array<keel::f64> lsf(scratch.allocator());
    (void)keel::signal::poly2lsf(av.as_ref(), lsf, scratch.allocator());
    return colVec(lsf.as_ref(), mr);
}

Value lsf2poly(const Value &lsf, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto lv = readVec(lsf, scratch.allocator());
    keel::Array<keel::f64> a(scratch.allocator());
    (void)keel::signal::lsf2poly(lv.as_ref(), a, scratch.allocator());
    return rowVec(a.as_ref(), mr);
}

keel::Tuple<Value, Value>
invfreqs(const Value &H, const Value &w, keel::i32 nb, keel::i32 na, keel::Allocator *mr)
{
    invfreq_argcheck(H, w, nb, na, "invfreqs");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto Hv = readComplexVec(H, scratch.allocator());
    auto wv = readVec(w, scratch.allocator());
    keel::signal::TfResult tf;
    (void)keel::signal::invfreqs(Hv.as_ref(), wv.as_ref(), nb, na, tf, scratch.allocator());
    return keel::make_tuple(rowVec(tf.b.as_ref(), mr), rowVec(tf.a.as_ref(), mr));
}

keel::Tuple<Value, Value>
invfreqz(const Value &H, const Value &w, keel::i32 nb, keel::i32 na, keel::Allocator *mr)
{
    invfreq_argcheck(H, w, nb, na, "invfreqz");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto Hv = readComplexVec(H, scratch.allocator());
    auto wv = readVec(w, scratch.allocator());
    keel::signal::TfResult tf;
    (void)keel::signal::invfreqz(Hv.as_ref(), wv.as_ref(), nb, na, tf, scratch.allocator());
    return keel::make_tuple(rowVec(tf.b.as_ref(), mr), rowVec(tf.a.as_ref(), mr));
}

} // namespace numkit::signal
