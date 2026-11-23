// toolboxes/signal/src/filter_design/analog_filters.cpp
//
// Analog prototypes + frequency transforms + bilinear / impinvar + freqs.
// Thin Value wrappers delegating numerical kernels to keel::signal.

#include <keel/arena.h>
#include <keel/tuple.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/filter_design/analog_filters.hpp>
#include <numkit/value/error.hpp>

#include <numkit/value/value.hpp>


namespace numkit::signal {

namespace {

keel::Array<keel::c64> readComplexVec(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::c64> out(n, alloc);
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

keel::Array<keel::f64> readVec(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, alloc);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

Value packComplexCol(keel::ArrayRef<keel::c64> v, keel::Allocator *mr)
{
    if (v.empty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto out = Value::complexMatrix(v.size(), 1, mr);
    keel::copy_n(out.complexDataMut(), v.data(), v.size());
    return out;
}

Value packDoubleRow(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

keel::Tuple<Value, Value, Value> packZpk(const keel::signal::ZpkResult &res, keel::Allocator *mr)
{
    return keel::make_tuple(packComplexCol(res.z.as_ref(), mr),
                           packComplexCol(res.p.as_ref(), mr),
                           Value::scalar(res.k, mr));
}

void impinvar_argcheck(const Value &b, const Value &a)
{
    if (a.numel() < 2)
        throw numkit::Error("impinvar: a must have degree >= 1");
    if (b.numel() >= a.numel())
        throw numkit::Error("impinvar: numerator degree must be less than denominator");
}

} // anonymous

keel::Tuple<Value, Value, Value>
buttap(keel::i32 N, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::signal::ZpkResult res;
    (void)keel::signal::buttap(N, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
cheb1ap(keel::i32 N, keel::f64 Rp, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::signal::ZpkResult res;
    (void)keel::signal::cheb1ap(N, Rp, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
cheb2ap(keel::i32 N, keel::f64 Rs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::signal::ZpkResult res;
    (void)keel::signal::cheb2ap(N, Rs, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
besselap(keel::i32 N, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::signal::ZpkResult res;
    (void)keel::signal::besselap(N, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
ellipap(keel::i32 N, keel::f64 Rp, keel::f64 Rs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::signal::ZpkResult res;
    (void)keel::signal::ellipap(N, Rp, Rs, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
lp2lp(const Value &z, const Value &p, keel::f64 k, keel::f64 Wo, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto zv = readComplexVec(z, scratch.allocator());
    const auto pv = readComplexVec(p, scratch.allocator());
    keel::signal::ZpkResult res;
    (void)keel::signal::lp2lp(zv.as_ref(), pv.as_ref(), k, Wo, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
lp2hp(const Value &z, const Value &p, keel::f64 k, keel::f64 Wo, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto zv = readComplexVec(z, scratch.allocator());
    const auto pv = readComplexVec(p, scratch.allocator());
    keel::signal::ZpkResult res;
    (void)keel::signal::lp2hp(zv.as_ref(), pv.as_ref(), k, Wo, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
lp2bp(const Value &z, const Value &p, keel::f64 k, keel::f64 Wo, keel::f64 Bw, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto zv = readComplexVec(z, scratch.allocator());
    const auto pv = readComplexVec(p, scratch.allocator());
    keel::signal::ZpkResult res;
    (void)keel::signal::lp2bp(zv.as_ref(), pv.as_ref(), k, Wo, Bw, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value, Value>
lp2bs(const Value &z, const Value &p, keel::f64 k, keel::f64 Wo, keel::f64 Bw, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto zv = readComplexVec(z, scratch.allocator());
    const auto pv = readComplexVec(p, scratch.allocator());
    keel::signal::ZpkResult res;
    (void)keel::signal::lp2bs(zv.as_ref(), pv.as_ref(), k, Wo, Bw, res, scratch.allocator());
    return packZpk(res, mr);
}

keel::Tuple<Value, Value>
bilinear(const Value &b, const Value &a, keel::f64 fs, keel::f64 fp, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto bv = readVec(b, scratch.allocator());
    const auto av = readVec(a, scratch.allocator());
    keel::signal::TfResult res;
    (void)keel::signal::bilinear(bv.as_ref(), av.as_ref(), fs, fp, res, scratch.allocator());
    return keel::make_tuple(packDoubleRow(res.b.as_ref(), mr), packDoubleRow(res.a.as_ref(), mr));
}

keel::Tuple<Value, Value>
impinvar(const Value &b, const Value &a, keel::f64 fs, keel::f64 /*tol*/, keel::Allocator *mr)
{
    impinvar_argcheck(b, a);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto bv = readVec(b, scratch.allocator());
    const auto av = readVec(a, scratch.allocator());
    keel::signal::TfResult res;
    const auto st = keel::signal::impinvar(bv.as_ref(), av.as_ref(), fs, res, scratch.allocator());
    if (st != keel::signal::SignalStatus::kOk)
        throw numkit::Error("impinvar: roots() returned unexpected count");
    return keel::make_tuple(packDoubleRow(res.b.as_ref(), mr), packDoubleRow(res.a.as_ref(), mr));
}

Value freqs(const Value &b, const Value &a, const Value &w, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto bv = readVec(b, scratch.allocator());
    const auto av = readVec(a, scratch.allocator());
    const auto wv = readVec(w, scratch.allocator());
    const keel::usize outRows = (w.dims().rows() == 1 && w.dims().cols() == wv.size()) ? 1 : wv.size();
    const keel::usize outCols = (outRows == 1) ? wv.size() : 1;
    auto out = Value::complexMatrix(outRows, outCols, mr);
    (void)keel::signal::freqs(bv.as_ref(), av.as_ref(), wv.as_ref(),
                              keel::MutArrayRef<keel::c64>(out.complexDataMut(), wv.size()));
    return out;
}

} // namespace numkit::signal
