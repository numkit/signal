// toolboxes/signal/src/filter_design/iir_designs.cpp
//
// Top-level IIR filter designs (butter / cheby1 / cheby2 / ellip / besself)
// and IIR/FIR order estimators (buttord / cheb1ord / cheb2ord / ellipord /
// firpmord / kaiserord), delegating pure kernels to keel::signal.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <keel/string.h>
#include <numkit/signal/filter_design/iir_designs.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <cstring>
#include <stdexcept>

namespace numkit::signal {

namespace {

void iirOrder_argcheck(keel::i32 N, const char *fn)
{
    if (N <= 0)
        throw Error(keel::format("{}: filter order N must be a positive integer (got {}).", fn, N),
                    0, 0, fn, "",
                    keel::format("numkit:{}:orderNotPositive", fn));
}

keel::Array<keel::f64> readVec(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, alloc);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void validateWn(FilterType ftype, keel::ArrayRef<keel::f64> Wn)
{
    const bool isBand = (ftype == FilterType::Bandpass || ftype == FilterType::Bandstop);
    if (isBand && Wn.size() != 2)
        throw std::runtime_error("bandpass/bandstop requires 2-element Wn");
    if (!isBand && Wn.size() != 1)
        throw std::runtime_error("lowpass/highpass requires scalar Wn");
}

keel::signal::FilterType toKeelFilterType(FilterType ftype)
{
    switch (ftype) {
        case FilterType::Lowpass:  return keel::signal::FilterType::kLowpass;
        case FilterType::Highpass: return keel::signal::FilterType::kHighpass;
        case FilterType::Bandpass: return keel::signal::FilterType::kBandpass;
        case FilterType::Bandstop: return keel::signal::FilterType::kBandstop;
    }
    throw std::runtime_error("iir_designs: bad ftype");
}

Value packDoubleRow(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::memcpy(out.doubleDataMut(), v.data(), v.size() * sizeof(keel::f64));
    return out;
}

std::tuple<Value, Value> packTf(const keel::signal::TfResult &tf, keel::Allocator *mr)
{
    return std::make_tuple(packDoubleRow(tf.b.as_ref(), mr), packDoubleRow(tf.a.as_ref(), mr));
}

Value packOrderWn(const keel::signal::OrderResult &ord, keel::Allocator *mr)
{
    if (ord.wn_count == 1)
        return Value::scalar(ord.wn[0], mr);
    auto out = Value::matrix(1, ord.wn_count, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < ord.wn_count; ++i)
        out.doubleDataMut()[i] = ord.wn[i];
    return out;
}

void validateOrdSpecs(keel::ArrayRef<keel::f64> Wp, keel::ArrayRef<keel::f64> Ws,
                      keel::f64 Rp, keel::f64 Rs, const char *fn)
{
    if (!((Wp.size() == 1 && Ws.size() == 1) || (Wp.size() == 2 && Ws.size() == 2)))
        throw std::runtime_error("ord: Wp and Ws must both be scalar or both 2-vectors");
    if (Wp.size() == 1 && Wp[0] == Ws[0])
        throw std::runtime_error("ord: Wp and Ws must differ");
    if (Rp <= 0.0 || Rs <= 0.0)
        throw std::runtime_error(keel::format("{}: Rp, Rs must be positive", fn).c_str());
}

} // anonymous

std::tuple<Value, Value>
cheby1(keel::i32 N, keel::f64 Rp, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "cheby1");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto wn = readVec(Wn, scratch.allocator());
    validateWn(ftype, wn.as_ref());
    keel::signal::TfResult tf;
    (void)keel::signal::cheby1(N, Rp, wn.as_ref(), toKeelFilterType(ftype), analog, tf, scratch.allocator());
    return packTf(tf, mr);
}

std::tuple<Value, Value>
cheby2(keel::i32 N, keel::f64 Rs, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "cheby2");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto wn = readVec(Wn, scratch.allocator());
    validateWn(ftype, wn.as_ref());
    keel::signal::TfResult tf;
    (void)keel::signal::cheby2(N, Rs, wn.as_ref(), toKeelFilterType(ftype), analog, tf, scratch.allocator());
    return packTf(tf, mr);
}

std::tuple<Value, Value>
besself(keel::i32 N, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "besself");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto wn = readVec(Wn, scratch.allocator());
    validateWn(ftype, wn.as_ref());
    keel::signal::TfResult tf;
    (void)keel::signal::besself(N, wn.as_ref(), toKeelFilterType(ftype), analog, tf, scratch.allocator());
    return packTf(tf, mr);
}

std::tuple<Value, Value>
butter(keel::i32 N, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "butter");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto wn = readVec(Wn, scratch.allocator());
    validateWn(ftype, wn.as_ref());
    keel::signal::TfResult tf;
    (void)keel::signal::butter(N, wn.as_ref(), toKeelFilterType(ftype), analog, tf, scratch.allocator());
    return packTf(tf, mr);
}

std::tuple<Value, Value>
ellip(keel::i32 N, keel::f64 Rp, keel::f64 Rs, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "ellip");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto wn = readVec(Wn, scratch.allocator());
    validateWn(ftype, wn.as_ref());
    keel::signal::TfResult tf;
    (void)keel::signal::ellip(N, Rp, Rs, wn.as_ref(), toKeelFilterType(ftype), analog, tf, scratch.allocator());
    return packTf(tf, mr);
}

std::tuple<keel::i32, Value>
buttord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto Wp = readVec(Wp_v, scratch.allocator());
    const auto Ws = readVec(Ws_v, scratch.allocator());
    validateOrdSpecs(Wp.as_ref(), Ws.as_ref(), Rp, Rs, "buttord");
    keel::signal::OrderResult ord;
    const auto st = keel::signal::buttord(Wp.as_ref(), Ws.as_ref(), Rp, Rs, analog, ord);
    if (st != keel::signal::SignalStatus::kOk)
        throw std::runtime_error("buttord: invalid filter specification");
    return std::make_tuple(ord.order, packOrderWn(ord, mr));
}

std::tuple<keel::i32, Value>
cheb1ord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto Wp = readVec(Wp_v, scratch.allocator());
    const auto Ws = readVec(Ws_v, scratch.allocator());
    validateOrdSpecs(Wp.as_ref(), Ws.as_ref(), Rp, Rs, "cheb1ord");
    keel::signal::OrderResult ord;
    const auto st = keel::signal::cheb1ord(Wp.as_ref(), Ws.as_ref(), Rp, Rs, analog, ord);
    if (st != keel::signal::SignalStatus::kOk)
        throw std::runtime_error("cheb1ord: invalid filter specification");
    return std::make_tuple(ord.order, packOrderWn(ord, mr));
}

std::tuple<keel::i32, Value>
cheb2ord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto Wp = readVec(Wp_v, scratch.allocator());
    const auto Ws = readVec(Ws_v, scratch.allocator());
    validateOrdSpecs(Wp.as_ref(), Ws.as_ref(), Rp, Rs, "cheb2ord");
    keel::signal::OrderResult ord;
    const auto st = keel::signal::cheb2ord(Wp.as_ref(), Ws.as_ref(), Rp, Rs, analog, ord);
    if (st != keel::signal::SignalStatus::kOk)
        throw std::runtime_error("cheb2ord: invalid filter specification");
    return std::make_tuple(ord.order, packOrderWn(ord, mr));
}

std::tuple<keel::i32, Value>
ellipord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    if (Rp <= 0.0 || Rs <= 0.0)
        throw Error("ellipord: Rp, Rs must be positive",
                    0, 0, "ellipord", "", "numkit:ellipord:BadRpRs");
    if (Wp_v.numel() != Ws_v.numel())
        throw Error("ellipord: Wp and Ws must have same length",
                    0, 0, "ellipord", "", "numkit:ellipord:DimMismatch");
    const keel::usize numW = Wp_v.numel();
    if (numW != 1 && numW != 2)
        throw Error("ellipord: Wp must be scalar or 2-vector",
                    0, 0, "ellipord", "", "numkit:ellipord:BadWp");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto wp = readVec(Wp_v, scratch.allocator());
    const auto ws = readVec(Ws_v, scratch.allocator());
    keel::signal::OrderResult ord;
    const auto st = keel::signal::ellipord(wp.as_ref(), ws.as_ref(), Rp, Rs, analog, ord);
    if (st != keel::signal::SignalStatus::kOk)
        throw Error("ellipord: invalid filter spec",
                    0, 0, "ellipord", "", "numkit:ellipord:BadSpec");

    Value Wn = Value::matrix(1, ord.wn_count, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < ord.wn_count; ++i)
        Wn.doubleDataMut()[i] = ord.wn[i];
    return {ord.order, std::move(Wn)};
}

std::tuple<keel::i32, Value, Value, Value>
firpmord(const Value &F, const Value &A, const Value &dev, keel::f64 fs, keel::Allocator *mr)
{
    const keel::usize mf     = F.numel();
    const keel::usize nbands = A.numel();
    const keel::usize ndevs  = dev.numel();

    if (nbands != ndevs)
        throw Error("firpmord: A and DEV must have same length",
                    0, 0, "firpmord", "", "numkit:firpmord:MismatchedVectorLength");
    if (mf != 2 * (nbands - 1))
        throw Error("firpmord: numel(F) must equal 2*(numel(A)-1)",
                    0, 0, "firpmord", "", "numkit:firpmord:InvalidLength");
    if (fs <= 0.0)
        throw Error("firpmord: Fs must be positive",
                    0, 0, "firpmord", "", "numkit:firpmord:BadFs");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto fv = readVec(F, scratch.allocator());
    const auto av = readVec(A, scratch.allocator());
    const auto dv = readVec(dev, scratch.allocator());
    for (keel::usize i = 0; i < mf; ++i) {
        if (fv[i] / fs > 0.5)
            throw Error("firpmord: F edges must be <= Fs/2",
                        0, 0, "firpmord", "", "numkit:firpmord:InvalidRange");
    }

    keel::signal::FirpmOrdResult res;
    (void)keel::signal::firpmord(fv.as_ref(), av.as_ref(), dv.as_ref(), fs, res, scratch.allocator());
    return {res.order,
            packDoubleRow(res.f.as_ref(), mr),
            packDoubleRow(res.a.as_ref(), mr),
            packDoubleRow(res.w.as_ref(), mr)};
}

std::tuple<keel::i32, Value, keel::f64, keel::String>
kaiserord(const Value &F, const Value &A, const Value &dev, keel::f64 fs, keel::Allocator *mr)
{
    const keel::usize mf     = F.numel();
    const keel::usize nbands = A.numel();
    const keel::usize ndevs  = dev.numel();

    if (nbands != ndevs)
        throw Error("kaiserord: A and DEV must have same length",
                    0, 0, "kaiserord", "", "numkit:kaiserord:InvalidDimensionsADEV");
    if (mf != 2 * (nbands - 1))
        throw Error("kaiserord: numel(F) must equal 2*(numel(A)-1)",
                    0, 0, "kaiserord", "", "numkit:kaiserord:InvalidDimensionsLengthF");
    if (fs <= 0.0)
        throw Error("kaiserord: Fs must be positive",
                    0, 0, "kaiserord", "", "numkit:kaiserord:BadFs");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto fv = readVec(F, scratch.allocator());
    const auto av = readVec(A, scratch.allocator());
    const auto dv = readVec(dev, scratch.allocator());
    for (keel::usize i = 0; i < mf; ++i) {
        if (fv[i] / fs >= 0.5)
            throw Error("kaiserord: F edges must be < Fs/2",
                        0, 0, "kaiserord", "", "numkit:kaiserord:InvalidRange");
    }

    keel::signal::KaiserOrdResult res;
    (void)keel::signal::kaiserord(fv.as_ref(), av.as_ref(), dv.as_ref(), fs, res, scratch.allocator());

    const char *ftype_str = "low";
    switch (res.ftype) {
        case keel::signal::KaiserBandType::kLow:  ftype_str = "low"; break;
        case keel::signal::KaiserBandType::kHigh: ftype_str = "high"; break;
        case keel::signal::KaiserBandType::kStop: ftype_str = "stop"; break;
        case keel::signal::KaiserBandType::kDc0:  ftype_str = "DC-0"; break;
        case keel::signal::KaiserBandType::kDc1:  ftype_str = "DC-1"; break;
    }
    return {res.order, packDoubleRow(res.wn.as_ref(), mr), res.beta, keel::String::from(ftype_str, mr)};
}

} // namespace numkit::signal
