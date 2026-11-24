// toolboxes/signal/src/filter_analysis/frequency_response.cpp
//
// Frequency-domain analysis of an existing filter — freqz / phasez /
// grpdelay (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/filter_analysis/frequency_response.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

void freqz_argcheck(const Value &b, const Value &a)
{
    (void)b;
    (void)a;
}

keel::Array<keel::c64> asComplexBuf(const Value &v, keel::Allocator *mr)
{
    const keel::usize n = v.numel();
    keel::Array<keel::c64> out(n, mr);
    if (v.isComplex()) {
        const keel::c64 *p = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = p[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

} // namespace

keel::Tuple<Value, Value>
freqz(const Value &b, const Value &a, keel::usize npts, bool whole, keel::f64 fs, keel::Allocator *mr)
{
    freqz_argcheck(b, a);
    auto W = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    auto H = Value::complexMatrix(npts, 1, mr);
    if (b.isComplex() || a.isComplex()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto bc = asComplexBuf(b, scratch.allocator());
        auto ac = asComplexBuf(a, scratch.allocator());
        (void)keel::signal::freqz_grid_c64(
            bc.as_ref(), ac.as_ref(), npts, whole, fs,
            keel::MutArrayRef<keel::c64>(H.complexDataMut(), npts),
            keel::MutArrayRef<keel::f64>(W.doubleDataMut(), npts));
    } else {
        (void)keel::signal::freqz_grid(
            keel::ArrayRef<keel::f64>(b.doubleData(), b.numel()),
            keel::ArrayRef<keel::f64>(a.doubleData(), a.numel()),
            npts, whole, fs,
            keel::MutArrayRef<keel::c64>(H.complexDataMut(), npts),
            keel::MutArrayRef<keel::f64>(W.doubleDataMut(), npts));
    }
    return keel::make_tuple(keel::move(H), keel::move(W));
}

keel::Tuple<Value, Value>
freqz(const Value &b, const Value &a, const Value &wvec, keel::Allocator *mr)
{
    const keel::usize n = wvec.numel();
    const keel::usize rows = wvec.dims().rows(), cols = wvec.dims().cols();
    auto W = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    auto H = Value::complexMatrix(rows, cols, mr);
    if (n > 0) {
        keel::copy_n(W.doubleDataMut(), wvec.doubleData(), n);
    }
    if (b.isComplex() || a.isComplex()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto bc = asComplexBuf(b, scratch.allocator());
        auto ac = asComplexBuf(a, scratch.allocator());
        (void)keel::signal::freqz_c64(
            bc.as_ref(), ac.as_ref(),
            keel::ArrayRef<keel::f64>(wvec.doubleData(), n),
            keel::MutArrayRef<keel::c64>(H.complexDataMut(), n));
    } else {
        (void)keel::signal::freqz(
            keel::ArrayRef<keel::f64>(b.doubleData(), b.numel()),
            keel::ArrayRef<keel::f64>(a.doubleData(), a.numel()),
            keel::ArrayRef<keel::f64>(wvec.doubleData(), n),
            keel::MutArrayRef<keel::c64>(H.complexDataMut(), n));
    }
    return keel::make_tuple(keel::move(H), keel::move(W));
}

keel::Tuple<Value, Value>
phasez(const Value &b, const Value &a, keel::usize npts, keel::f64 fs, keel::Allocator *mr)
{
    auto phi = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    auto W   = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    if (b.isComplex() || a.isComplex()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto bc = asComplexBuf(b, scratch.allocator());
        auto ac = asComplexBuf(a, scratch.allocator());
        (void)keel::signal::phasez_c64(
            bc.as_ref(), ac.as_ref(), npts, fs,
            keel::MutArrayRef<keel::f64>(phi.doubleDataMut(), npts),
            keel::MutArrayRef<keel::f64>(W.doubleDataMut(), npts));
    } else {
        (void)keel::signal::phasez(
            keel::ArrayRef<keel::f64>(b.doubleData(), b.numel()),
            keel::ArrayRef<keel::f64>(a.doubleData(), a.numel()),
            npts, fs,
            keel::MutArrayRef<keel::f64>(phi.doubleDataMut(), npts),
            keel::MutArrayRef<keel::f64>(W.doubleDataMut(), npts));
    }
    return keel::make_tuple(keel::move(phi), keel::move(W));
}

keel::Tuple<Value, Value>
grpdelay(const Value &b, const Value &a, keel::usize npts, keel::f64 fs, keel::Allocator *mr)
{
    auto W  = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    auto gd = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::grpdelay(
        keel::ArrayRef<keel::f64>(b.doubleData(), b.numel()),
        keel::ArrayRef<keel::f64>(a.doubleData(), a.numel()),
        npts, fs,
        keel::MutArrayRef<keel::f64>(gd.doubleDataMut(), npts),
        keel::MutArrayRef<keel::f64>(W.doubleDataMut(), npts), mr);
    return keel::make_tuple(keel::move(gd), keel::move(W));
}

} // namespace numkit::signal
