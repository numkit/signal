// toolboxes/signal/src/digital_filtering/sosfilt.cpp
//
// Apply an SOS biquad cascade (and zero-phase sosfiltfilt) to a signal,
// delegated to keel::signal. Conversions zp2sos / tf2sos live in
// filter_implementation/conversions.cpp.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/signal.h>
#include <numkit/signal/digital_filtering/sosfilt.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (argcheck got-clauses)

#include <numkit/ops/helpers.hpp>

#include <utility>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

keel::usize validateSosMatrix(const Value &sos, keel::StringRef fn)
{
    if (sos.dims().ndim() != 2 || sos.dims().cols() != 6 || sos.dims().rows() == 0)
        throw Error(keel::format("{}: sos matrix must be L×6 with L >= 1", fn),
                     0, 0, fn, "",
                     keel::format("numkit:{}:sosShape", fn));
    if (sos.type() == ValueType::DOUBLE || sos.type() == ValueType::SINGLE)
        return sos.dims().rows();
    throw Error(keel::format("{}: sos matrix must be DOUBLE or SINGLE", fn),
                 0, 0, fn, "",
                 keel::format("numkit:{}:sosType", fn));
}

void sosfilt_argcheck(const Value &sos, const Value &x)
{
    validateSosMatrix(sos, "sosfilt");
    if (x.type() != ValueType::DOUBLE && x.type() != ValueType::SINGLE
        && x.type() != ValueType::COMPLEX) {
        throw Error(keel::format("sosfilt: signal x must have floating point precision (got a {}x{} {} array)",
                                 x.dims().rows(), x.dims().cols(), mtypeName(x.type())),
                     0, 0, "sosfilt", "", "numkit:sosfilt:xType");
    }
}

void sosfiltfilt_argcheck(const Value &sos, const Value &x)
{
    validateSosMatrix(sos, "sosfiltfilt");
    if (x.type() != ValueType::DOUBLE && x.type() != ValueType::SINGLE) {
        throw Error(keel::format("sosfiltfilt: signal x must have floating point precision (got a {}x{} {} array)",
                                 x.dims().rows(), x.dims().cols(), mtypeName(x.type())),
                     0, 0, "sosfiltfilt", "", "numkit:sosfiltfilt:xType");
    }
}

void checkZeroLead(keel::signal::SignalStatus st, keel::StringRef fn)
{
    if (st == keel::signal::SignalStatus::kZeroLeadingCoeff) {
        throw Error(keel::format("{}: section a0 is zero", fn),
                     0, 0, fn, "", keel::format("numkit:{}:zeroLead", fn));
    }
}

} // namespace

Value sosfilt(const Value &sos, const Value &x, keel::Allocator *mr)
{
    sosfilt_argcheck(sos, x);
    const bool anySingle = (sos.type() == ValueType::SINGLE) || (x.type() == ValueType::SINGLE);
    Value sosD = detail::widenSingleToDouble(sos, mr);
    if (x.type() == ValueType::COMPLEX) {
        if (x.isEmpty())
            return createLike(x, ValueType::COMPLEX, mr);
        const keel::usize N = x.numel();
        Value xr = createLike(x, ValueType::DOUBLE, mr);
        Value xi = createLike(x, ValueType::DOUBLE, mr);
        keel::f64 *xrd = xr.doubleDataMut();
        keel::f64 *xid = xi.doubleDataMut();
        const keel::c64 *xcd = x.complexData();
        for (keel::usize i = 0; i < N; ++i) {
            xrd[i] = xcd[i].real();
            xid[i] = xcd[i].imag();
        }
        Value yr = sosfilt(sosD, xr, mr);
        Value yi = sosfilt(sosD, xi, mr);
        Value out = createLike(x, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        const keel::f64 *yrd = yr.doubleData();
        const keel::f64 *yid = yi.doubleData();
        for (keel::usize i = 0; i < N; ++i)
            od[i] = keel::c64(yrd[i], yid[i]);
        return out;
    }
    if (x.isEmpty())
        return createLike(x, anySingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value xD = detail::widenSingleToDouble(x, mr);
    auto out = createLike(x, ValueType::DOUBLE, mr);
    const keel::usize L = sosD.dims().rows();
    const keel::ArrayRef<keel::f64> sosRef(sosD.doubleData(), L * 6);
    if (xD.dims().isVector() || xD.isScalar()) {
        const keel::usize n = xD.numel();
        checkZeroLead(keel::signal::sosfilt_col_major(sosRef, L, {xD.doubleData(), n},
                                                      {out.doubleDataMut(), n}, {}, mr),
                      "sosfilt");
        return detail::narrowDoubleToSingleIf(std::move(out), anySingle, mr);
    }
    const keel::usize rows = xD.dims().rows();
    const keel::usize cols = xD.dims().cols();
    const keel::f64 *src = xD.doubleData();
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize c = 0; c < cols; ++c) {
        checkZeroLead(keel::signal::sosfilt_col_major(sosRef, L, {src + c * rows, rows},
                                                      {dst + c * rows, rows}, {}, mr),
                      "sosfilt");
    }
    return detail::narrowDoubleToSingleIf(std::move(out), anySingle, mr);
}

Value sosfiltfilt(const Value &sos, const Value &x, keel::Allocator *mr)
{
    sosfiltfilt_argcheck(sos, x);
    const bool anySingle = (sos.type() == ValueType::SINGLE) || (x.type() == ValueType::SINGLE);
    if (x.isEmpty())
        return createLike(x, anySingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value sosD = detail::widenSingleToDouble(sos, mr);
    Value xD = detail::widenSingleToDouble(x, mr);
    auto out = createLike(x, ValueType::DOUBLE, mr);
    const keel::usize L = sosD.dims().rows();
    const keel::ArrayRef<keel::f64> sosRef(sosD.doubleData(), L * 6);
    if (xD.dims().isVector() || xD.isScalar()) {
        const keel::usize n = xD.numel();
        checkZeroLead(keel::signal::sosfiltfilt_col_major(sosRef, L, {xD.doubleData(), n},
                                                          {out.doubleDataMut(), n}, mr),
                      "sosfiltfilt");
        return detail::narrowDoubleToSingleIf(std::move(out), anySingle, mr);
    }
    const keel::usize rows = xD.dims().rows();
    const keel::usize cols = xD.dims().cols();
    const keel::f64 *src = xD.doubleData();
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize c = 0; c < cols; ++c) {
        checkZeroLead(keel::signal::sosfiltfilt_col_major(sosRef, L, {src + c * rows, rows},
                                                          {dst + c * rows, rows}, mr),
                      "sosfiltfilt");
    }
    return detail::narrowDoubleToSingleIf(std::move(out), anySingle, mr);
}

} // namespace numkit::signal
