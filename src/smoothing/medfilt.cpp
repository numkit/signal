// toolboxes/signal/src/smoothing/medfilt.cpp
//
// medfilt1 and fillgaps delegating 1-D kernels to keel::signal.

#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/smoothing/medfilt.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

void fillgaps_argcheck(const Value &x, keel::usize maxlen, keel::usize order) {
    detail::requireFloatInput(x, "fillgaps");
    if (x.isEmpty())
        throw Error("fillgaps: input X must be non-empty",
                    0, 0, "fillgaps", "", "numkit:fillgaps:emptyInput");
    if (x.dims().ndims() > 2)
        throw Error("fillgaps: input X must be a vector or 2-D matrix",
                    0, 0, "fillgaps", "", "numkit:fillgaps:expected2D");
    if (maxlen > 0 && order > 0 && order >= maxlen)
        throw Error(keel::format("fillgaps: ORDER must be strictly less than MAXLEN (got order={}, maxlen={})",
                                 order, maxlen),
                    0, 0, "fillgaps", "", "numkit:fillgaps:orderTooLarge");
}

} // namespace

Value medfilt1(const Value &x, keel::usize k, bool zeropad, keel::Allocator *mr) {
    detail::requireRealFloatInput(x, "medfilt1");
    const bool xSingle = (x.type() == ValueType::SINGLE);

    const keel::usize total = x.numel();
    auto r = createLike(x, ValueType::DOUBLE, mr);
    if (total == 0) return detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr);

    Value xD = detail::widenSingleToDouble(x, mr);
    const keel::f64 *src = xD.doubleData();
    keel::f64 *dst = r.doubleDataMut();

    const keel::usize R = keel::as<keel::usize>(xD.dims().dim(0));
    if (R <= 1 || xD.dims().isVector()) {
        (void)keel::signal::medfilt1(keel::ArrayRef<keel::f64>(src, total), k, zeropad,
                                     keel::MutArrayRef<keel::f64>(dst, total), mr);
    } else {
        const keel::usize ncols = total / R;
        for (keel::usize c = 0; c < ncols; ++c) {
            (void)keel::signal::medfilt1(keel::ArrayRef<keel::f64>(src + c * R, R), k, zeropad,
                                         keel::MutArrayRef<keel::f64>(dst + c * R, R), mr);
        }
    }
    return detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr);
}

Value fillgaps(const Value &x, keel::usize maxlen, keel::usize order, keel::Allocator *mr) {
    fillgaps_argcheck(x, maxlen, order);
    const bool xSingle = (x.type() == ValueType::SINGLE);
    const keel::usize total = x.numel();

    if (x.type() == ValueType::COMPLEX) {
        auto r = createLike(x, ValueType::COMPLEX, mr);
        const keel::c64 *src = x.complexData();
        keel::c64 *dst = r.complexDataMut();
        const keel::usize R = x.dims().rows();
        if (R <= 1 || x.dims().isVector()) {
            (void)keel::signal::fillgaps_c64(keel::ArrayRef<keel::c64>(src, total), maxlen, order,
                                             keel::MutArrayRef<keel::c64>(dst, total), mr);
        } else {
            const keel::usize ncols = total / R;
            for (keel::usize c = 0; c < ncols; ++c) {
                (void)keel::signal::fillgaps_c64(keel::ArrayRef<keel::c64>(src + c * R, R),
                                                 maxlen, order,
                                                 keel::MutArrayRef<keel::c64>(dst + c * R, R), mr);
            }
        }
        return r;
    }

    Value xD = detail::widenSingleToDouble(x, mr);
    auto r = createLike(x, ValueType::DOUBLE, mr);
    const keel::f64 *src = xD.doubleData();
    keel::f64 *dst = r.doubleDataMut();
    const keel::usize R = xD.dims().rows();
    if (R <= 1 || xD.dims().isVector()) {
        (void)keel::signal::fillgaps(keel::ArrayRef<keel::f64>(src, total), maxlen, order,
                                     keel::MutArrayRef<keel::f64>(dst, total), mr);
    } else {
        const keel::usize ncols = total / R;
        for (keel::usize c = 0; c < ncols; ++c) {
            (void)keel::signal::fillgaps(keel::ArrayRef<keel::f64>(src + c * R, R), maxlen, order,
                                         keel::MutArrayRef<keel::f64>(dst + c * R, R), mr);
        }
    }
    return detail::narrowDoubleToSingleIf(keel::move(r), xSingle, mr);
}

} // namespace numkit::signal
