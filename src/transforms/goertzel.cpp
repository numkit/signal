// toolboxes/signal/src/transforms/goertzel.cpp
//
// Goertzel single-bin DFT, delegated to keel::signal.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/transforms/goertzel.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

void goertzel_argcheck(const Value &x, const Value &ind)
{
    detail::requireFloatInput(x, "goertzel");
    detail::requireNumericOrLogicalInput(ind, "goertzel", "ind", 2);
}

} // namespace

Value goertzel(const Value &x, const Value &ind, keel::Allocator *mr)
{
    goertzel_argcheck(x, ind);

    const keel::usize N = x.numel();
    const keel::usize M = ind.numel();
    auto r = Value::complexMatrix(ind.dims().rows(), ind.dims().cols(), mr);

    if (N == 0 || M == 0) return r;

    const bool xComplex = x.isComplex();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> id(M, scratch.allocator());
    for (keel::usize j = 0; j < M; ++j) id[j] = ind.elemAsDouble(j);

    keel::c64 *od = r.complexDataMut();
    if (xComplex) {
        const keel::ArrayRef<keel::c64> xRef(x.complexData(), N);
        for (keel::usize m = 0; m < M; ++m) {
            od[m] = keel::signal::goertzel_bin_c64(xRef, id[m] - 1.0);
        }
    } else if (x.type() == ValueType::DOUBLE) {
        const keel::ArrayRef<keel::f64> xRef(x.doubleData(), N);
        for (keel::usize m = 0; m < M; ++m) {
            od[m] = keel::signal::goertzel_bin(xRef, id[m] - 1.0);
        }
    } else {
        keel::Array<keel::f64> xr(N, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) xr[i] = x.elemAsDouble(i);
        for (keel::usize m = 0; m < M; ++m) {
            od[m] = keel::signal::goertzel_bin(xr, id[m] - 1.0);
        }
    }
    return r;
}

} // namespace numkit::signal
