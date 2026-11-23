// toolboxes/signal/src/transforms/dct.cpp
//
// Orthonormal DCT/IDCT Types 1-4 delegating numerical kernels to keel::signal.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/transforms/dct.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/value/value.hpp>
#include "dct_detail.hpp"


namespace numkit::signal {

namespace {

keel::i32 resolveDim(const Value &x, keel::i32 dim) {
    if (dim != 0) return dim;
    return (x.dims().rows() > 1) ? 1 : 2;
}

Value runDctLines(const Value &x, keel::i32 n, keel::i32 dim, keel::i32 type, bool inverse,
                  keel::Allocator *mr) {
    if (x.numel() == 0) return createLike(x, ValueType::DOUBLE, mr);
    const keel::i32 d = resolveDim(x, dim);
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    const keel::usize nNative = (d == 1) ? R : C;
    const keel::usize nOut    = (n > 0) ? keel::as<keel::usize>(n) : nNative;
    const keel::usize nLines  = (d == 1) ? C : R;

    Value out = (d == 1) ? Value::matrix(nOut, C, ValueType::DOUBLE, mr)
                         : Value::matrix(R, nOut, ValueType::DOUBLE, mr);
    const keel::f64 *src = x.doubleData();
    keel::f64 *dst = out.doubleDataMut();

    keel::Array<keel::f64> inBuf(nOut, 0.0, mr);
    keel::Array<keel::f64> outBuf(nOut, 0.0, mr);
    const keel::usize copyN = keel::min(nOut, nNative);

    for (keel::usize lineIdx = 0; lineIdx < nLines; ++lineIdx) {
        if (d == 1) {
            const keel::f64 *colSrc = src + lineIdx * R;
            for (keel::usize i = 0; i < copyN; ++i) inBuf[i] = colSrc[i];
        } else {
            for (keel::usize i = 0; i < copyN; ++i) inBuf[i] = src[lineIdx + i * R];
        }
        for (keel::usize i = copyN; i < nOut; ++i) inBuf[i] = 0.0;

        if (inverse) {
            (void)keel::signal::idct(inBuf.as_ref(), outBuf.as_mut_ref(), type, mr);
        } else {
            (void)keel::signal::dct(inBuf.as_ref(), outBuf.as_mut_ref(), type, mr);
        }

        if (d == 1) {
            keel::f64 *colDst = dst + lineIdx * nOut;
            for (keel::usize i = 0; i < nOut; ++i) colDst[i] = outBuf[i];
        } else {
            for (keel::usize i = 0; i < nOut; ++i) dst[lineIdx + i * R] = outBuf[i];
        }
    }
    return out;
}

} // anonymous

Value dct(const Value &x, keel::Allocator *mr) {
    const keel::usize N = x.numel();
    auto r = createLike(x, ValueType::DOUBLE, mr);
    if (N == 0) return r;
    (void)keel::signal::dct(keel::ArrayRef<keel::f64>(x.doubleData(), N),
                            keel::MutArrayRef<keel::f64>(r.doubleDataMut(), N), 2, mr);
    return r;
}

Value idct(const Value &x, keel::Allocator *mr) {
    const keel::usize N = x.numel();
    auto r = createLike(x, ValueType::DOUBLE, mr);
    if (N == 0) return r;
    (void)keel::signal::idct(keel::ArrayRef<keel::f64>(x.doubleData(), N),
                             keel::MutArrayRef<keel::f64>(r.doubleDataMut(), N), 2, mr);
    return r;
}

Value dct(const Value &x, keel::i32 n, keel::i32 dim, keel::Allocator *mr) {
    return runDctLines(x, n, dim, 2, false, mr);
}

Value idct(const Value &x, keel::i32 n, keel::i32 dim, keel::Allocator *mr) {
    return runDctLines(x, n, dim, 2, true, mr);
}

Value dctTyped(const Value &x, keel::i32 n, keel::i32 dim, keel::f64 type, bool inverse,
               keel::Allocator *mr) {
    const keel::i32 t = (type == 1.0) ? 1 : (type == 3.0) ? 3 : (type == 4.0) ? 4 : 2;
    return runDctLines(x, n, dim, t, inverse, mr);
}

} // namespace numkit::signal
