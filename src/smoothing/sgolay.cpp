// toolboxes/signal/src/smoothing/sgolay.cpp
//
// Savitzky-Golay projection/differentiation matrices and filtering delegating to keel::signal.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/smoothing/sgolay.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

void sgolay_argcheck(keel::i32 order, keel::i32 framelen) {
    if (framelen <= 0)
        throw Error("sgolay: framelen must be positive",
                    0, 0, "sgolay", "", "numkit:sgolay:badArg");
    if ((framelen & 1) == 0)
        throw Error("sgolay: framelen must be odd",
                    0, 0, "sgolay", "", "numkit:sgolay:evenFramelen");
    if (order < 0)
        throw Error("sgolay: order must be non-negative",
                    0, 0, "sgolay", "", "numkit:sgolay:badArg");
    if (order >= framelen)
        throw Error("sgolay: order must be less than framelen",
                    0, 0, "sgolay", "", "numkit:sgolay:orderTooHigh");
}

Value sgolayfiltImpl(const Value &x, keel::i32 order, keel::i32 framelen,
                     keel::ArrayRef<keel::f64> weights, keel::i32 dim, keel::Allocator *mr) {
    if (x.type() == ValueType::COMPLEX)
        throw Error("sgolayfilt: complex inputs are not supported",
                    0, 0, "sgolayfilt", "", "numkit:sgolayfilt:complex");
    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("sgolayfilt: N-D (>2) inputs are not supported",
                    0, 0, "sgolayfilt", "", "numkit:sgolayfilt:ndims");

    const keel::usize R = d.rows(), C = d.cols();
    keel::i32 useDim = dim;
    if (useDim <= 0) useDim = (R > 1) ? 1 : 2;
    if (useDim != 1 && useDim != 2)
        throw Error("sgolayfilt: dim must be 1 or 2",
                    0, 0, "sgolayfilt", "", "numkit:sgolayfilt:badDim");

    const keel::i32 n = keel::as<keel::i32>(useDim == 1 ? R : C);
    if (n < framelen)
        throw Error("sgolayfilt: signal length along dim must be >= framelen",
                    0, 0, "sgolayfilt", "", "numkit:sgolayfilt:tooShort");
    sgolay_argcheck(order, framelen);

    auto out = createLike(x, ValueType::DOUBLE, mr);
    if (x.numel() == 0) return out;
    keel::f64 *dst = out.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src(x.numel(), 0.0, scratch.allocator());
    for (keel::usize i = 0; i < x.numel(); ++i) src[i] = x.elemAsDouble(i);

    if (useDim == 1) {
        for (keel::usize c = 0; c < C; ++c) {
            const auto st = keel::signal::sgolayfilt(
                keel::ArrayRef<keel::f64>(src.data() + c * R, R), order, framelen,
                keel::MutArrayRef<keel::f64>(dst + c * R, R), weights, scratch.allocator());
            if (st == keel::signal::SignalStatus::kSingular)
                throw Error("sgolay: singular normal equations (framelen too small for order)",
                            0, 0, "sgolay", "", "numkit:sgolay:singular");
        }
    } else {
        keel::Array<keel::f64> rbuf(C, 0.0, scratch.allocator());
        keel::Array<keel::f64> obuf(C, 0.0, scratch.allocator());
        for (keel::usize r = 0; r < R; ++r) {
            for (keel::usize c = 0; c < C; ++c) rbuf[c] = src[r + c * R];
            const auto st = keel::signal::sgolayfilt(
                rbuf.as_ref(), order, framelen, obuf.as_mut_ref(), weights, scratch.allocator());
            if (st == keel::signal::SignalStatus::kSingular)
                throw Error("sgolay: singular normal equations (framelen too small for order)",
                            0, 0, "sgolay", "", "numkit:sgolay:singular");
            for (keel::usize c = 0; c < C; ++c) dst[r + c * R] = obuf[c];
        }
    }
    return out;
}

} // namespace

Value sgolay(keel::i32 order, keel::i32 framelen, keel::Allocator *mr) {
    sgolay_argcheck(order, framelen);
    const keel::usize n = keel::as<keel::usize>(framelen);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> B(n * n, 0.0, scratch.allocator());
    if (keel::signal::sgolay(order, framelen, B.as_mut_ref(), {}, {}, scratch.allocator()) ==
        keel::signal::SignalStatus::kSingular) {
        throw Error("sgolay: singular normal equations (framelen too small for order)",
                    0, 0, "sgolay", "", "numkit:sgolay:singular");
    }
    auto out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < n; ++j)
            dst[j * n + i] = B[i * n + j];
    return out;
}

Value sgolayDiff(keel::i32 order, keel::i32 framelen, keel::Allocator *mr) {
    sgolay_argcheck(order, framelen);
    const keel::usize n = keel::as<keel::usize>(framelen);
    const keel::usize p = keel::as<keel::usize>(order + 1);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> G(n * p, 0.0, scratch.allocator());
    if (keel::signal::sgolay(order, framelen, {}, G.as_mut_ref(), {}, scratch.allocator()) ==
        keel::signal::SignalStatus::kSingular) {
        throw Error("sgolay: singular normal equations (framelen too small for order)",
                    0, 0, "sgolay", "", "numkit:sgolay:singular");
    }
    auto out = Value::matrix(n, p, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < p; ++j)
            dst[j * n + i] = G[i * p + j];
    return out;
}

Value sgolayfilt(const Value &x, keel::i32 order, keel::i32 framelen, keel::Allocator *mr) {
    return sgolayfiltImpl(x, order, framelen, {}, 0, mr);
}

Value sgolayfilt(const Value &x, keel::i32 order, keel::i32 framelen,
                 const Value &weights, keel::i32 dim, keel::Allocator *mr) {
    if (weights.isEmpty())
        return sgolayfiltImpl(x, order, framelen, {}, dim, mr);
    if (weights.type() == ValueType::COMPLEX)
        throw Error("sgolayfilt: weights must be real and positive",
                    0, 0, "sgolayfilt", "", "numkit:sgolayfilt:weights");
    if (keel::as<keel::i32>(weights.numel()) != framelen)
        throw Error("sgolayfilt: weights must have length framelen",
                    0, 0, "sgolayfilt", "", "numkit:sgolayfilt:weightsLen");
    keel::ArenaTemp wa = keel::get_scratch(mr);
    keel::Array<keel::f64> w(keel::as<keel::usize>(framelen), 0.0, wa.allocator());
    for (keel::i32 i = 0; i < framelen; ++i) {
        const keel::f64 wi = weights.elemAsDouble(i);
        if (!(wi > 0.0))
            throw Error("sgolayfilt: weights must be real and positive",
                        0, 0, "sgolayfilt", "", "numkit:sgolayfilt:weights");
        w[keel::as<keel::usize>(i)] = wi;
    }
    return sgolayfiltImpl(x, order, framelen, w.as_ref(), dim, mr);
}

} // namespace numkit::signal
