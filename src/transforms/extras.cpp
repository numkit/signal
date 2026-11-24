// toolboxes/signal/src/transforms/extras.cpp
// dftmtx / bitrevorder / dst / idst / rceps / cceps / icceps
// (delegates numeric kernels to keel::signal).

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/transforms/extras.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../signal_detail.hpp"
#include "extras_detail.hpp"

namespace numkit::signal {

keel::usize bitReverse(keel::usize v, keel::usize bits)
{
    keel::usize r = 0;
    for (keel::usize i = 0; i < bits; ++i) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

namespace {

bool isPow2(keel::usize n) { return n > 0 && (n & (n - 1)) == 0; }

Value realLike(const Value &x, keel::usize n, keel::Allocator *mr)
{
    const bool isRow = (x.dims().rows() == 1 && x.dims().cols() > 1);
    return isRow ? Value::matrix(1, n, ValueType::DOUBLE, mr)
                 : Value::matrix(n, 1, ValueType::DOUBLE, mr);
}

keel::Array<keel::f64> readReal(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> out(n, mr);
    for (keel::usize i = 0; i < n; ++i) out[i] = x.elemAsDouble(i);
    return out;
}

void bitrevorder_argcheck(const Value &x)
{
    detail::requireNumericOrLogicalInput(x, "bitrevorder");
}

} // namespace

Value dftmtx(keel::usize N, keel::Allocator *mr)
{
    if (N == 0)
        throw Error("dftmtx: N must be positive",
                     0, 0, "dftmtx", "", "numkit:dftmtx:badN");
    auto out = Value::complexMatrix(N, N, mr);
    (void)keel::signal::dftmtx(N, keel::MutArrayRef<keel::c64>(out.complexDataMut(), N * N));
    return out;
}

Value bitrevorder(const Value &x, keel::Allocator *mr)
{
    bitrevorder_argcheck(x);
    const ValueType t = x.type();
    const keel::usize n = x.numel();
    if (!isPow2(n))
        throw Error("bitrevorder: input length must be a power of two",
                     0, 0, "bitrevorder", "", "numkit:bitrevorder:badLength");
    keel::usize bits = 0;
    for (keel::usize v = n; v > 1; v >>= 1) ++bits;

    const bool isRow = (x.dims().rows() == 1);
    auto out = Value::matrix(isRow ? 1 : n, isRow ? n : 1, t, mr);
    const keel::usize es = x.rawBytes() / n;
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(out.rawDataMut());
    for (keel::usize i = 0; i < n; ++i)
        keel::mem_copy(dst + bitReverse(i, bits) * es, src + i * es, es);
    return out;
}

Value dst(const Value &x, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readReal(x, scratch.allocator());
    (void)keel::signal::dst(xv.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N));
    return out;
}

Value idst(const Value &y, keel::Allocator *mr)
{
    const keel::usize N = y.numel();
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto yv = readReal(y, scratch.allocator());
    (void)keel::signal::idst(yv.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N));
    return out;
}

Value rceps(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    auto out = realLike(x, n, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readReal(x, scratch.allocator());
    (void)keel::signal::rceps(xv.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n),
                              scratch.allocator());
    return out;
}

keel::Pair<Value, Value> rcepsMinPhase(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    Value y = realLike(x, n, mr);
    Value ym = realLike(x, n, mr);
    if (n == 0) return {keel::move(y), keel::move(ym)};
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readReal(x, scratch.allocator());
    (void)keel::signal::rceps_minphase(
        xv.as_ref(),
        keel::MutArrayRef<keel::f64>(y.doubleDataMut(), n),
        keel::MutArrayRef<keel::f64>(ym.doubleDataMut(), n),
        scratch.allocator());
    return {keel::move(y), keel::move(ym)};
}

Value cceps(const Value &x, keel::f64 *ndOut, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (ndOut) *ndOut = 0.0;
    auto out = realLike(x, n, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readReal(x, scratch.allocator());
    (void)keel::signal::cceps(xv.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n),
                              ndOut, scratch.allocator());
    return out;
}

Value icceps(const Value &c, keel::Allocator *mr)
{
    const keel::usize n = c.numel();
    auto out = realLike(c, n, mr);
    if (n == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto cv = readReal(c, scratch.allocator());
    (void)keel::signal::icceps(cv.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n),
                               scratch.allocator());
    return out;
}

} // namespace numkit::signal
