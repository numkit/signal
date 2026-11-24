// toolboxes/signal/src/convolution/extras.cpp
//
// cconv / convmtx / xcorr2 / finddelay / alignsignals (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/convolution/extras.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "signal_detail.hpp"

namespace numkit::signal {

namespace {

keel::Array<keel::f64> readRealVec(const Value &v, keel::Allocator *mr)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, mr);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void convmtx_argcheck(const Value &h, keel::usize n)
{
    if (n == 0)
        throw Error("convmtx: n must be positive",
                     0, 0, "convmtx", "", "numkit:convmtx:badN");
    if (h.numel() == 0)
        throw Error("convmtx: h must be non-empty",
                     0, 0, "convmtx", "", "numkit:convmtx:emptyH");
    detail::requireVectorInput(h, "convmtx", "h", 1);
    const ValueType ht = h.type();
    if (ht != ValueType::DOUBLE && ht != ValueType::SINGLE && ht != ValueType::COMPLEX && !isIntegerType(ht)) {
        detail::throwInputType(h, "convmtx", "h", 1, "numeric");
    }
}

void xcorr2_argcheck(const Value &A, const Value &B)
{
    detail::requireNumericSignalInput(A, "xcorr2", "A", 1);
    detail::requireNumericSignalInput(B, "xcorr2", "B", 2);
    if (A.dims().ndim() > 2 || B.dims().ndim() > 2)
        throw Error("xcorr2: N-D arrays are not supported",
                     0, 0, "xcorr2", "", "numkit:xcorr2:nd");
}

} // namespace

Value cconv(const Value &x, const Value &y, keel::usize n, keel::Allocator *mr)
{
    const keel::usize nx = x.numel(), ny = y.numel();
    if (n == 0) n = (nx == 0 || ny == 0) ? 0 : (nx + ny - 1);
    auto out = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    auto yv = readRealVec(y, scratch.allocator());
    (void)keel::signal::cconv(xv.as_ref(), yv.as_ref(), n,
                              keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n),
                              scratch.allocator());
    return out;
}

Value convmtx(const Value &h, keel::usize n, keel::Allocator *mr)
{
    convmtx_argcheck(h, n);
    const keel::usize nh = h.numel();
    const ValueType ht = h.type();
    const bool hIsColumn = (h.dims().rows() > 1 && h.dims().cols() == 1);
    const keel::usize outRows = hIsColumn ? (n + nh - 1) : n;
    const keel::usize outCols = hIsColumn ? n : (n + nh - 1);
    auto out = Value::matrix(outRows, outCols, ht, mr);
    const keel::usize es = h.rawBytes() / nh;
    const char *src = static_cast<const char *>(h.rawData());
    char *dst = static_cast<char *>(out.rawDataMut());
    if (hIsColumn) {
        for (keel::usize col = 0; col < n; ++col)
            for (keel::usize i = 0; i < nh; ++i)
                keel::mem_copy(dst + ((col + i) + col * outRows) * es, src + i * es, es);
    } else {
        for (keel::usize row = 0; row < n; ++row)
            for (keel::usize i = 0; i < nh; ++i)
                keel::mem_copy(dst + (row + (row + i) * outRows) * es, src + i * es, es);
    }
    return out;
}

Value xcorr2(const Value &A, const Value &B, keel::Allocator *mr)
{
    xcorr2_argcheck(A, B);
    const bool anySingle = (A.type() == ValueType::SINGLE) || (B.type() == ValueType::SINGLE);
    const bool anyCplx = A.isComplex() || B.isComplex();
    const keel::usize rA = A.dims().rows(), cA = A.dims().cols();
    const keel::usize rB = B.dims().rows(), cB = B.dims().cols();
    if (rA == 0 || cA == 0 || rB == 0 || cB == 0)
        return Value::matrix(0, 0, anyCplx ? ValueType::COMPLEX : (anySingle ? ValueType::SINGLE : ValueType::DOUBLE), mr);

    const keel::usize rOut = rA + rB - 1;
    const keel::usize cOut = cA + cB - 1;
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (anyCplx) {
        auto toC = [&](const Value &v) {
            keel::Array<keel::c64> buf(v.numel(), scratch.allocator());
            if (v.isComplex()) keel::copy_n(buf.data(), v.complexData(), v.numel());
            else for (keel::usize i = 0; i < v.numel(); ++i) buf[i] = keel::c64(v.elemAsDouble(i), 0.0);
            return buf;
        };
        auto ac = toC(A);
        auto bc = toC(B);
        auto outC = Value::matrix(rOut, cOut, ValueType::COMPLEX, mr);
        (void)keel::signal::xcorr2_c64(
            ac.as_ref(), rA, cA, bc.as_ref(), rB, cB,
            keel::MutArrayRef<keel::c64>(outC.complexDataMut(), rOut * cOut));
        return outC;
    }

    auto ar = readRealVec(A, scratch.allocator());
    auto br = readRealVec(B, scratch.allocator());
    auto out = Value::matrix(rOut, cOut, ValueType::DOUBLE, mr);
    (void)keel::signal::xcorr2(
        ar.as_ref(), rA, cA, br.as_ref(), rB, cB,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), rOut * cOut),
        scratch.allocator());
    return detail::narrowDoubleToSingleIf(keel::move(out), anySingle, mr);
}

keel::i64 finddelay(const Value &x, const Value &y, keel::i64 max_lag, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    auto yv = readRealVec(y, scratch.allocator());
    keel::i64 delay = 0;
    (void)keel::signal::finddelay(xv.as_ref(), yv.as_ref(), max_lag, delay, scratch.allocator());
    return delay;
}

keel::Tuple<Value, Value>
alignsignals(const Value &x, const Value &y, keel::i64 max_lag, keel::Allocator *mr)
{
    const keel::i64 d = finddelay(x, y, max_lag, mr);
    const keel::usize nx = x.numel(), ny = y.numel();
    const keel::usize pad_x = (d > 0) ? keel::as<keel::usize>(d) : 0;
    const keel::usize pad_y = (d < 0) ? keel::as<keel::usize>(-d) : 0;
    const keel::usize finalLen = keel::max(nx + pad_x, ny + pad_y);

    auto orient = [&](const Value &src, keel::usize lead_pad) {
        const bool isRow = (src.dims().rows() == 1);
        auto out = isRow ? Value::matrix(1, finalLen, ValueType::DOUBLE, mr)
                         : Value::matrix(finalLen, 1, ValueType::DOUBLE, mr);
        keel::f64 *dst = out.doubleDataMut();
        keel::fill(dst, dst + finalLen, 0.0);
        for (keel::usize i = 0; i < src.numel(); ++i)
            dst[lead_pad + i] = src.elemAsDouble(i);
        return out;
    };
    return keel::make_tuple(orient(x, pad_x), orient(y, pad_y));
}

} // namespace numkit::signal
