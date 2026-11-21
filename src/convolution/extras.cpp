// toolboxes/signal/src/convolution/extras.cpp
//
// cconv / convmtx / xcorr2 / finddelay / alignsignals.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/convolution/extras.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <tuple>
#include <numkit/signal/convolution/convolution.hpp>     // xcorr
#include <numkit/signal/transforms/fft.hpp>              // fft / ifft

#include <algorithm>
#include <cmath>
#include <cstring>

#include "signal_detail.hpp"     // shared argcheck cores

namespace numkit::signal {

namespace {

double readReal(const Value &v, keel::usize i)
{
    return v.elemAsDouble(i);
}

} // namespace

// ── cconv ─────────────────────────────────────────────────────────────
// MATLAB default for cconv(x, y) (no N): N = length(x) + length(y) - 1
// — the linear-convolution length. The 3-arg form cconv(x, y, n) does
// true circular convolution with period n. See BUGS.md #33.
Value cconv(const Value &x, const Value &y, keel::usize n, keel::Allocator *mr)
{
    const keel::usize nx = x.numel(), ny = y.numel();
    if (n == 0) {
        if (nx == 0 || ny == 0) n = 0;
        else                    n = nx + ny - 1;
    }
    auto out = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    double *dst = out.doubleDataMut();
    std::fill(dst, dst + n, 0.0);

    // Fast path: when n ≥ nx + ny - 1 (the default form), circular
    // conv of period n equals linear conv truncated/padded to n. We
    // pad x and y to length nextPow2(n), FFT both, multiply, IFFT,
    // and copy the first n samples. Closes the ~30× perf gap to
    // FFTW-backed Octave / MATLAB.
    if (nx > 0 && ny > 0 && n >= nx + ny - 1) {
        // Promote both to length-n column vectors so fft picks up the
        // axis. Pad with zeros past nx / ny.
        auto xp = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        auto yp = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        double *xd = xp.doubleDataMut();
        double *yd = yp.doubleDataMut();
        std::fill(xd, xd + n, 0.0);
        std::fill(yd, yd + n, 0.0);
        for (keel::usize i = 0; i < nx; ++i) xd[i] = readReal(x, i);
        for (keel::usize i = 0; i < ny; ++i) yd[i] = readReal(y, i);
        // nk's fft pads non-pow2 to nextPow2 internally, so for
        // arbitrary n this would corrupt the spectrum. We only enter
        // this branch when we don't need true length-n circular conv;
        // padding to any size ≥ nx+ny-1 yields the same first-n
        // samples of the linear conv. Force length-nextPow2 explicitly
        // for predictable behaviour.
        const keel::usize fftLen = [&]() {
            keel::usize r = 1; while (r < n) r <<= 1; return r;
        }();
        Value X = fft(xp, keel::as<int>(fftLen), /*dim=*/0, mr);
        Value Y = fft(yp, keel::as<int>(fftLen), /*dim=*/0, mr);
        // Pointwise multiply.
        auto Z = Value::complexMatrix(fftLen, 1, mr);
        const keel::c64 *Xc = X.complexData();
        const keel::c64 *Yc = Y.complexData();
        keel::c64 *Zc = Z.complexDataMut();
        for (keel::usize k = 0; k < fftLen; ++k) Zc[k] = Xc[k] * Yc[k];
        Value z = ifft(Z, /*n=*/-1, /*dim=*/0, mr);
        // ifft can return REAL when the spectrum is conjugate-symmetric.
        if (z.type() == ValueType::COMPLEX) {
            const keel::c64 *zd = z.complexData();
            for (keel::usize k = 0; k < n; ++k) dst[k] = zd[k].real();
        } else {
            const double *zd = z.doubleData();
            for (keel::usize k = 0; k < n; ++k) dst[k] = zd[k];
        }
        return out;
    }

    // Slow path: n < nx + ny - 1 means actual aliased circular conv;
    // use the direct O(n²) loop.
    for (keel::usize k = 0; k < n; ++k) {
        double s = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            const double xi = (i < nx) ? readReal(x, i) : 0.0;
            const keel::usize j = (k + n - i) % n;     // (k - i) mod n
            const double yj = (j < ny) ? readReal(y, j) : 0.0;
            s += xi * yj;
        }
        dst[k] = s;
    }
    return out;
}

// ── convmtx ───────────────────────────────────────────────────────────
// MATLAB shape rules (from `help convmtx`):
//   * h is a row    → returns n × (n+nh-1) matrix; row k is h shifted
//     right by k (so X*h_col == conv(x, h)).
//   * h is a column → returns (n+nh-1) × n matrix; column k is h
//     shifted down by k.
// See BUGS.md #34.
//
// Type contract (probed R2025b): h accepts double / single / complex /
// integer classes but NOT logical; single and complex results are
// class-preserving (the matrix cells are copies of h, so the exact
// element bits survive). Integer classes would need integer-class
// outputs which the Value kernel layer cannot allocate yet
// (bugs/opened/signal/integer-class-outputs.md).
namespace {

void convmtx_argcheck(const Value &h, keel::usize n)
{
    if (n == 0)
        throw Error("convmtx: n must be positive",
                     0, 0, "convmtx", "", "numkit:convmtx:badN");
    const keel::usize nh = h.numel();
    if (nh == 0)
        throw Error("convmtx: h must be non-empty",
                     0, 0, "convmtx", "", "numkit:convmtx:emptyH");
    detail::requireVectorInput(h, "convmtx", "h", 1);
    const ValueType ht = h.type();
    const bool isSingle = (ht == ValueType::SINGLE);
    const bool isComplex = (ht == ValueType::COMPLEX);
    const bool isInt = isIntegerType(ht);
    if (ht != ValueType::DOUBLE && !isSingle && !isComplex && !isInt) {
        // The reference accepts every numeric class EXCEPT logical.
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

Value convmtx(const Value &h, keel::usize n, keel::Allocator *mr)
{
    convmtx_argcheck(h, n);
    const keel::usize nh = h.numel();
    const ValueType ht = h.type();

    // Default to row-form when h is a vector with rows() == 1 OR is
    // 1-D / scalar. Column form only when h is explicitly Nx1 (rows>1).
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
                std::memcpy(dst + ((col + i) + col * outRows) * es, src + i * es, es);
    } else {
        for (keel::usize row = 0; row < n; ++row)
            for (keel::usize i = 0; i < nh; ++i)
                std::memcpy(dst + (row + (row + i) * outRows) * es, src + i * es, es);
    }
    return out;
}

// ── xcorr2 ────────────────────────────────────────────────────────────
// Direct O(rA*cA*rB*cB). Output shape (rA+rB-1) × (cA+cB-1).
// Type contract (probed R2025b): numeric + logical inputs pass (result
// double), N-D input is rejected, single and complex are honest kernel
// gaps (bugs/opened/signal/single-float-inputs.md,
// bugs/opened/signal/complex-kernel-gaps.md).
Value xcorr2(const Value &A, const Value &B, keel::Allocator *mr)
{
    xcorr2_argcheck(A, B);
    const bool anySingle = (A.type() == ValueType::SINGLE) || (B.type() == ValueType::SINGLE);
    Value aHold, bHold;
    if (!A.isComplex() && (A.isLogical() || isIntegerType(A.type()) || A.type() == ValueType::SINGLE))
        aHold = [&]() {
            const auto &d = A.dims();
            Value r = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < r.numel(); ++i)
                r.doubleDataMut()[i] = A.elemAsDouble(i);
            return r;
        }();
    if (!B.isComplex() && (B.isLogical() || isIntegerType(B.type()) || B.type() == ValueType::SINGLE))
        bHold = [&]() {
            const auto &d = B.dims();
            Value r = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < r.numel(); ++i)
                r.doubleDataMut()[i] = B.elemAsDouble(i);
            return r;
        }();
    const Value &Av = aHold.isEmpty() ? A : aHold;
    const Value &Bv = bHold.isEmpty() ? B : bHold;

    const keel::usize rA = Av.dims().rows(), cA = Av.dims().cols();
    const keel::usize rB = Bv.dims().rows(), cB = Bv.dims().cols();
    const bool anyCplx = Av.isComplex() || Bv.isComplex();
    if (rA == 0 || cA == 0 || rB == 0 || cB == 0)
        return Value::matrix(0, 0, anyCplx ? ValueType::COMPLEX : (anySingle ? ValueType::SINGLE : ValueType::DOUBLE), mr);
    const keel::usize rOut = rA + rB - 1;
    const keel::usize cOut = cA + cB - 1;
    if (anyCplx) {
        auto outC = Value::matrix(rOut, cOut, ValueType::COMPLEX, mr);
        keel::c64 *dstC = outC.complexDataMut();
        std::fill(dstC, dstC + rOut * cOut, keel::c64(0.0, 0.0));
        auto getC = [](const Value &v, keel::usize r, keel::usize c, keel::usize rows) -> keel::c64 {
            const keel::usize idx = r + c * rows;
            return v.isComplex() ? v.complexData()[idx] : keel::c64(v.elemAsDouble(idx), 0.0);
        };
        for (keel::usize i = 0; i < rA; ++i) {
            for (keel::usize j = 0; j < cA; ++j) {
                const keel::c64 a = getC(Av, i, j, rA);
                if (a == keel::c64(0.0, 0.0)) continue;
                for (keel::usize p = 0; p < rB; ++p) {
                    const long kRow = keel::as<long>(i) - keel::as<long>(p) + keel::as<long>(rB) - 1;
                    if (kRow < 0 || kRow >= keel::as<long>(rOut)) continue;
                    for (keel::usize q = 0; q < cB; ++q) {
                        const long kCol = keel::as<long>(j) - keel::as<long>(q) + keel::as<long>(cB) - 1;
                        if (kCol < 0 || kCol >= keel::as<long>(cOut)) continue;
                        dstC[kRow + kCol * rOut] += a * keel::conj(getC(Bv, p, q, rB));
                    }
                }
            }
        }
        return outC;
    }
    auto out = Value::matrix(rOut, cOut, ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    std::fill(dst, dst + rOut * cOut, 0.0);

    // FFT-based fast path: cross-correlation == conv2(A, flip(B)). Pad
    // A (zero) and the flipped B into rOut × cOut grids, fft2 both,
    // multiply pointwise, ifft2, take real part. Threshold below the
    // FFT path's startup cost — for tiny problems direct is faster.
    const keel::usize cellsOut = rOut * cOut;
    const keel::usize cellsDirect = rA * cA * rB * cB;
    const bool useFft = cellsDirect > 8192 && cellsOut >= 64;

    if (useFft) {
        auto Ap = Value::matrix(rOut, cOut, ValueType::DOUBLE, mr);
        auto Bf = Value::matrix(rOut, cOut, ValueType::DOUBLE, mr);
        double *ad = Ap.doubleDataMut();
        double *bd = Bf.doubleDataMut();
        std::fill(ad, ad + cellsOut, 0.0);
        std::fill(bd, bd + cellsOut, 0.0);
        // Place A in the top-left.
        for (keel::usize j = 0; j < cA; ++j)
            for (keel::usize i = 0; i < rA; ++i)
                ad[i + j * rOut] = Av(i, j);
        // Place flipped B (both axes) in the top-left.
        for (keel::usize j = 0; j < cB; ++j)
            for (keel::usize i = 0; i < rB; ++i)
                bd[i + j * rOut] = Bv(rB - 1 - i, cB - 1 - j);

        Value FA = fft2(Ap, -1, -1, mr);
        Value FB = fft2(Bf, -1, -1, mr);
        const keel::c64 *fa = FA.complexData();
        const keel::c64 *fb = FB.complexData();
        auto Z = Value::complexMatrix(rOut, cOut, mr);
        keel::c64 *zd = Z.complexDataMut();
        for (keel::usize k = 0; k < cellsOut; ++k) zd[k] = fa[k] * fb[k];
        Value z = ifft2(Z, -1, -1, mr);
        if (z.type() == ValueType::COMPLEX) {
            const keel::c64 *zr = z.complexData();
            for (keel::usize k = 0; k < cellsOut; ++k) dst[k] = zr[k].real();
        } else {
            std::memcpy(dst, z.doubleData(), cellsOut * sizeof(double));
        }
        return detail::narrowDoubleToSingleIf(std::move(out), anySingle, mr);
    }

    // Direct path for small inputs.
    for (keel::usize i = 0; i < rA; ++i) {
        for (keel::usize j = 0; j < cA; ++j) {
            const double a = Av(i, j);
            if (a == 0.0) continue;
            for (keel::usize p = 0; p < rB; ++p) {
                const long kRow = keel::as<long>(i) - keel::as<long>(p) + keel::as<long>(rB) - 1;
                if (kRow < 0 || kRow >= keel::as<long>(rOut)) continue;
                for (keel::usize q = 0; q < cB; ++q) {
                    const long kCol = keel::as<long>(j) - keel::as<long>(q) + keel::as<long>(cB) - 1;
                    if (kCol < 0 || kCol >= keel::as<long>(cOut)) continue;
                    dst[kRow + kCol * rOut] += a * Bv(p, q);
                }
            }
        }
    }
    return detail::narrowDoubleToSingleIf(std::move(out), anySingle, mr);
}

// ── finddelay ─────────────────────────────────────────────────────────
long finddelay(const Value &x, const Value &y, long max_lag, keel::Allocator *mr)
{
    auto [c, lags] = xcorr(x, y, mr);
    const keel::usize n = c.numel();
    const double *cp = c.doubleData();
    const double *lp = lags.doubleData();
    long best_lag = 0;
    double best_mag = -1.0;
    for (keel::usize i = 0; i < n; ++i) {
        const long lag = keel::as<long>(lp[i]);
        if (max_lag > 0 && keel::abs(lag) > max_lag) continue;
        const double mag = keel::abs(cp[i]);
        if (mag > best_mag) {
            best_mag = mag;
            best_lag = lag;
        }
    }
    // MATLAB convention: positive lag means y is delayed relative to x
    //   → finddelay returns -lag_of_xcorr_peak.
    return -best_lag;
}

// ── alignsignals ──────────────────────────────────────────────────────
std::tuple<Value, Value>
alignsignals(const Value &x, const Value &y, long max_lag, keel::Allocator *mr)
{
    const long d = finddelay(x, y, max_lag, mr);
    const keel::usize nx = x.numel(), ny = y.numel();

    // Lead-pad whichever signal is "leading" so peaks line up:
    //   d > 0: y is delayed by d → prepend d zeros to x.
    //   d < 0: x is delayed by |d| → prepend |d| zeros to y.
    const keel::usize pad_x = (d > 0) ? keel::as<keel::usize>(d) : 0;
    const keel::usize pad_y = (d < 0) ? keel::as<keel::usize>(-d) : 0;
    // Then trail-pad whichever ends short so both outputs are same length.
    const keel::usize finalLen = std::max(nx + pad_x, ny + pad_y);

    auto orient = [&](const Value &src, keel::usize lead_pad) {
        const bool isRow = (src.dims().rows() == 1);
        auto out = isRow
                    ? Value::matrix(1, finalLen, ValueType::DOUBLE, mr)
                    : Value::matrix(finalLen, 1, ValueType::DOUBLE, mr);
        double *dst = out.doubleDataMut();
        std::fill(dst, dst + finalLen, 0.0);
        const keel::usize n = src.numel();
        for (keel::usize i = 0; i < n; ++i)
            dst[lead_pad + i] = src.elemAsDouble(i);
        return out;
    };
    return std::make_tuple(orient(x, pad_x), orient(y, pad_y));
}

} // namespace numkit::signal
