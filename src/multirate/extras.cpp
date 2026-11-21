// toolboxes/signal/src/multirate/extras.cpp
//
// upfirdn / interp / intfilt / fftfilt.
//
// fftfilt uses the FFT-conv helpers from dsp_helpers.hpp; the others
// reuse intfilt (a sinc-with-Hamming-window FIR) and the existing
// filter() / upsample() / downsample() primitives.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/signal/multirate/extras.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/signal/digital_filtering/filter.hpp>
#include <numkit/signal/filter_design/filter_design.hpp>   // firls
#include <numkit/signal/multirate/multirate.hpp>

#include "../dsp_helpers.hpp"   // fftRadix2, fillFftTwiddles, nextPow2, Complex

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace {

// 1-D real vector accessor that handles both row and column orientation.
keel::f64 readReal(const Value &v, keel::usize i) { return v.elemAsDouble(i); }

// Allocate a real row/column vector matching the orientation of `like`.
Value sameOrientReal(const Value &like, keel::usize n, keel::Allocator *mr)
{
    const bool isRow = (like.dims().rows() == 1);
    return isRow
            ? Value::matrix(1, n, ValueType::DOUBLE, mr)
            : Value::matrix(n, 1, ValueType::DOUBLE, mr);
}

void intfilt_argcheck(keel::usize r, keel::usize n, keel::f64 alpha)
{
    if (r < 1)
        throw Error("intfilt: r must be >= 1",
                     0, 0, "intfilt", "", "numkit:intfilt:badR");
    if (n < 1)
        throw Error("intfilt: n must be >= 1",
                     0, 0, "intfilt", "", "numkit:intfilt:badN");
    if (!(alpha > 0.0) || !(alpha <= 1.0))
        throw Error("intfilt: alpha must be in (0, 1]",
                     0, 0, "intfilt", "", "numkit:intfilt:badAlpha");
}

void upfirdn_argcheck(keel::usize p, keel::usize q)
{
    if (p < 1 || q < 1)
        throw Error("upfirdn: p and q must be >= 1",
                     0, 0, "upfirdn", "", "numkit:upfirdn:badPQ");
}

void interp_argcheck(const Value &x, keel::usize n)
{
    if (x.numel() < 2 * n + 1)
        throw Error(keel::format("interp: input 1 (x) length ({}) must be at least 2*n+1 ({})",
                                 x.numel(), 2 * n + 1),
                    0, 0, "interp", "", "numkit:interp:inputTooShort");
}

} // namespace

// ── intfilt (FIR design) ──────────────────────────────────────────────
Value intfilt(keel::usize r, keel::usize n, keel::f64 alpha, keel::Allocator *mr)
{
    intfilt_argcheck(r, n, alpha);

    // MATLAB intfilt(R, L, freqmult): a bandlimited interpolation FIR of
    // length 2*R*L-1, designed by LEAST SQUARES (firls) over a piecewise
    // band/amplitude spec — NOT a windowed sinc. Ported verbatim from
    // intfilt.m's bandlimited ('b') branch: passband amplitude R around
    // each image-free band, zero in the images, then h = firls(n-1, F*2, M).
    if (2 * n * r < 1) return Value::matrix(1, 1, ValueType::DOUBLE, mr);
    const keel::i64 Ntaps = keel::as<keel::i64>(2 * n * r) - 1;
    const keel::i32  order = keel::as<keel::i32>(Ntaps - 1);   // even (2RL-2)
    const keel::f64 R   = keel::as<keel::f64>(r);

    keel::Array<keel::f64> F(mr), M(mr);
    if (alpha == 1.0) {
        M = {R, R, 0.0, 0.0};
        F = {0.0, 1.0 / (2.0 * R), 1.0 / (2.0 * R), 0.5};
    } else {
        const keel::f64 hw = alpha / (2.0 * R);          // half passband width
        M = {R, R};
        F = {0.0, hw};
        for (keel::usize k = 1; keel::as<keel::f64>(k) / R <= 0.5 + 1e-9; ++k) {
            const keel::f64 f = keel::as<keel::f64>(k) / R;
            F.push_back(f - hw);
            F.push_back(f + hw);
            M.push_back(0.0);
            M.push_back(0.0);
        }
        if (F.back() > 0.5) F.back() = 0.5;
    }
    // nk's firls normalises frequency to Nyquist = 1 → scale F by 2.
    Value Fv = Value::matrix(1, F.size(), ValueType::DOUBLE, mr);
    Value Mv = Value::matrix(1, M.size(), ValueType::DOUBLE, mr);
    keel::f64 *fd = Fv.doubleDataMut();
    keel::f64 *md = Mv.doubleDataMut();
    for (keel::usize i = 0; i < F.size(); ++i) { fd[i] = F[i] * 2.0; md[i] = M[i]; }
    return firls(order, Fv, Mv, mr);
}

// ── upfirdn ───────────────────────────────────────────────────────────
// MATLAB / scipy upfirdn output length = ceil(((Lx-1)*p + Lh) / q).
// Note: upsample puts x[i] at position i*p so the upsampled signal
// has length (Lx-1)*p + 1 (NOT Lx*p -- no trailing zeros). Then
// linear convolution with h gives length (Lx-1)*p + Lh, and downsample
// by q yields ceil(/q) samples.
Value upfirdn(const Value &x, const Value &h, keel::usize p, keel::usize q, keel::Allocator *mr)
{
    upfirdn_argcheck(p, q);
    const keel::usize Lx = x.numel();
    const keel::usize Lh = h.numel();
    if (Lx == 0 || Lh == 0) return sameOrientReal(x, 0, mr);

    // Length after upsample (no trailing zeros): (Lx-1)*p + 1.
    const keel::usize upLen = (Lx - 1) * p + 1;
    // Linear convolution length.
    const keel::usize convLen = upLen + Lh - 1;
    // Downsampled output length.
    const keel::usize outLen = (convLen + q - 1) / q;   // ceil(convLen / q)

    keel::ArenaTemp local = keel::get_scratch(mr);
    auto convOut = keel::Array<keel::f64>(convLen, local.allocator());
    std::fill(convOut.begin(), convOut.end(), 0.0);

    // Direct upsample-then-convolve in one pass: x[i] sits at upsampled
    // position i*p, so each x[i] contributes h[j] to output position
    // i*p + j.
    for (keel::usize i = 0; i < Lx; ++i) {
        const keel::f64 xi = readReal(x, i);
        if (xi == 0.0) continue;
        const keel::usize base = i * p;
        for (keel::usize j = 0; j < Lh; ++j) {
            convOut[base + j] += xi * readReal(h, j);
        }
    }

    auto out = sameOrientReal(x, outLen, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize k = 0; k < outLen; ++k) {
        od[k] = convOut[k * q];
    }
    return out;
}

// ── interp ────────────────────────────────────────────────────────────
Value interp(const Value &x, keel::usize r, keel::usize n, keel::f64 alpha, keel::Allocator *mr)
{
    interp_argcheck(x, n);
    if (r == 1) {
        // Pass-through: copy x.
        auto out = sameOrientReal(x, x.numel(), mr);
        std::memcpy(out.doubleDataMut(), x.doubleData(), x.numel() * sizeof(keel::f64));
        return out;
    }
    auto h = intfilt(r, n, alpha, mr);
    auto y = upfirdn(x, h, r, 1, mr);
    // Trim group delay introduced by the symmetric FIR (length L=2nr+1
    // gives delay nr in upsampled samples). MATLAB's interp keeps the
    // first nx*r samples after the symmetric leading delay.
    const keel::usize target = x.numel() * r;
    const keel::usize leadDelay = n * r;
    auto trimmed = sameOrientReal(x, target, mr);
    keel::f64 *dst = trimmed.doubleDataMut();
    const keel::f64 *src = y.doubleData();
    const keel::usize srcN = y.numel();
    for (keel::usize i = 0; i < target; ++i) {
        const keel::usize j = i + leadDelay;
        dst[i] = (j < srcN) ? src[j] : 0.0;
    }
    return trimmed;
}

// ── fftfilt ───────────────────────────────────────────────────────────
// Overlap-add convolution. Choose block length L per typical heuristic:
// L ≈ 2 * nb (rounded up to power of 2). NFFT = L + nb - 1 → next power
// of 2.
Value fftfilt(const Value &b, const Value &x, keel::usize nfft, keel::Allocator *mr)
{
    const keel::usize nb = b.numel();
    const keel::usize nx = x.numel();
    if (nb == 0 || nx == 0)
        return sameOrientReal(x, 0, mr);

    keel::usize blockL;
    if (nfft >= nb) {
        blockL = nfft - nb + 1;
    } else {
        // Heuristic: block ≈ max(nb, 32), then NFFT next pow2 of L+nb-1.
        blockL = std::max<keel::usize>(nb, 32);
        nfft = nextPow2(blockL + nb - 1);
        blockL = nfft - nb + 1;
    }
    if (blockL < 1) blockL = 1;
    if (nfft < blockL + nb - 1)
        nfft = nextPow2(blockL + nb - 1);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const bool isCx = b.isComplex() || x.isComplex();
    // Precompute B(f) = FFT(b padded to nfft).
    auto Bf = keel::Array<keel::c64>(nfft, scratch.allocator());
    for (keel::usize i = 0; i < nb; ++i)
        Bf[i] = b.isComplex() ? b.complexElem(i) : keel::c64(readReal(b, i), 0.0);
    auto W_fwd = keel::Array<keel::c64>(nfft / 2, scratch.allocator());
    auto W_inv = keel::Array<keel::c64>(nfft / 2, scratch.allocator());
    fillFftTwiddles(W_fwd.data(), nfft, +1);
    fillFftTwiddles(W_inv.data(), nfft, -1);
    fftRadix2(Bf.data(), nfft, W_fwd.data());

    auto Xf  = keel::Array<keel::c64>(nfft, scratch.allocator());
    const keel::f64 invN = 1.0 / keel::as<keel::f64>(nfft);

    if (isCx) {
        const bool row = (x.dims().rows() == 1 && x.dims().cols() >= 1);
        auto out = Value::complexMatrix(row ? 1 : nx, row ? nx : 1, mr);
        keel::c64 *dst = out.complexDataMut();
        std::fill(dst, dst + nx, keel::c64(0.0, 0.0));
        for (keel::usize pos = 0; pos < nx; pos += blockL) {
            const keel::usize L = std::min(blockL, nx - pos);
            std::fill(Xf.begin(), Xf.end(), keel::c64(0, 0));
            for (keel::usize i = 0; i < L; ++i)
                Xf[i] = x.isComplex() ? x.complexElem(pos + i) : keel::c64(readReal(x, pos + i), 0.0);
            fftRadix2(Xf.data(), nfft, W_fwd.data());
            for (keel::usize i = 0; i < nfft; ++i)
                Xf[i] *= Bf[i];
            fftRadix2(Xf.data(), nfft, W_inv.data());
            const keel::usize blockOut = L + nb - 1;
            for (keel::usize i = 0; i < blockOut; ++i) {
                const keel::usize outIdx = pos + i;
                if (outIdx < nx)
                    dst[outIdx] += Xf[i] * invN;
            }
        }
        return out;
    }

    auto out = sameOrientReal(x, nx, mr);
    keel::f64 *dst = out.doubleDataMut();
    std::fill(dst, dst + nx, 0.0);

    for (keel::usize pos = 0; pos < nx; pos += blockL) {
        const keel::usize L = std::min(blockL, nx - pos);
        std::fill(Xf.begin(), Xf.end(), keel::c64(0, 0));
        for (keel::usize i = 0; i < L; ++i)
            Xf[i] = keel::c64(readReal(x, pos + i), 0.0);
        fftRadix2(Xf.data(), nfft, W_fwd.data());
        for (keel::usize i = 0; i < nfft; ++i)
            Xf[i] *= Bf[i];
        fftRadix2(Xf.data(), nfft, W_inv.data());
        const keel::usize blockOut = L + nb - 1;
        for (keel::usize i = 0; i < blockOut; ++i) {
            const keel::usize outIdx = pos + i;
            if (outIdx < nx)
                dst[outIdx] += Xf[i].real() * invN;
        }
    }
    return out;
}

} // namespace numkit::signal
