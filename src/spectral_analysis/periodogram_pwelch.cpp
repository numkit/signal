// toolboxes/signal/src/spectral_analysis/periodogram_pwelch.cpp
//
// periodogram + pwelch. Split from spectral_analysis/. spectrogram lives in
// time_frequency/spectrogram.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/signal/internal/shims/specfun.hpp>
#include <numkit/signal/transforms/fft.hpp>

#include "../dsp_helpers.hpp"

#include <algorithm>
#include <cmath>

namespace numkit::signal {

namespace {

// Fill caller-provided buffer with Hamming window coefficients, same
// formula MATLAB uses. Buffer must have N elements; works with both
// std::vector and std::pmr::vector data().
void fillHammingWindow(keel::f64 *w, keel::usize N)
{
    if (N == 1) {
        w[0] = 1.0;
        return;
    }
    for (keel::usize i = 0; i < N; ++i)
        w[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (N - 1));
}

} // anonymous namespace

std::tuple<Value, Value>
periodogram(const Value &x, const Value &window, keel::usize nfft, keel::f64 fs, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    const bool isCplx = x.isComplex();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto win = keel::Array<keel::f64>(N, scratch.allocator());
    if (window.numel() == N) {
        const keel::f64 *w = window.doubleData();
        for (keel::usize i = 0; i < N; ++i)
            win[i] = w[i];
    } else {
        std::fill(win.begin(), win.end(), 1.0);
    }

    if (nfft == 0)
        nfft = std::max<keel::usize>(256, nextPow2(N));

    auto buf = keel::Array<keel::c64>(nfft, scratch.allocator());
    keel::f64 winPower = 0.0;
    if (isCplx) {
        const keel::c64 *xd = x.complexData();
        for (keel::usize i = 0; i < N; ++i) {
            buf[i] = xd[i] * win[i];
            winPower += win[i] * win[i];
        }
    } else {
        const keel::f64 *xd = x.doubleData();
        for (keel::usize i = 0; i < N; ++i) {
            buf[i] = keel::c64(xd[i] * win[i], 0.0);
            winPower += win[i] * win[i];
        }
    }

    const keel::c64 *spec;
    Value          Xv;
    if ((nfft & (nfft - 1)) == 0) {
        fftRadix2(scratch.allocator(), buf, 1);
        spec = buf.data();
    } else {
        Value xwin = Value::matrix(N, 1, isCplx ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
        if (isCplx) {
            keel::c64 *xw = xwin.complexDataMut();
            const keel::c64 *xd = x.complexData();
            for (keel::usize i = 0; i < N; ++i) xw[i] = xd[i] * win[i];
        } else {
            keel::f64 *xw = xwin.doubleDataMut();
            const keel::f64 *xd = x.doubleData();
            for (keel::usize i = 0; i < N; ++i) xw[i] = xd[i] * win[i];
        }
        Xv   = fft(xwin, keel::as<int>(nfft), /*dim=*/0, mr);
        spec = Xv.complexData();
    }

    const keel::f64 scale = 1.0 / (winPower * fs);
    if (isCplx) {
        auto Pxx = Value::matrix(nfft, 1, ValueType::DOUBLE, mr);
        auto F   = Value::matrix(nfft, 1, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < nfft; ++i) {
            Pxx.doubleDataMut()[i] = keel::norm(spec[i]) * scale;
            F.doubleDataMut()[i]   = fs * keel::as<keel::f64>(i) / keel::as<keel::f64>(nfft);
        }
        return std::make_tuple(std::move(Pxx), std::move(F));
    }

    const keel::usize nOut = nfft / 2 + 1;
    auto Pxx = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    auto F = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);

    for (keel::usize i = 0; i < nOut; ++i) {
        keel::f64 mag2 = keel::norm(spec[i]);
        if (i > 0 && i < nfft / 2)
            mag2 *= 2.0;
        Pxx.doubleDataMut()[i] = mag2 * scale;
        F.doubleDataMut()[i] = (fs / 2.0) * keel::as<keel::f64>(i)
                               / keel::as<keel::f64>(nOut - 1);
    }

    return std::make_tuple(std::move(Pxx), std::move(F));
}

Value
periodogramConf(const Value &Pxx, keel::f64 confidenceLevel, bool realInput, bool nfftEven,
                keel::Allocator *mr)
{
    const keel::usize  nf = Pxx.numel();
    const keel::f64 *p  = Pxx.doubleData();

    const keel::f64 alpha = 1.0 - confidenceLevel;
    const keel::f64 qLo   = 1.0 - alpha / 2.0;   // upper chi-square quantile → lower PSD bound
    const keel::f64 qHi   = alpha / 2.0;          // lower chi-square quantile → upper PSD bound

    // Closed-form chi-square inverse CDFs at the two coverage points:
    //   chi2inv(q, 2) = -2 ln(1-q)            (exponential / 2 DOF)
    //   chi2inv(q, 1) = (√2 · erfinv(q))²     (folded normal / 1 DOF)
    const keel::f64 c2Lo = -2.0 * std::log(1.0 - qLo);
    const keel::f64 c2Hi = -2.0 * std::log(1.0 - qHi);
    const keel::f64 eLo  = numkit::builtin::erfinv(Value::scalar(qLo, mr)).toScalar();
    const keel::f64 eHi  = numkit::builtin::erfinv(Value::scalar(qHi, mr)).toScalar();
    const keel::f64 c1Lo = 2.0 * eLo * eLo;
    const keel::f64 c1Hi = 2.0 * eHi * eHi;

    // PSD → [lower, upper] multipliers per degrees-of-freedom (Pxx · v / chi2inv).
    const keel::f64 r2Lo = 2.0 / c2Lo, r2Hi = 2.0 / c2Hi;   // v = 2 (interior bins)
    const keel::f64 r1Lo = 1.0 / c1Lo, r1Hi = 1.0 / c1Hi;   // v = 1 (real DC / Nyquist)

    Value   out = Value::matrix(nf, 2, ValueType::DOUBLE, mr);
    keel::f64 *o   = out.doubleDataMut();
    for (keel::usize i = 0; i < nf; ++i) {
        const bool oneDof = realInput && (i == 0 || (nfftEven && i + 1 == nf));
        o[i]      = p[i] * (oneDof ? r1Lo : r2Lo);   // column 0 — lower bound
        o[nf + i] = p[i] * (oneDof ? r1Hi : r2Hi);   // column 1 — upper bound
    }
    return out;
}

std::tuple<Value, Value>
pwelch(const Value &x, const Value &window, keel::usize noverlap, keel::usize nfft, keel::f64 fs, keel::Allocator *mr)
{
    const keel::usize nx = x.numel();
    const bool isCplx = x.isComplex();
    const keel::f64 *xd = isCplx ? nullptr : x.doubleData();
    const keel::c64 *xcd = isCplx ? x.complexData() : nullptr;

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    keel::usize winLen;
    keel::Array<keel::f64> win(scratch.allocator());
    if (window.numel() > 0) {
        winLen = window.numel();
        win.resize(winLen);
        const keel::f64 *w = window.doubleData();
        for (keel::usize i = 0; i < winLen; ++i)
            win[i] = w[i];
    } else {
        // MATLAB pwelch default: divide x into 8 segments with 50%
        // overlap. Solving:
        //   nx = (winLen - noverlap)*8 + noverlap = 8*winLen - 7*noverlap
        // with noverlap = floor(winLen/2) gives winLen = floor(nx / 4.5).
        winLen = std::max<keel::usize>(1,
            keel::as<keel::usize>(std::floor(keel::as<keel::f64>(nx) / 4.5)));
        if (winLen > nx) winLen = nx;
        win.resize(winLen);
        fillHammingWindow(win.data(), winLen);
    }

    if (noverlap == 0)
        noverlap = winLen / 2;
    if (nfft == 0)
        nfft = std::max<keel::usize>(256, nextPow2(winLen));

    keel::f64 winPower = 0.0;
    for (keel::usize i = 0; i < winLen; ++i)
        winPower += win[i] * win[i];

    const keel::usize nOut = nfft / 2 + 1;
    auto psd = keel::Array<keel::f64>(nOut, scratch.allocator());
    keel::usize nSegments = 0;
    const keel::usize step = winLen - noverlap;

    // Per-segment FFT buffer hoisted out of the loop: under arena
    // semantics a fresh `vec<Complex>(nfft)` per iteration would bump-
    // allocate without reuse, growing the arena footprint to
    // O(nSegments × nfft). Reusing one buffer keeps it at O(nfft).
    auto buf = keel::Array<keel::c64>(nfft, scratch.allocator());

    const keel::usize nOutReal = isCplx ? nfft : nOut;
    if (isCplx) psd.resize(nOutReal);
    for (keel::usize start = 0; start + winLen <= nx; start += step) {
        if (isCplx) {
            for (keel::usize i = 0; i < winLen; ++i)
                buf[i] = xcd[start + i] * win[i];
        } else {
            for (keel::usize i = 0; i < winLen; ++i)
                buf[i] = keel::c64(xd[start + i] * win[i], 0.0);
        }
        for (keel::usize i = winLen; i < nfft; ++i)
            buf[i] = keel::c64(0.0, 0.0);

        fftRadix2(scratch.allocator(), buf, 1);

        for (keel::usize i = 0; i < nOutReal; ++i) {
            keel::f64 mag2 = keel::norm(buf[i]);
            if (!isCplx && i > 0 && i < nfft / 2)
                mag2 *= 2.0;
            psd[i] += mag2;
        }
        nSegments++;
    }

    // MATLAB Welch PSD scaling: Pxx[k] = mean_segments(|X|^2) /
    // (winPower * fs). With default fs = 2*pi this matches MATLAB's
    // pwelch(x) without explicit fs.
    const keel::f64 scale = 1.0 / (winPower * fs * nSegments);
    auto Pxx = Value::matrix(nOutReal, 1, ValueType::DOUBLE, mr);
    auto F = Value::matrix(nOutReal, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nOutReal; ++i) {
        Pxx.doubleDataMut()[i] = psd[i] * scale;
        F.doubleDataMut()[i] = isCplx
            ? (fs * keel::as<keel::f64>(i) / keel::as<keel::f64>(nfft))
            : ((fs / 2.0) * keel::as<keel::f64>(i) / keel::as<keel::f64>(nOutReal - 1));
    }

    return std::make_tuple(std::move(Pxx), std::move(F));
}

// ════════════════════════════════════════════════════════════════════
// cpsd / mscohere — cross spectra via Welch's method
// ════════════════════════════════════════════════════════════════════
//
// Common machinery: walk overlapping windowed segments of x and y,
// FFT each, accumulate three spectra:
//   Sxx[k] = Σ |X[k]|²
//   Syy[k] = Σ |Y[k]|²
//   Sxy[k] = Σ X[k]·conj(Y[k])
// `cpsd` returns Sxy normalised the same way pwelch normalises Pxx.
// `mscohere` returns |Sxy|² / (Sxx · Syy) — independent of any
// per-segment scaling, so the normalisation cancels out and the
// result is a real number in [0, 1].

namespace {

struct CrossWelchOut {
    keel::Array<keel::c64> Sxy;
    keel::Array<keel::f64> Sxx, Syy;
    keel::Array<keel::f64> F;
    keel::f64 winPower = 0.0;
    keel::usize nSegments = 0;
    keel::usize nfft = 0;
};

void crossWelch_argcheck(const Value &x, const Value &y)
{
    if (x.numel() != y.numel())
        throw Error("cpsd/mscohere: x and y must have the same length",
                    0, 0, "cpsd", "", "numkit:cpsd:size");
}

CrossWelchOut crossWelch(const Value &x, const Value &y, const Value &window, keel::usize noverlap, keel::usize nfft, keel::Allocator *mr)
{
    const keel::usize nx = x.numel();
    crossWelch_argcheck(x, y);
    const keel::usize ny = y.numel();
    if (nx != ny)
        throw Error("cpsd/mscohere: x and y must have the same length",
                    0, 0, "cpsd", "", "numkit:cpsd:size");
    const keel::f64 *xd = x.doubleData();
    const keel::f64 *yd = y.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::usize winLen;
    keel::Array<keel::f64> win(scratch.allocator());
    if (window.numel() > 0) {
        winLen = window.numel();
        win.resize(winLen);
        const keel::f64 *w = window.doubleData();
        for (keel::usize i = 0; i < winLen; ++i) win[i] = w[i];
    } else {
        // MATLAB cpsd / mscohere / tfestimate default: 8 Hamming-windowed
        // segments with 50% overlap (winLen = floor(nx / 4.5)).
        winLen = std::max<keel::usize>(1,
            keel::as<keel::usize>(std::floor(keel::as<keel::f64>(nx) / 4.5)));
        if (winLen > nx) winLen = nx;
        win.resize(winLen);
        fillHammingWindow(win.data(), winLen);
    }
    if (noverlap == 0) noverlap = winLen / 2;
    if (nfft == 0)     nfft = std::max<keel::usize>(256, nextPow2(winLen));

    keel::f64 winPower = 0.0;
    for (keel::usize i = 0; i < winLen; ++i) winPower += win[i] * win[i];

    const keel::usize nOut = nfft / 2 + 1;
    auto bufX = keel::Array<keel::c64>(nfft, scratch.allocator());
    auto bufY = keel::Array<keel::c64>(nfft, scratch.allocator());

    CrossWelchOut o;
    o.Sxx.assign(nOut, 0.0);
    o.Syy.assign(nOut, 0.0);
    o.Sxy.assign(nOut, keel::c64(0.0, 0.0));

    const keel::usize step = winLen - noverlap;
    for (keel::usize start = 0; start + winLen <= nx; start += step) {
        for (keel::usize i = 0; i < winLen; ++i) {
            bufX[i] = keel::c64(xd[start + i] * win[i], 0.0);
            bufY[i] = keel::c64(yd[start + i] * win[i], 0.0);
        }
        for (keel::usize i = winLen; i < nfft; ++i) {
            bufX[i] = keel::c64(0.0, 0.0);
            bufY[i] = keel::c64(0.0, 0.0);
        }
        fftRadix2(scratch.allocator(), bufX, 1);
        fftRadix2(scratch.allocator(), bufY, 1);
        for (keel::usize i = 0; i < nOut; ++i) {
            const keel::f64 sx = keel::norm(bufX[i]);
            const keel::f64 sy = keel::norm(bufY[i]);
            const keel::c64 sxy = bufX[i] * keel::conj(bufY[i]);
            o.Sxx[i] += sx;
            o.Syy[i] += sy;
            o.Sxy[i] += sxy;
        }
        ++o.nSegments;
    }
    o.winPower = winPower;
    o.nfft = nfft;
    o.F.assign(nOut, 0.0);
    for (keel::usize i = 0; i < nOut; ++i)
        o.F[i] = M_PI * keel::as<keel::f64>(i) / keel::as<keel::f64>(nOut - 1);
    return o;
}

} // anonymous

std::tuple<Value, Value>
cpsd(const Value &x, const Value &y, const Value &window, keel::usize noverlap, keel::usize nfft, keel::f64 fs, keel::Allocator *mr)
{
    auto o = crossWelch(x, y, window, noverlap, nfft, mr);
    const keel::usize nOut = o.Sxx.size();
    // Rescale frequency vector to [0, fs/2] (crossWelch uses [0, pi]).
    auto rescaleF = [&]() {
        for (keel::usize i = 0; i < nOut; ++i)
            o.F[i] = (fs / 2.0) * keel::as<keel::f64>(i)
                     / keel::as<keel::f64>(nOut - 1);
    };
    rescaleF();
    if (o.nSegments == 0) {
        Value Pxy0 = Value::matrix(nOut, 1, ValueType::COMPLEX, mr);
        Value F0   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
        if (nOut > 0) std::copy(o.F.begin(), o.F.end(), F0.doubleDataMut());
        return std::make_tuple(std::move(Pxy0), std::move(F0));
    }
    // MATLAB cpsd scaling: Pxy[k] = mean_segments(X*conj(Y)) /
    // (winPower * fs). Default fs = 2*pi mirrors MATLAB.
    const keel::f64 scale = 1.0 / (o.winPower * fs *
                                 keel::as<keel::f64>(o.nSegments));
    Value Pxy = Value::matrix(nOut, 1, ValueType::COMPLEX, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    auto *cd = Pxy.complexDataMut();
    auto *fd = F.doubleDataMut();
    for (keel::usize i = 0; i < nOut; ++i) {
        keel::c64 s = o.Sxy[i] * scale;
        if (i > 0 && i < o.nfft / 2) s *= 2.0;
        cd[i] = s;
        fd[i] = o.F[i];
    }
    return std::make_tuple(std::move(Pxy), std::move(F));
}

std::tuple<Value, Value>
mscohere(const Value &x, const Value &y, const Value &window, keel::usize noverlap, keel::usize nfft, keel::f64 fs, keel::Allocator *mr)
{
    auto o = crossWelch(x, y, window, noverlap, nfft, mr);
    const keel::usize nOut = o.Sxx.size();
    Value Cxy = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    auto *cd = Cxy.doubleDataMut();
    auto *fd = F.doubleDataMut();
    for (keel::usize i = 0; i < nOut; ++i) {
        const keel::f64 denom = o.Sxx[i] * o.Syy[i];
        const keel::f64 num   = keel::norm(o.Sxy[i]);
        cd[i] = (denom > 0.0) ? num / denom : 0.0;
        fd[i] = (fs / 2.0) * keel::as<keel::f64>(i)
                / keel::as<keel::f64>(nOut - 1);
    }
    return std::make_tuple(std::move(Cxy), std::move(F));
}

std::tuple<Value, Value>
tfestimate(const Value &x, const Value &y, const Value &window, keel::usize noverlap, keel::usize nfft, keel::f64 fs, keel::Allocator *mr)
{
    auto o = crossWelch(x, y, window, noverlap, nfft, mr);
    const keel::usize nOut = o.Sxx.size();
    Value Txy = Value::matrix(nOut, 1, ValueType::COMPLEX, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    auto *td = Txy.complexDataMut();
    auto *fd = F.doubleDataMut();
    // MATLAB convention: Txy = Pyx / Pxx where Pyx = E[Y · conj(X)]
    // = conj(Pxy). The per-segment scaling factor cancels in the
    // ratio, so the raw accumulators are sufficient.
    for (keel::usize i = 0; i < nOut; ++i) {
        const keel::c64 Syx = keel::conj(o.Sxy[i]);
        td[i] = (o.Sxx[i] > 0.0)
                ? Syx / o.Sxx[i]
                : keel::c64(0.0, 0.0);
        fd[i] = (fs / 2.0) * keel::as<keel::f64>(i)
                / keel::as<keel::f64>(nOut - 1);
    }
    return std::make_tuple(std::move(Txy), std::move(F));
}

} // namespace numkit::signal
