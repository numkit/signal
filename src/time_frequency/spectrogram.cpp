// toolboxes/signal/src/time_frequency/spectrogram.cpp
//
// Short-time Fourier transform. Split from spectral_analysis/. The
// time-collapsed power-spectrum estimators periodogram / pwelch
// live in spectral_analysis/periodogram_pwelch.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/time_frequency/spectrogram.hpp>
#include <numkit/signal/transforms/fft.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "../dsp_helpers.hpp"
#include "../signal_detail.hpp"     // shared argcheck cores

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace numkit::signal {

namespace {

// Fill caller-provided buffer with Hamming window coefficients, same
// formula MATLAB uses.
void iscola_argcheck(const Value &window, keel::usize noverlap, keel::StringRef method)
{
    const keel::usize M = window.numel();
    if (M == 0)
        throw Error("iscola: window must be non-empty",
                     0, 0, "iscola", "", "numkit:iscola:badWindow");
    if (noverlap >= M)
        throw Error("iscola: noverlap must be < window length",
                     0, 0, "iscola", "", "numkit:iscola:badOverlap");
    if (method != "ola" && method != "wola")
        throw Error("iscola: method must be 'ola' or 'wola'",
                     0, 0, "iscola", "", "numkit:iscola:badMethod");
}

void fillHammingWindow(double *w, keel::usize N)
{
    if (N == 1) {
        w[0] = 1.0;
        return;
    }
    for (keel::usize i = 0; i < N; ++i)
        w[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (N - 1));
}

// Periodic Hann window (MATLAB hann(N, 'periodic') — used as the stft
// default. Note the divisor is N, NOT N-1 (that's the symmetric form).
void fillHannPeriodic(double *w, keel::usize N)
{
    if (N == 1) { w[0] = 1.0; return; }
    for (keel::usize i = 0; i < N; ++i)
        w[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / N));
}

} // anonymous namespace

std::tuple<Value, Value, Value>
spectrogram(const Value &x, const Value &window, keel::usize noverlap, keel::usize nfft, keel::Allocator *mr)
{
    Value xD = detail::widenSingleToDouble(x, mr);
    const keel::usize nx = xD.numel();
    const double *xd = xD.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    keel::usize winLen;
    keel::Array<double> win(scratch.allocator());
    if (window.numel() > 0) {
        winLen = window.numel();
        win.resize(winLen);
        for (keel::usize i = 0; i < winLen; ++i)
            win[i] = window.elemAsDouble(i);
    } else {
        // MATLAB spectrogram default: 8 Hamming-windowed segments with
        // 50% overlap (winLen = floor(nx / 4.5)).
        winLen = std::max<keel::usize>(1,
            keel::as<keel::usize>(std::floor(keel::as<double>(nx) / 4.5)));
        if (winLen > nx) winLen = nx;
        win.resize(winLen);
        fillHammingWindow(win.data(), winLen);
    }

    if (winLen > nx)
        winLen = nx;

    if (noverlap == 0)
        noverlap = winLen / 2;
    if (nfft == 0)
        nfft = std::max<keel::usize>(256, nextPow2(winLen));

    const keel::usize nFreqs = nfft / 2 + 1;
    const keel::usize step = winLen - noverlap;

    keel::usize nSegments = 0;
    for (keel::usize start = 0; start + winLen <= nx; start += step)
        nSegments++;

    auto S = Value::complexMatrix(nFreqs, nSegments, mr);
    auto F = Value::matrix(nFreqs, 1, ValueType::DOUBLE, mr);
    auto T = Value::matrix(1, nSegments, ValueType::DOUBLE, mr);

    // Per-segment FFT buffer hoisted: see pwelch for the rationale —
    // a fresh allocation per loop iteration would grow the arena to
    // O(nSegments × nfft) instead of O(nfft).
    auto buf = keel::Array<keel::c64>(nfft, scratch.allocator());

    keel::usize seg = 0;
    for (keel::usize start = 0; start + winLen <= nx; start += step) {
        for (keel::usize i = 0; i < winLen; ++i)
            buf[i] = keel::c64(xd[start + i] * win[i], 0.0);
        for (keel::usize i = winLen; i < nfft; ++i)
            buf[i] = keel::c64(0.0, 0.0);

        fftRadix2(scratch.allocator(), buf, -1);

        for (keel::usize i = 0; i < nFreqs; ++i)
            S.complexDataMut()[i + seg * nFreqs] = buf[i];

        T.doubleDataMut()[seg] = keel::as<double>(start + winLen / 2);
        seg++;
    }

    for (keel::usize i = 0; i < nFreqs; ++i)
        F.doubleDataMut()[i] = M_PI * i / (nFreqs - 1);

    return std::make_tuple(std::move(S), std::move(F), std::move(T));
}

// ── stft / istft (MATLAB-compat) ─────────────────────────────────────
//
// MATLAB stft defaults: hann(128, 'periodic'), overlap = 96,
// fftLength = 128, range = 'twosided'. Output is FFT-bin × frame.
// MATLAB packs the columns in standard fft order; 'centered' applies
// an fftshift along the frequency axis, 'onesided' keeps bins 0..N/2.
//
// istft inverts via overlap-add with the same window — for the
// canonical COLA configurations (hann/periodic + 50%/75% overlap) the
// round-trip is bit-exact within ulp.
namespace {

void resolveWindow(const Value &win, keel::Array<double> &out, keel::Allocator *mr)
{
    if (win.numel() > 0) {
        const keel::usize n = win.numel();
        out.resize(n);
        for (keel::usize i = 0; i < n; ++i) out[i] = win.elemAsDouble(i);
        return;
    }
    // Default: hann(128, 'periodic').
    out.resize(128);
    fillHannPeriodic(out.data(), 128);
    (void) mr;
}

} // anonymous namespace

Value stft(const Value &x, const Value &window, keel::usize overlap, keel::usize fftLength, keel::StringRef range, keel::Allocator *mr)
{
// Contract (probed R2025b): x accepts double/single/complex; logical/integer
    // and 3-D+ arrays are rejected. Multi-channel 2-D matrices (N x C) frame
    // each column independently into a 3-D [outRows x K x C] complex array.
    detail::requireFloatInput(x, "stft");
    if (x.dims().ndim() > 2)
        throw Error("stft: X must be two-dimensional",
                     0, 0, "stft", "", "numkit:stft:expected2D");
    const bool isMatrix = (x.dims().rows() > 1 && x.dims().cols() > 1);
    const keel::usize N   = isMatrix ? x.dims().rows() : x.numel();
    const keel::usize nCh = isMatrix ? x.dims().cols() : 1;
    if (N == 0 || x.numel() == 0)
        return Value::complexMatrix(0, 0, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> win(scratch.allocator());
    resolveWindow(window, win, mr);
    const keel::usize M = win.size();
    if (M == 0)
        throw Error("stft: window must be non-empty",
                     0, 0, "stft", "", "numkit:stft:badWindow");
    if (M > N)
        throw Error("stft: signal shorter than window length",
                     0, 0, "stft", "", "numkit:stft:shortSignal");

    const keel::usize OL  = (overlap == SIZE_MAX) ? (3 * M) / 4 : overlap;
    if (OL >= M)
        throw Error("stft: OverlapLength must be < window length",
                     0, 0, "stft", "", "numkit:stft:badOverlap");
    const keel::usize NFFT = (fftLength == 0) ? M : fftLength;
    if (NFFT < M)
        throw Error("stft: FFTLength must be >= window length",
                     0, 0, "stft", "", "numkit:stft:badNfft");

    if (range != "twosided" && range != "onesided" && range != "centered")
        throw Error("stft: FrequencyRange must be 'twosided', 'centered' "
                    "or 'onesided'",
                     0, 0, "stft", "", "numkit:stft:badRange");

    const keel::usize hop = M - OL;
    const keel::usize K   = (N - M) / hop + 1;
    const bool   isOneSided = (range == "onesided");
    const bool   isCentered = (range == "centered");
    const keel::usize outRows = isOneSided ? (NFFT / 2 + 1) : NFFT;
    // Centered rotation: MATLAB places bins as [-N/2+1, ..., N/2] for
    // even N (Nyquist at the END), or [-(N-1)/2, ..., (N-1)/2] for odd
    // N. So DC sits at index N/2-1 (even) or (N-1)/2 (odd). To produce
    //   Sd[k] = Fd[(k + cShift) mod N]
    // with Sd[DC_idx] = Fd[0], we need cShift = N - DC_idx:
    //   even N: cShift = N/2 + 1
    //   odd  N: cShift = (N + 1) / 2
    const keel::usize cShift = (NFFT % 2 == 0) ? (NFFT / 2 + 1)
                                          : ((NFFT + 1) / 2);

    // x as doubles. Complex input also supported via fft(complex frame).
    const bool xCplx = x.isComplex();
    Value xD = xCplx ? Value() : detail::widenSingleToDouble(x, mr);
    const double *xr = xCplx ? nullptr : xD.doubleData();
    const keel::c64     *xc = xCplx ? x.complexData() : nullptr;

    Value S = (nCh > 1)
        ? Value::matrix3d(outRows, K, nCh, ValueType::COMPLEX, mr)
        : Value::complexMatrix(outRows, K, mr);
    keel::c64 *Sd = S.complexDataMut();

    // Per-frame buffer reused across iterations.
    Value frameV = xCplx
        ? Value::complexMatrix(NFFT, 1, mr)
        : Value::matrix(NFFT, 1, ValueType::DOUBLE, mr);
    keel::c64     *frameCd = xCplx ? frameV.complexDataMut() : nullptr;
    double *frameRd = xCplx ? nullptr               : frameV.doubleDataMut();

    for (keel::usize ch = 0; ch < nCh; ++ch) {
        const keel::usize chOff = ch * N;
        keel::c64 *SdCh = Sd + ch * (outRows * K);
        for (keel::usize k = 0; k < K; ++k) {
            const keel::usize start = chOff + k * hop;
            if (xCplx) {
                for (keel::usize i = 0; i < M; ++i)
                    frameCd[i] = xc[start + i] * win[i];
                for (keel::usize i = M; i < NFFT; ++i) frameCd[i] = keel::c64(0, 0);
            } else {
                for (keel::usize i = 0; i < M; ++i)
                    frameRd[i] = xr[start + i] * win[i];
                for (keel::usize i = M; i < NFFT; ++i) frameRd[i] = 0.0;
            }
            Value F = fft(frameV, keel::as<int>(NFFT), 1, mr);
            const keel::c64 *Fd = F.complexData();
            if (isCentered) {
                for (keel::usize r = 0; r < outRows; ++r)
                    SdCh[k * outRows + r] = Fd[(r + cShift) % NFFT];
            } else {
                // twosided / onesided: copy the first outRows bins as-is.
                for (keel::usize r = 0; r < outRows; ++r)
                    SdCh[k * outRows + r] = Fd[r];
            }
        }
    }
    return S;
}

Value istft(const Value &S, const Value &window, keel::usize overlap, keel::usize fftLength, keel::StringRef range, keel::Allocator *mr)
{
const auto &d = S.dims();
    const keel::usize inRows = d.rows();
    const keel::usize K      = d.cols();
    if (K == 0 || inRows == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> win(scratch.allocator());
    resolveWindow(window, win, mr);
    const keel::usize M = win.size();
    if (M == 0)
        throw Error("istft: window must be non-empty",
                     0, 0, "istft", "", "numkit:istft:badWindow");

    const keel::usize OL  = (overlap == SIZE_MAX) ? (3 * M) / 4 : overlap;
    if (OL >= M)
        throw Error("istft: OverlapLength must be < window length",
                     0, 0, "istft", "", "numkit:istft:badOverlap");

    if (range != "twosided" && range != "onesided" && range != "centered")
        throw Error("istft: FrequencyRange must be 'twosided', 'centered' "
                    "or 'onesided'",
                     0, 0, "istft", "", "numkit:istft:badRange");

    const bool   isOneSided = (range == "onesided");
    const bool   isCentered = (range == "centered");
    keel::usize NFFT = (fftLength == 0)
                    ? (isOneSided ? 2 * (inRows - 1) : inRows)
                    : fftLength;
    // Inverse of the centered rotation (see stft above): the input
    // column came from twosided via Sd[k] = Fd[(k + cShift) mod N], so
    // we recover Fd[j] = Sd[(j - cShift + N) mod N]. Equivalent forward
    // shift = N - cShift.
    const keel::usize invCShift = NFFT - ((NFFT % 2 == 0) ? (NFFT / 2 + 1)
                                                     : ((NFFT + 1) / 2));
    if (!isOneSided && inRows != NFFT)
        throw Error("istft: STFT row count does not match FFTLength",
                     0, 0, "istft", "", "numkit:istft:badShape");
    if (isOneSided && inRows != NFFT / 2 + 1)
        throw Error("istft: one-sided STFT row count must equal "
                    "FFTLength/2 + 1",
                     0, 0, "istft", "", "numkit:istft:badShape");
    if (NFFT < M)
        throw Error("istft: FFTLength must be >= window length",
                     0, 0, "istft", "", "numkit:istft:badNfft");

    const keel::usize hop = M - OL;
    const keel::usize Nout = (K - 1) * hop + M;

    Value out = Value::matrix(Nout, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    std::fill(od, od + Nout, 0.0);

    // Per-sample window-square accumulator for normalization.
    keel::Array<double> wnorm(Nout, 0.0, scratch.allocator());

    // Per-frame full spectrum (after undoing centered / onesided
    // packing) and ifft-result.
    Value Fcol = Value::complexMatrix(NFFT, 1, mr);
    keel::c64 *Fd = Fcol.complexDataMut();
    const keel::c64 *Sd = S.complexData();

    for (keel::usize k = 0; k < K; ++k) {
        // Repopulate Fcol from S column k according to range.
        if (isOneSided) {
            // Hermitian reflection: F[N-i] = conj(F[i]) for i=1..N/2-1.
            for (keel::usize r = 0; r < inRows; ++r) Fd[r] = Sd[k * inRows + r];
            for (keel::usize r = 1; r < NFFT - inRows + 1; ++r)
                Fd[NFFT - r] = keel::conj(Fd[r]);
        } else if (isCentered) {
            // Undo the centered rotation.
            for (keel::usize r = 0; r < NFFT; ++r)
                Fd[r] = Sd[k * inRows + ((r + invCShift) % NFFT)];
        } else {
            for (keel::usize r = 0; r < NFFT; ++r) Fd[r] = Sd[k * inRows + r];
        }

        Value t = ifft(Fcol, keel::as<int>(NFFT), 1, mr);
        // ifft may return DOUBLE or COMPLEX.
        const keel::c64     *tc = (t.type() == ValueType::COMPLEX) ? t.complexData() : nullptr;
        const double *tr = (t.type() == ValueType::COMPLEX) ? nullptr        : t.doubleData();

        const keel::usize start = k * hop;
        for (keel::usize i = 0; i < M; ++i) {
            const double real_i = tc ? tc[i].real() : tr[i];
            od[start + i]    += win[i] * real_i;
            wnorm[start + i] += win[i] * win[i];
        }
    }

    // Normalize sample-wise so COLA-compliant windows reach identity
    // reconstruction. The wnorm[n] = 0 case can only happen outside the
    // covered support (impossible by construction); guard anyway.
    for (keel::usize i = 0; i < Nout; ++i)
        if (wnorm[i] > 0.0) od[i] /= wnorm[i];

    return out;
}

// ── iscola — Constant OverLap-Add compliance check ─────────────────
//
// MATLAB sig: [tf, m, maxDev] = iscola(window, noverlap[, method]).
// Default method is 'wola' (sum of squared window); 'ola' uses the
// unsquared window. The algorithm builds K = ceil(M/hop) + 2 frames
// stacked at multiples of hop = M - noverlap, sums them up sample-by-
// sample, then inspects the stable overlap region (where every interior
// sample is covered by `nOverlapWindows = ceil(M/hop)` frames). The
// region length is at least `hop` so the periodicity-1 sum can be
// fully characterised by its median and its maximum deviation. Match
// boundary: `maxDev < 2 * eps` (probed against MATLAB R2025b: dev = 1·eps
// passes, dev ≥ 2·eps fails).

std::tuple<Value, Value, Value>
iscola(const Value &window, keel::usize noverlap, keel::StringRef method,
       keel::Allocator *mr)
{
    const keel::usize M = window.numel();
    if (M == 0)
        throw Error("iscola: window must be non-empty",
                     0, 0, "iscola", "", "numkit:iscola:badWindow");
    if (noverlap >= M)
        throw Error("iscola: noverlap must be < window length",
                     0, 0, "iscola", "", "numkit:iscola:badOverlap");
    iscola_argcheck(window, noverlap, method);
    if (method != "ola" && method != "wola")
        throw Error("iscola: method must be 'ola' or 'wola'",
                     0, 0, "iscola", "", "numkit:iscola:badMethod");

    const keel::usize hop  = M - noverlap;
    const keel::usize nOvw = (M + hop - 1) / hop;        // ceil(M/hop)
    const keel::usize K    = nOvw + 2;
    const keel::usize Ltot = (K - 1) * hop + M;
    const bool isWola      = (method == "wola");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> s(Ltot, 0.0, scratch.allocator());
    for (keel::usize k = 0; k < K; ++k)
        for (keel::usize i = 0; i < M; ++i) {
            const double wi = window.elemAsDouble(i);
            s[k * hop + i] += isWola ? wi * wi : wi;
        }

    // Stable region: [(nOvw - 1)*hop, K*hop). Take the first `hop`-long
    // window of it — by periodicity within the stable region every
    // hop-sized slice has the same content (up to FP).
    const keel::usize stableStart = (nOvw - 1) * hop;
    keel::Array<double> stable(s.slice(stableStart, hop), scratch.allocator());

    // Median.
    keel::Array<double> sorted = stable.clone(scratch.allocator());
    std::sort(sorted.begin(), sorted.end());
    const keel::usize H = sorted.size();
    const double median = (H % 2 == 0)
        ? 0.5 * (sorted[H / 2 - 1] + sorted[H / 2])
        : sorted[H / 2];

    // Max absolute deviation from the median.
    double maxDev = 0.0;
    for (double v : stable)
        maxDev = std::max(maxDev, keel::abs(v - median));

    // MATLAB tolerance is relative to the median (probed): dev/|m| ≤ eps
    // passes, dev/|m| > eps fails. e.g. hann@48 'ola' has dev = 2·eps and
    // m = 2 → ratio = eps (exact boundary), tf = 1.
    const double eps = std::numeric_limits<double>::epsilon();
    const double tol = (keel::abs(median) > 0.0) ? keel::abs(median) * eps : eps;
    const bool   tf  = (maxDev <= tol);

    Value out_tf  = Value::scalar(tf ? 1.0 : 0.0, mr);
    Value out_m   = Value::scalar(median, mr);
    Value out_dev = Value::scalar(maxDev, mr);
    return { std::move(out_tf), std::move(out_m), std::move(out_dev) };
}

} // namespace numkit::signal
