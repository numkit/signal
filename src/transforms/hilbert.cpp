// toolboxes/signal/src/transforms/hilbert.cpp
//
// FFT-based Hilbert transform + envelope. unwrap moved to
// filter_analysis/unwrap.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/transforms/hilbert.hpp>

#include <numkit/signal/transforms/fft.hpp>  // general fft/ifft for non-pow2 N
#include <numkit/signal/internal/shims/polyfun.hpp>     // interp1 (spline)
#include <numkit/signal/windows/windows.hpp>          // kaiser

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "../dsp_helpers.hpp"  // Complex, FFT helpers
#include <numkit/ops/helpers.hpp>         // createLike

#include <algorithm>
#include <cmath>

namespace numkit::signal {

namespace {

// Shared FFT-based Hilbert transform kernel used by both hilbert() and
// envelope(). Returns a buffer of length fftLen holding the analytic
// signal (possibly zero-padded beyond N). Caller slices to the first N
// samples. Backed by the caller's scratch arena.
keel::Array<keel::c64> hilbertBuf(const Value &x, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    keel::Array<keel::c64> buf(N, mr);
    if (N == 0) return buf;

    // The analytic signal must be computed at the SIGNAL length N — NOT a
    // zero-padded power-of-two (that corrupts the spectrum and breaks the
    // constant-envelope property for non-pow2 N: bugs/signal/hilbert-nonpow2).
    const bool pow2 = (N & (N - 1)) == 0;
    if (pow2) {
        auto tmp = prepareFFTBuffer(mr, x, N, N);   // fftLen = N, no padding
        fftRadix2(mr, tmp, -1);                       // forward (exp(-2πi·k/N))
        std::copy(tmp.begin(), tmp.end(), buf.begin());
    } else {
        // Non-power-of-two → general (mixed-radix / Bluestein) length-N FFT.
        Value xcol = Value::matrix(N, 1, ValueType::DOUBLE, mr);
        keel::f64 *xd = xcol.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) xd[i] = x.elemAsDouble(i);
        Value X = fft(xcol, keel::as<keel::i32>(N), /*dim=*/0, mr);
        const keel::c64 *Xc = X.complexData();
        std::copy(Xc, Xc + N, buf.begin());
    }

    // Analytic mask at length N: double the positive frequencies, keep DC
    // (and the Nyquist bin when N is even), zero the negative frequencies.
    for (keel::usize i = 1; i < (N + 1) / 2; ++i) buf[i] *= 2.0;
    for (keel::usize i = N / 2 + 1; i < N; ++i) buf[i] = keel::c64(0.0, 0.0);

    if (pow2) {
        // IFFT via conjugate trick: ifft(X) = conj(fft(conj(X)))/N.
        for (auto &v : buf) v = keel::conj(v);
        fftRadix2(mr, buf, -1);
        const keel::f64 invN = 1.0 / keel::as<keel::f64>(N);
        for (auto &v : buf) v = keel::conj(v) * invN;
    } else {
        Value Xm = Value::matrix(N, 1, ValueType::COMPLEX, mr);
        std::copy(buf.begin(), buf.end(), Xm.complexDataMut());
        Value zt = ifft(Xm, keel::as<keel::i32>(N), /*dim=*/0, mr);
        const keel::c64 *zc = zt.complexData();
        std::copy(zc, zc + N, buf.begin());
    }
    return buf;
}

} // anonymous namespace

namespace {

void hilbert_argcheck(const Value &x)
{
    (void)x;
}

} // namespace

Value hilbert(const Value &x, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    hilbert_argcheck(x);
    auto buf = hilbertBuf(x, scratch.allocator());
    return packComplexResult(buf.data(), N, mr);
}

// ── Envelope modes (matches MATLAB R2025b envelope.m) ────────────────
namespace {

// Read x into a flat scratch buffer.
keel::Array<keel::f64> read_real(const Value &x, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    keel::Array<keel::f64> v(N, mr);
    for (keel::usize i = 0; i < N; ++i) v[i] = x.elemAsDouble(i);
    return v;
}

// |hilbert(x)| amplitude — used by the default and 'analytic' (no n) paths.
keel::Array<keel::f64> ampl_fft_hilbert(const Value &x, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    auto buf = hilbertBuf(x, mr);
    keel::Array<keel::f64> a(N, mr);
    for (keel::usize i = 0; i < N; ++i) a[i] = keel::abs(buf[i]);
    return a;
}

// envFIR — windowed (n-tap) ideal-Hilbert FIR filter, Kaiser-tapered,
// applied via 'same' mode complex convolution. MATLAB R2025b algorithm:
//   t = ((1-n)/2 : (n-1)/2) / 2
//   hfilt = sinc(t) .* exp(i·π·t)
//   firFilter = hfilt .* kaiser(n, 8)
//   firFilter /= sum(real(firFilter))
//   y = abs(conv2(x, firFilter, 'same'))
keel::Array<keel::f64> ampl_fir(const Value &x_centered, keel::usize n, keel::Allocator *mr)
{
    const keel::usize N = x_centered.numel();
    auto x = read_real(x_centered, mr);

    // Kaiser window via numkit::signal::kaiser (returns Value 1×n).
    Value kw = numkit::signal::kaiser(n, 8.0, mr);
    const keel::f64 *kd = kw.doubleData();

    // hfilt[k] = sinc(t_k) · exp(iπ·t_k), windowed.
    keel::Array<keel::c64> filt(n, mr);
    keel::c64 normR(0.0, 0.0);
    for (keel::usize k = 0; k < n; ++k) {
        const keel::f64 t = 0.5 * (keel::f64(k) - 0.5 * (keel::f64(n) - 1.0));
        const keel::f64 s = (std::fabs(t) < 1e-15)
                         ? 1.0
                         : std::sin(M_PI * t) / (M_PI * t);
        const keel::f64 w = kd[k];
        const keel::c64 h = keel::c64(s * w * std::cos(M_PI * t),
                                  s * w * std::sin(M_PI * t));
        filt[k] = h;
        normR += keel::c64(h.real(), 0.0);
    }
    const keel::f64 inv_norm = 1.0 / normR.real();
    for (auto &c : filt) c *= inv_norm;

    // 'same' complex conv of length-N real signal with length-n complex
    // filter. Output length = N. MATLAB's conv2 'same' centers using
    // floor(n/2) for the leading offset (verified vs R2025b on n=8).
    const keel::usize pad = n / 2;
    keel::Array<keel::f64> out(N, mr);
    for (keel::usize i = 0; i < N; ++i) {
        keel::c64 acc(0.0, 0.0);
        // y[i] = sum_k x[i+pad-k] · filt[k] (matching conv 'same').
        for (keel::usize k = 0; k < n; ++k) {
            const keel::i64 src = (keel::i64)i + (keel::i64)pad
                                  - (keel::i64)k;
            if (src >= 0 && (keel::usize)src < N)
                acc += filt[k] * x[(keel::usize)src];
        }
        out[i] = keel::abs(acc);
    }
    return out;
}

// envRMS — sliding-window RMS. y = sqrt(movmean(x², n)).
keel::Array<keel::f64> ampl_rms(const Value &x_centered, keel::usize n, keel::Allocator *mr)
{
    const keel::usize N = x_centered.numel();
    auto x = read_real(x_centered, mr);

    // movmean with 'centered' window of length n. MATLAB's movmean by
    // default uses a centered window: for odd n, [-(n-1)/2 .. (n-1)/2];
    // for even n, [-n/2 .. n/2-1] (asymmetric). Edge handling: shrink
    // window to available samples.
    const keel::i64 left  = (keel::i64)(n / 2);
    const keel::i64 right = (keel::i64)(n - 1) - left;  // for even n: left = n/2, right = n/2-1
    keel::Array<keel::f64> out(N, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 lo = std::max((keel::i64)0, (keel::i64)i - left);
        const keel::i64 hi = std::min((keel::i64)N - 1, (keel::i64)i + right);
        keel::f64 s = 0.0;
        keel::i64 cnt = 0;
        for (keel::i64 j = lo; j <= hi; ++j) { s += x[(keel::usize)j] * x[(keel::usize)j]; ++cnt; }
        out[i] = std::sqrt(s / keel::f64(cnt));
    }
    return out;
}

// findpeaks with MinPeakDistance ≥ n. Greedy: sort indices by value
// descending, accept peak if no already-accepted peak is within n samples.
// Returns indices (0-based) in ascending order.
keel::Array<keel::usize> findpeaks_mindist(const keel::Array<keel::f64> &x, keel::usize n, keel::Allocator *mr)
{
    const keel::usize N = x.size();
    keel::Array<keel::usize> peaks(mr);
    if (N < 3) return peaks;
    // Strict local maxima.
    keel::Array<keel::usize> cand(mr);
    for (keel::usize i = 1; i + 1 < N; ++i)
        if (x[i] > x[i - 1] && x[i] > x[i + 1]) cand.push_back(i);
    if (cand.empty()) return peaks;
    if (n <= 1) {
        peaks.assign(cand);
        return peaks;
    }
    // Sort by value descending.
    keel::Array<keel::usize> order = cand.clone(mr);
    std::sort(order.begin(), order.end(),
              [&](keel::usize a, keel::usize b) { return x[a] > x[b]; });
    keel::Array<keel::u8> accepted(N, 0, mr);
    for (keel::usize idx : order) {
        bool ok = true;
        const keel::i64 lo = (keel::i64)idx - (keel::i64)n;
        const keel::i64 hi = (keel::i64)idx + (keel::i64)n;
        for (keel::i64 j = std::max((keel::i64)0, lo);
             j <= std::min((keel::i64)N - 1, hi); ++j)
            if (accepted[(keel::usize)j]) { ok = false; break; }
        if (ok) accepted[idx] = 1;
    }
    for (keel::usize i = 0; i < N; ++i) if (accepted[i]) peaks.push_back(i);
    return peaks;
}

// envPeak — spline through local maxima/minima with MinPeakDistance n.
// MATLAB convention: if <2 peaks found, fallback to including endpoints.
// No DC removal in peak mode.
void env_peak(const Value &x, keel::usize n, keel::Array<keel::f64> &up, keel::Array<keel::f64> &lo, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    auto v = read_real(x, mr);
    keel::Array<keel::f64> neg(N, mr);
    for (keel::usize i = 0; i < N; ++i) neg[i] = -v[i];

    auto build_envelope = [&](const keel::Array<keel::f64> &sig,
                              keel::Array<keel::f64> &out) {
        keel::Array<keel::usize> peaks(mr);
        if (N > n + 1)
            peaks = findpeaks_mindist(sig, n, mr);
        // If <2 peaks, fall back to endpoints + any peaks.
        keel::Array<keel::usize> locs(mr);
        if (peaks.size() < 2) {
            keel::Array<keel::usize> draft(mr);
            draft.push_back(0);
            for (keel::usize p : peaks) draft.push_back(p);
            draft.push_back(N - 1);
            // De-dup if endpoints coincide with peaks.
            for (keel::usize l : draft) {
                if (locs.empty() || locs.back() != l) locs.push_back(l);
            }
        } else {
            locs.assign(peaks);
        }
        // MATLAB's `spline` fits a PARABOLA when given exactly 3
        // knots (not-a-knot is degenerate). nk's general spline
        // falls back to natural BCs which doesn't match — handle the
        // 3-point case inline.
        out.resize(N);
        if (locs.size() == 3) {
            const keel::f64 x1 = keel::f64(locs[0] + 1);
            const keel::f64 x2 = keel::f64(locs[1] + 1);
            const keel::f64 x3 = keel::f64(locs[2] + 1);
            const keel::f64 y1 = sig[locs[0]];
            const keel::f64 y2 = sig[locs[1]];
            const keel::f64 y3 = sig[locs[2]];
            // Solve y = a·x² + b·x + c via Lagrange-style determinants.
            const keel::f64 det = (x1 - x2) * (x1 - x3) * (x2 - x3);
            const keel::f64 a = (y1 * (x2 - x3) - y2 * (x1 - x3) + y3 * (x1 - x2))
                             / det;
            const keel::f64 b = (y1 * (x3 * x3 - x2 * x2)
                              - y2 * (x3 * x3 - x1 * x1)
                              + y3 * (x2 * x2 - x1 * x1)) / det;
            const keel::f64 c = (y1 * (x2 * x2 * x3 - x2 * x3 * x3)
                              - y2 * (x1 * x1 * x3 - x1 * x3 * x3)
                              + y3 * (x1 * x1 * x2 - x1 * x2 * x2)) / det;
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 xi = keel::f64(i + 1);
                out[i] = a * xi * xi + b * xi + c;
            }
            return;
        }
        // interp1 with 'spline' for 2 (linear) or 4+ (not-a-knot).
        Value x_loc = Value::matrix(locs.size(), 1, ValueType::DOUBLE, mr);
        Value y_loc = Value::matrix(locs.size(), 1, ValueType::DOUBLE, mr);
        Value xq    = Value::matrix(N, 1, ValueType::DOUBLE, mr);
        keel::f64 *xd = x_loc.doubleDataMut();
        keel::f64 *yd = y_loc.doubleDataMut();
        keel::f64 *qd = xq.doubleDataMut();
        for (keel::usize i = 0; i < locs.size(); ++i) {
            xd[i] = keel::f64(locs[i] + 1);
            yd[i] = sig[locs[i]];
        }
        for (keel::usize i = 0; i < N; ++i) qd[i] = keel::f64(i + 1);
        Value yi = numkit::builtin::interp1(x_loc, y_loc, xq, "spline", mr);
        for (keel::usize i = 0; i < N; ++i) out[i] = yi.elemAsDouble(i);
    };
    build_envelope(v, up);
    build_envelope(neg, lo);
    for (auto &v_ : lo) v_ = -v_;
}

} // anonymous

Value envelope(const Value &x, keel::Allocator *mr)
{
    // Default: mean + |hilbert(x - mean)|.
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = read_real(x, scratch.allocator());
    keel::f64 xmean = 0.0;
    for (keel::f64 d : v) xmean += d;
    xmean /= keel::f64(N);
    Value xc = Value::matrix(N, 1, ValueType::DOUBLE, scratch.allocator());
    {
        keel::f64 *cd = xc.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) cd[i] = v[i] - xmean;
    }
    auto a = ampl_fft_hilbert(xc, scratch.allocator());
    auto r = createLike(x, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < N; ++i) r.doubleDataMut()[i] = xmean + a[i];
    return r;
}

// Full multi-mode envelope: returns (yupper, ylower).
//   mode = 0 default (n unused — uses FFT |hilbert(x-mean)|)
//   mode = 1 'analytic' (n-tap Kaiser-Hilbert FIR)
//   mode = 2 'rms'      (sliding RMS over n samples)
//   mode = 3 'peak'     (spline through extrema with MinPeakDistance n)
std::pair<Value, Value>
envelope_full(const Value &x, keel::i32 mode, keel::usize n, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (mode == 3) {
        // Peak mode — no DC removal.
        keel::Array<keel::f64> up(scratch.allocator()), lo(scratch.allocator());
        env_peak(x, n, up, lo, scratch.allocator());
        Value yupper = createLike(x, ValueType::DOUBLE, mr);
        Value ylower = createLike(x, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < N; ++i) yupper.doubleDataMut()[i] = up[i];
        for (keel::usize i = 0; i < N; ++i) ylower.doubleDataMut()[i] = lo[i];
        return {std::move(yupper), std::move(ylower)};
    }

    // Modes 0/1/2: remove DC, compute amplitude, restore.
    auto v = read_real(x, scratch.allocator());
    keel::f64 xmean = 0.0;
    for (keel::f64 d : v) xmean += d;
    xmean /= keel::f64(N);
    Value xc = Value::matrix(N, 1, ValueType::DOUBLE, scratch.allocator());
    {
        keel::f64 *cd = xc.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) cd[i] = v[i] - xmean;
    }

    keel::Array<keel::f64> a(scratch.allocator());
    if      (mode == 0) a = ampl_fft_hilbert(xc, scratch.allocator());
    else if (mode == 1) a = ampl_fir(xc, n, scratch.allocator());
    else                a = ampl_rms(xc, n, scratch.allocator());

    Value yupper = createLike(x, ValueType::DOUBLE, mr);
    Value ylower = createLike(x, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < N; ++i) yupper.doubleDataMut()[i] = xmean + a[i];
    for (keel::usize i = 0; i < N; ++i) ylower.doubleDataMut()[i] = xmean - a[i];
    return {std::move(yupper), std::move(ylower)};
}

// Back-compat 2-output adapter: dispatches to default mode.
std::pair<Value, Value>
envelope_pair(const Value &x, keel::Allocator *mr)
{
    return envelope_full(x, 0, 0, mr);
}

} // namespace numkit::signal
