// toolboxes/signal/src/transforms/extras.cpp
// dftmtx / bitrevorder / dst / idst / rceps / cceps / icceps.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/transforms/extras.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/signal/filter_analysis/unwrap.hpp>      // (used externally; not needed here)
#include <numkit/signal/transforms/fft.hpp>

#include "../dsp_helpers.hpp"   // fftRadix2, fillFftTwiddles, nextPow2, Complex
#include "../signal_detail.hpp" // shared argcheck cores
#include "extras_detail.hpp"    // bitReverse (shared compute/register)

#include <numkit/value/value_type.hpp>   // mtypeName / isIntegerType



namespace numkit::signal {

// External linkage (declared in extras_detail.hpp): bit-reversal permutation
// index, shared by the engine-free compute (bitrevorder) and the CallContext
// register half (bitrevorder_reg's 2nd index output) in extras_reg.cpp.
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

// In-place DFT at the EXACT length n, matching fftRadix2's twiddle
// convention W[k] = exp(dir·2πi·k/N) (dir=-1 forward, dir=+1 inverse and
// unscaled). Radix-2 when n is a power of two; otherwise a direct O(n²)
// DFT. The cepstrum MUST transform at the signal length — zero-padding to
// a power of two corrupts log|X| at the interpolated near-zero bins (a
// padded spectral null sends log|X| to ~-690 and smears it across every
// output sample), so the historical nextPow2 path produced garbage for
// non-power-of-two lengths.
void dftExact(keel::c64 *buf, keel::usize n, keel::i32 dir, keel::Allocator *mr)
{
    if (n <= 1) return;
    if (isPow2(n)) { fftRadix2(mr, buf, n, dir); return; }
    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto out = keel::Array<keel::c64>(n, arena.allocator());
    const keel::f64 base = keel::as<keel::f64>(dir) * 2.0 * keel::kPi / keel::as<keel::f64>(n);
    for (keel::usize k = 0; k < n; ++k) {
        keel::c64 s(0.0, 0.0);
        for (keel::usize m = 0; m < n; ++m) {
            const keel::f64 a = base * keel::as<keel::f64>(k) * keel::as<keel::f64>(m);
            s += buf[m] * keel::c64(keel::cos(a), keel::sin(a));
        }
        out[k] = s;
    }
    for (keel::usize i = 0; i < n; ++i) buf[i] = out[i];
}

// A real column (or row, to match the input orientation) of length n.
Value realLike(const Value &x, keel::usize n, keel::Allocator *mr)
{
    const bool isRow = (x.dims().rows() == 1 && x.dims().cols() > 1);
    return isRow ? Value::matrix(1, n, ValueType::DOUBLE, mr)
                 : Value::matrix(n, 1, ValueType::DOUBLE, mr);
}

} // namespace

// ── dftmtx ────────────────────────────────────────────────────────────
Value dftmtx(keel::usize N, keel::Allocator *mr)
{
    if (N == 0)
        throw Error("dftmtx: N must be positive",
                     0, 0, "dftmtx", "", "numkit:dftmtx:badN");
    auto out = Value::complexMatrix(N, N, mr);
    keel::c64 *dst = out.complexDataMut();
    const keel::f64 base = -2.0 * keel::kPi / keel::as<keel::f64>(N);
    for (keel::usize k = 0; k < N; ++k)
        for (keel::usize n = 0; n < N; ++n) {
            const keel::f64 a = base * keel::as<keel::f64>(k) * keel::as<keel::f64>(n);
            dst[n * N + k] = keel::c64(keel::cos(a), keel::sin(a));   // column-major
        }
    return out;
}

// ── bitrevorder ───────────────────────────────────────────────────────
// Contract (probed R2025b): every numeric class plus logical passes and
// the result PRESERVES the class (this is a pure permutation, so
// single/logical/complex results are exact copies); integer classes
// would need integer-class outputs the Value layer cannot allocate yet
// (bugs/opened/signal/integer-class-outputs.md); char is rejected
// (MustBeNumericOrLogical).
namespace {

void bitrevorder_argcheck(const Value &x)
{
    detail::requireNumericOrLogicalInput(x, "bitrevorder");
}

} // namespace

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

// ── dst / idst (Type-II) ──────────────────────────────────────────────
// Y[k] = sum x[n] sin(π (n+1) (k+1) / (N+1)),  k=0..N-1.
// Direct O(N²) — acceptable for N up to a few thousand.
Value dst(const Value &x, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *dst = out.doubleDataMut();
    const keel::f64 scale = keel::kPi / keel::as<keel::f64>(N + 1);
    for (keel::usize k = 0; k < N; ++k) {
        keel::f64 s = 0.0;
        for (keel::usize n = 0; n < N; ++n)
            s += x.elemAsDouble(n) * keel::sin(scale * (n + 1) * (k + 1));
        dst[k] = s;
    }
    return out;
}

Value idst(const Value &y, keel::Allocator *mr)
{
    // Type-II DST is self-inverse up to a factor of 2/(N+1).
    auto x = dst(y, mr);
    const keel::usize N = x.numel();
    if (N == 0) return x;
    const keel::f64 scale = 2.0 / keel::as<keel::f64>(N + 1);
    keel::f64 *p = x.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) p[i] *= scale;
    return x;
}

// ── rceps ─────────────────────────────────────────────────────────────
// Real cepstrum xhat = real(ifft(log|fft(x)|)), transformed at the exact
// signal length n (see dftExact).
Value rceps(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (n == 0) return realLike(x, 0, mr);
    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto X = keel::Array<keel::c64>(n, arena.allocator());
    for (keel::usize i = 0; i < n; ++i) X[i] = keel::c64(x.elemAsDouble(i), 0.0);
    dftExact(X.data(), n, +1, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 mag = keel::abs(X[i]);
        X[i] = keel::c64(keel::log(keel::max(mag, 1e-300)), 0.0);
    }
    dftExact(X.data(), n, -1, mr);
    const keel::f64 invN = 1.0 / keel::as<keel::f64>(n);
    auto out = realLike(x, n, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) dst[i] = X[i].real() * invN;
    return out;
}

// ── rceps (2-output) ───────────────────────────────────────────────────
// Also returns the minimum-phase reconstruction
//   ym = real(ifft(exp(fft(w .* xhat))))
// where w is the cepstral folding window [1, 2,…,2, (1 if n even), 0,…,0].
keel::Pair<Value, Value> rcepsMinPhase(const Value &x, keel::Allocator *mr)
{
    Value y = rceps(x, mr);
    const keel::usize n = x.numel();
    Value ym = realLike(x, n, mr);
    if (n == 0) return {keel::move(y), keel::move(ym)};
    const keel::f64 *yd = y.doubleData();
    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto W = keel::Array<keel::c64>(n, arena.allocator());
    const keel::usize khalf = (n - 1) / 2;
    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 w;
        if (i == 0)                              w = 1.0;
        else if (i <= khalf)                     w = 2.0;
        else if (n % 2 == 0 && i == n / 2)       w = 1.0;   // Nyquist (even n)
        else                                     w = 0.0;
        W[i] = keel::c64(w * yd[i], 0.0);
    }
    dftExact(W.data(), n, -1, mr);                // fft
    for (keel::usize i = 0; i < n; ++i) W[i] = keel::exp(W[i]);
    dftExact(W.data(), n, +1, mr);                // ifft (unscaled)
    const keel::f64 invN = 1.0 / keel::as<keel::f64>(n);
    keel::f64 *md = ym.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) md[i] = W[i].real() * invN;
    return {keel::move(y), keel::move(ym)};
}

// ── cceps ─────────────────────────────────────────────────────────────
// Complex cepstrum: ifft(log(fft(x))) with simple phase unwrapping along
// the frequency axis. Only the real part of x is used (matches MATLAB
// when input is real).
// Sign-convention note (nk's fftRadix2 dir argument):
//   dir = -1  →  W[k] = exp(-2πj k/N)  →  FORWARD DFT
//   dir = +1  →  W[k] = exp(+2πj k/N)  →  INVERSE DFT (caller scales by 1/N)
// The inverse-step here MUST be dir=+1; using -1 produces a forward DFT
// which time-reverses the output .
Value cceps(const Value &x, keel::f64 *ndOut, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (ndOut) *ndOut = 0.0;
    if (n == 0) return realLike(x, 0, mr);
    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto X = keel::Array<keel::c64>(n, arena.allocator());
    for (keel::usize i = 0; i < n; ++i) X[i] = keel::c64(x.elemAsDouble(i), 0.0);
    dftExact(X.data(), n, -1, mr);                  // FORWARD fft (h); the inverse
                                                    // step below is dir=+1. (Using
                                                    // +1 here too time-reverses the
                                                    // cepstrum and flips nd's sign.)
    // log magnitude + UNWRAPPED phase (relative to the previous bin = standard
    // unwrap), then apply MATLAB's rcunwrap linear-phase removal: the cepstrum
    // needs no phase discontinuity at ±pi, so subtract the integer-lag term
    //   nd = round(unwrapped(nh+1)/pi);  phase -= pi*nd*(0:n-1)/nh
    // with nh = fix((n+1)/2). `nd` is the (circular) delay removed (2nd output).
    // Without this term the per-sample result diverges past DC for non-2^n n
    // (see bugs/signal/cceps-nd-phase).
    auto ph   = keel::Array<keel::f64>(n, arena.allocator());
    auto lmag = keel::Array<keel::f64>(n, arena.allocator());
    keel::f64 prevPhase = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 mag = keel::abs(X[i]);
        keel::f64 phase = keel::arg(X[i]);
        while (phase - prevPhase > keel::kPi)  phase -= 2.0 * keel::kPi;
        while (phase - prevPhase < -keel::kPi) phase += 2.0 * keel::kPi;
        prevPhase = phase;
        ph[i]   = phase;
        lmag[i] = keel::log(keel::max(mag, 1e-300));
    }
    const keel::usize nh  = (n + 1) / 2;                 // fix((n+1)/2)
    const keel::usize idx = (n == 1) ? 0 : nh;           // MATLAB's 1-based nh+1
    const keel::f64 nd  = keel::round(ph[idx] / keel::kPi);
    for (keel::usize i = 0; i < n; ++i)
        ph[i] -= keel::kPi * nd * keel::as<keel::f64>(i) / keel::as<keel::f64>(nh);
    if (ndOut) *ndOut = nd;
    for (keel::usize i = 0; i < n; ++i) X[i] = keel::c64(lmag[i], ph[i]);
    dftExact(X.data(), n, +1, mr);
    const keel::f64 invN = 1.0 / keel::as<keel::f64>(n);
    auto out = realLike(x, n, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) dst[i] = X[i].real() * invN;
    return out;
}

// ── icceps ────────────────────────────────────────────────────────────
// Inverse complex cepstrum: real(ifft(exp(fft(c)))). FORWARD fft is dir=-1,
// the inverse pass dir=+1 — mirroring cceps. (Using +1 for both time-reverses
// the result, which the cceps fix exposed via the round-trip test. Without an
// `nd` argument this is MATLAB's icceps(xhat) with nd=0; it recovers cceps's
// input up to the circular shift cceps removed.)
Value icceps(const Value &c, keel::Allocator *mr)
{
    const keel::usize n = c.numel();
    if (n == 0) return realLike(c, 0, mr);
    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto X = keel::Array<keel::c64>(n, arena.allocator());
    for (keel::usize i = 0; i < n; ++i) X[i] = keel::c64(c.elemAsDouble(i), 0.0);
    dftExact(X.data(), n, -1, mr);                  // FORWARD fft
    for (keel::usize i = 0; i < n; ++i) X[i] = keel::exp(X[i]);
    dftExact(X.data(), n, +1, mr);                  // inverse
    const keel::f64 invN = 1.0 / keel::as<keel::f64>(n);
    auto out = realLike(c, n, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) dst[i] = X[i].real() * invN;
    return out;
}

} // namespace numkit::signal
