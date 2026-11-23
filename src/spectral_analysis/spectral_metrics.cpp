// toolboxes/signal/src/spectral_analysis/spectral_metrics.cpp
//
// Spectral measurement functions on power spectral densities. All take
// either a time-series (PSD computed via periodogram with rectangular
// window) or, in select cases, a pre-computed (Pxx, F).
//
// Frequency convention: when fs is omitted, frequencies run [0, π] and
// power-band integrations yield rad/s. When fs is supplied, freqs run
// [0, fs/2] and integrations yield Hz.

#include <keel/array.h>
#include <keel/tuple.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/spectral_analysis/spectral_metrics.hpp>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/signal/transforms/hilbert.hpp>
#include <numkit/signal/windows/windows.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "../signal_detail.hpp"     // shared argcheck cores



namespace numkit::signal {

namespace {

keel::f64 scalarOr(const Value &v, keel::f64 dflt)
{
    return v.isEmpty() ? dflt : v.toScalar();
}

// Compute single-segment periodogram of x, returning (Pxx, F) where F
// is in physical units: [0, fs/2] when fs > 0, else [0, π] when fs <= 0.
struct PsdPair {
    keel::Array<keel::f64> Pxx;
    keel::Array<keel::f64> F;
};

PsdPair computePsd(const Value &x, keel::f64 fs, keel::Allocator *mr)
{
    // periodogram(x, [], nfft, effFs) returns Pxx already normalised by
    // 1/(winPower * effFs) with one-sided doubling on interior bins, so
    // sum(Pxx) * df = mean(x^2) (Parseval). Use nfft = x.numel() (matching
    // MATLAB's bandpower/obw rectangular-window periodogram at length N).
    Value win;  // empty → rectangular
    const keel::f64 effFs = (fs > 0.0) ? fs : (2.0 * keel::kPi);
    Value xD = detail::widenSingleToDouble(x, mr);
    auto [Pxx, F] = periodogram(xD, win, xD.numel(), effFs, mr);
    const keel::usize n = Pxx.numel();
    PsdPair p{keel::Array<keel::f64>(n, mr), keel::Array<keel::f64>(n, mr)};
    const keel::f64 *fd = F.doubleData();
    const keel::f64 *pd = Pxx.doubleData();
    for (keel::usize i = 0; i < n; ++i) {
        p.F[i]   = fd[i];
        p.Pxx[i] = pd[i];
    }
    return p;
}

// Trapezoidal integration of P over the frequency grid F, with a band
// restriction to [fLo, fHi] (inclusive). Endpoints are linearly
// interpolated when they fall inside an interval.
keel::f64 integrate(keel::ArrayRef<keel::f64> P, keel::ArrayRef<keel::f64> F,
                 keel::f64 fLo, keel::f64 fHi)
{
    const keel::usize n = P.size();
    if (n < 2) return 0.0;
    if (fLo > fHi) keel::swap(fLo, fHi);
    keel::f64 s = 0.0;
    for (keel::usize i = 0; i + 1 < n; ++i) {
        keel::f64 a = F[i], b = F[i + 1];
        if (b <= fLo) continue;
        if (a >= fHi) break;
        keel::f64 pa = P[i], pb = P[i + 1];
        // Clip interval to band.
        if (a < fLo) {
            const keel::f64 t = (fLo - a) / (b - a);
            pa = pa + t * (pb - pa);
            a = fLo;
        }
        if (b > fHi) {
            const keel::f64 t = (fHi - a) / (F[i + 1] - F[i]);
            pb = P[i] + t * (P[i + 1] - P[i]);
            b = fHi;
        }
        s += 0.5 * (pa + pb) * (b - a);
    }
    return s;
}

} // anonymous

// ── bandpower ──────────────────────────────────────────────────────

Value bandpower(const Value &x, const Value &fs, const Value &freqrange, keel::Allocator *mr)
{
    const bool xSingle = (x.type() == ValueType::SINGLE);
    if (fs.isEmpty()) {
        detail::requireFloatInput(x, "bandpower");
        // 1-arg: total signal power = mean(|x|^2). Cheaper than going
        // through PSD (Parseval's theorem makes them equal anyway).
        const keel::usize n = x.numel();
        if (n == 0) return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), xSingle, mr);
        keel::f64 s = 0.0;
        if (x.isComplex()) {
            const keel::c64 *xd = x.complexData();
            for (keel::usize i = 0; i < n; ++i)
                s += keel::norm(xd[i]);
        } else {
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 v = x.elemAsDouble(i);
                s += v * v;
            }
        }
        return detail::narrowDoubleToSingleIf(Value::scalar(s / n, mr), xSingle, mr);
    }
    detail::requireFloatInput(x, "bandpower");
    if (freqrange.isEmpty())
        throw Error("bandpower: fs requires freqrange — specify both "
                    "together (bandpower(x, fs, freqrange))",
                    0, 0, "bandpower", "", "numkit:bandpower:freqRangeRequired");
    if (freqrange.numel() != 2)
        throw Error(keel::format("bandpower: freqrange must be a 2-element vector [fmin, fmax] (got {} elements)",
                                 freqrange.numel()),
                    0, 0, "bandpower", "", "numkit:bandpower:freqRange");
    if (fs.numel() != 1 || fs.isComplex())
        throw Error("bandpower: fs must be a real scalar",
                    0, 0, "bandpower", "", "numkit:bandpower:fs");
    const keel::f64 fsv = fs.toScalar();
    auto p = computePsd(x, fsv, mr);
    const keel::f64 fLo = freqrange.elemAsDouble(0);
    const keel::f64 fHi = freqrange.elemAsDouble(1);
    return detail::narrowDoubleToSingleIf(Value::scalar(integrate(p.Pxx, p.F, fLo, fHi), mr), xSingle, mr);
}

Value bandpowerPsd(const Value &pxx, const Value &f, const Value &freqrange, keel::Allocator *mr)
{
    detail::requireRealFloatInput(pxx, "bandpower", "pxx", 1);
    detail::requireRealFloatInput(f, "bandpower", "f", 2);
    const bool anySingle = (pxx.type() == ValueType::SINGLE) || (f.type() == ValueType::SINGLE);
    const keel::usize n = pxx.numel();
    if (f.numel() != n || n < 2)
        throw Error("bandpower: frequency vector f must have the same length as pxx (>= 2)",
                    0, 0, "bandpower", "", "numkit:bandpower:freqVecLength");
    keel::Array<keel::f64> P(n, mr), F(n, mr);
    for (keel::usize i = 0; i < n; ++i) {
        P[i] = pxx.elemAsDouble(i);
        F[i] = f.elemAsDouble(i);
    }
    if (freqrange.isEmpty()) {
        // Full-range rectangular integration matching MATLAB's psdbandpower:
        // each bin has width w_i = (F[1]-F[0]) at i=0, (F[i+1]-F[i-1])/2 interior,
        // and (F[n-1]-F[n-2]) at i=n-1 (on a uniform grid every bin has width df).
        keel::f64 total = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 w = 0.0;
            if (i == 0) w = F[1] - F[0];
            else if (i + 1 == n) w = F[n - 1] - F[n - 2];
            else w = 0.5 * (F[i + 1] - F[i - 1]);
            total += P[i] * w;
        }
        return detail::narrowDoubleToSingleIf(Value::scalar(total, mr), anySingle, mr);
    }
    if (freqrange.numel() != 2)
        throw Error(keel::format("bandpower: freqrange must be a 2-element vector [fmin, fmax] (got {} elements)",
                                 freqrange.numel()),
                    0, 0, "bandpower", "", "numkit:bandpower:freqRange");
    const keel::f64 fLo = freqrange.elemAsDouble(0);
    const keel::f64 fHi = freqrange.elemAsDouble(1);
    if (fLo <= F.front() && fHi >= F.back()) {
        keel::f64 total = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 w = 0.0;
            if (i == 0) w = F[1] - F[0];
            else if (i + 1 == n) w = F[n - 1] - F[n - 2];
            else w = 0.5 * (F[i + 1] - F[i - 1]);
            total += P[i] * w;
        }
        return detail::narrowDoubleToSingleIf(Value::scalar(total, mr), anySingle, mr);
    }
    return detail::narrowDoubleToSingleIf(Value::scalar(integrate(P, F, fLo, fHi), mr), anySingle, mr);
}

// ── meanfreq / medfreq ────────────────────────────────────────────

Value meanfreq(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    keel::f64 num = 0.0, den = 0.0;
    for (keel::usize i = 0; i + 1 < p.F.size(); ++i) {
        const keel::f64 df = p.F[i + 1] - p.F[i];
        const keel::f64 pmid = 0.5 * (p.Pxx[i] + p.Pxx[i + 1]);
        const keel::f64 fmid = 0.5 * (p.F[i] + p.F[i + 1]);
        num += fmid * pmid * df;
        den += pmid * df;
    }
    return Value::scalar(den > 0 ? num / den : 0.0, mr);
}

Value medfreq(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    if (p.F.size() < 2) return Value::scalar(0.0, mr);
    // Cumulative trapezoid of P over F.
    keel::Array<keel::f64> cum(p.F.size(), 0.0, mr);
    for (keel::usize i = 1; i < p.F.size(); ++i) {
        cum[i] = cum[i - 1]
               + 0.5 * (p.Pxx[i - 1] + p.Pxx[i]) * (p.F[i] - p.F[i - 1]);
    }
    const keel::f64 half = 0.5 * cum.back();
    if (cum.back() <= 0.0) return Value::scalar(0.0, mr);
    // Linear-interp the frequency where cum == half.
    for (keel::usize i = 1; i < cum.size(); ++i) {
        if (cum[i] >= half) {
            const keel::f64 t = (half - cum[i - 1]) / (cum[i] - cum[i - 1]);
            return Value::scalar(p.F[i - 1] + t * (p.F[i] - p.F[i - 1]), mr);
        }
    }
    return Value::scalar(p.F.back(), mr);
}

// ── enbw ──────────────────────────────────────────────────────────

Value enbw(const Value &window, const Value &fs, keel::Allocator *mr)
{
    const keel::usize n = window.numel();
    if (n == 0) return Value::scalar(0.0, mr);
    keel::f64 sumsq = 0.0, sm = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 w = window.elemAsDouble(i);
        sumsq += w * w;
        sm    += w;
    }
    const keel::f64 bin = (sm * sm > 0) ? n * sumsq / (sm * sm) : 0.0;
    const keel::f64 fsv = scalarOr(fs, keel::as<keel::f64>(n));
    return Value::scalar(bin * fsv / n, mr);
}

// ── obw ───────────────────────────────────────────────────────────

keel::Tuple<Value, Value, Value, Value>
obw(const Value &x, const Value &fs, keel::f64 p, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    const keel::usize N = x.numel();
    auto zeros = [&]() {
        Value z = Value::scalar(0.0, mr);
        return keel::make_tuple(z, z, z, z);
    };
    if (N < 2) return zeros();

    // MATLAB obw uses a rectangular-windowed periodogram at nfft = N (no
    // zero-padding). nk's `periodogram` helper is radix-2 only, so compute
    // the length-N DFT through the general fft (Bluestein for non-pow2 N) and
    // form the one-sided PSD here: Pxx[k] = (2 if folded)·|X[k]|² / (fs·N),
    // which obeys Σ Pxx·df = mean(x²) (rectangular window power = N).
    Value          Xv = fft(x, keel::as<keel::i32>(N), /*dim=*/0, mr);
    const keel::c64 *X  = Xv.complexData();
    const keel::usize   nOut = N / 2 + 1;
    const keel::f64   fsHz = (fsv > 0.0) ? fsv : (2.0 * keel::kPi);  // rad/s when fs omitted
    const keel::f64   df   = fsHz / keel::as<keel::f64>(N);

    keel::Array<keel::f64> P(nOut, mr), F(nOut, mr);
    for (keel::usize k = 0; k < nOut; ++k) {
        keel::f64 mag2 = keel::norm(X[k]);
        if (k > 0 && 2 * k != N) mag2 *= 2.0;   // one-sided fold (skip DC & Nyquist)
        P[k] = mag2 / (fsHz * keel::as<keel::f64>(N));
        F[k] = keel::as<keel::f64>(k) * df;
    }

    // Rectangle-rule cumulative power (MATLAB obw): cum[k] = Σ_{j≤k} P[j]·df.
    keel::Array<keel::f64> cum(nOut, 0.0, mr);
    cum[0] = P[0] * df;
    for (keel::usize i = 1; i < nOut; ++i)
        cum[i] = cum[i - 1] + P[i] * df;
    const keel::f64 total = cum.back();
    if (total <= 0.0) return zeros();

    const keel::f64 pLow  = (1.0 - p) * 0.5 * total;
    const keel::f64 pHigh = (1.0 + p) * 0.5 * total;
    // A band edge sits at the bin's UPPER edge (F + df/2) where the cumulative
    // power reaches the target — matches MATLAB exactly.
    auto findF = [&](keel::f64 target) {
        for (keel::usize i = 1; i < nOut; ++i) {
            if (cum[i] >= target) {
                const keel::f64 t = (target - cum[i - 1])
                               / keel::max(cum[i] - cum[i - 1], 1e-300);
                return F[i - 1] + t * (F[i] - F[i - 1]) + 0.5 * df;
            }
        }
        return F.back() + 0.5 * df;
    };
    const keel::f64 flo = findF(pLow);
    const keel::f64 fhi = findF(pHigh);
    return keel::make_tuple(Value::scalar(fhi - flo, mr),
                           Value::scalar(flo, mr),
                           Value::scalar(fhi, mr),
                           Value::scalar(p * total, mr));
}

// ── powerbw ───────────────────────────────────────────────────────

Value powerbw(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    if (p.F.size() < 2) return Value::scalar(0.0, mr);
    // Find dominant peak.
    keel::usize pk = 0;
    keel::f64 pkV = p.Pxx[0];
    for (keel::usize i = 1; i < p.Pxx.size(); ++i)
        if (p.Pxx[i] > pkV) { pkV = p.Pxx[i]; pk = i; }
    const keel::f64 thresh = pkV * 0.5;
    // Walk left and right from pk to find the half-power crossings.
    auto interpCross = [&](keel::usize i, keel::usize j) {
        const keel::f64 a = p.Pxx[i], b = p.Pxx[j];
        if (a == b) return p.F[i];
        const keel::f64 t = (thresh - a) / (b - a);
        return p.F[i] + t * (p.F[j] - p.F[i]);
    };
    keel::f64 fLo = p.F.front(), fHi = p.F.back();
    for (keel::usize i = pk; i > 0; --i)
        if (p.Pxx[i - 1] < thresh) { fLo = interpCross(i - 1, i); break; }
    for (keel::usize i = pk; i + 1 < p.Pxx.size(); ++i)
        if (p.Pxx[i + 1] < thresh) { fHi = interpCross(i, i + 1); break; }
    return Value::scalar(fHi - fLo, mr);
}

// ── spectral shape statistics ─────────────────────────────────────
// All of these treat the normalised PSD as a probability distribution
// over frequency.

namespace {

struct PsdMoments {
    keel::f64 sumP;
    keel::f64 mean;
    keel::f64 var;
    keel::f64 m3;   // third central moment
    keel::f64 m4;   // fourth central moment
};

PsdMoments computePsdMoments(keel::ArrayRef<keel::f64> P,
                             keel::ArrayRef<keel::f64> F)
{
    keel::f64 sumP = 0.0;
    for (keel::f64 v : P) sumP += v;
    if (sumP <= 0.0) return {0.0, 0.0, 0.0, 0.0, 0.0};
    keel::f64 mean = 0.0;
    for (keel::usize i = 0; i < P.size(); ++i) mean += F[i] * P[i];
    mean /= sumP;
    keel::f64 var = 0.0, m3 = 0.0, m4 = 0.0;
    for (keel::usize i = 0; i < P.size(); ++i) {
        const keel::f64 d = F[i] - mean;
        var += d * d * P[i];
        m3  += d * d * d * P[i];
        m4  += d * d * d * d * P[i];
    }
    var /= sumP;
    m3  /= sumP;
    m4  /= sumP;
    return {sumP, mean, var, m3, m4};
}

} // anonymous

Value spectralcrest(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    if (p.Pxx.empty()) return Value::scalar(0.0, mr);
    keel::f64 mx = p.Pxx[0], sm = 0.0;
    for (keel::f64 v : p.Pxx) { if (v > mx) mx = v; sm += v; }
    const keel::f64 mean = sm / p.Pxx.size();
    return Value::scalar(mean > 0 ? mx / mean : 0.0, mr);
}

Value spectralflatness(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    if (p.Pxx.empty()) return Value::scalar(0.0, mr);
    keel::f64 logSum = 0.0, sm = 0.0;
    keel::usize valid = 0;
    for (keel::f64 v : p.Pxx) {
        sm += v;
        if (v > 0) { logSum += keel::log(v); ++valid; }
    }
    if (valid == 0 || sm <= 0.0) return Value::scalar(0.0, mr);
    const keel::f64 geom = keel::exp(logSum / valid);
    const keel::f64 arith = sm / p.Pxx.size();
    return Value::scalar(geom / arith, mr);
}

Value spectralentropy(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    if (p.Pxx.empty()) return Value::scalar(0.0, mr);
    keel::f64 sm = 0.0;
    for (keel::f64 v : p.Pxx) sm += v;
    if (sm <= 0.0) return Value::scalar(0.0, mr);
    keel::f64 H = 0.0;
    for (keel::f64 v : p.Pxx) {
        const keel::f64 q = v / sm;
        if (q > 0) H -= q * keel::log(q);
    }
    // Normalise by log(N) so the result lies in [0, 1].
    return Value::scalar(H / keel::log(keel::as<keel::f64>(p.Pxx.size())), mr);
}

Value spectralkurtosis(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    auto m = computePsdMoments(p.Pxx, p.F);
    if (m.var <= 0.0) return Value::scalar(0.0, mr);
    return Value::scalar(m.m4 / (m.var * m.var), mr);
}

Value spectralskewness(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsd(x, fsv, mr);
    auto m = computePsdMoments(p.Pxx, p.F);
    if (m.var <= 0.0) return Value::scalar(0.0, mr);
    return Value::scalar(m.m3 / keel::pow(m.var, 1.5), mr);
}

// ════════════════════════════════════════════════════════════════════
// Harmonic / SNR family
// ════════════════════════════════════════════════════════════════════
//
// Strategy: compute a Kaiser-windowed periodogram (matches MATLAB's
// default snr: Kaiser(N, 38) gives near-zero sidelobes so leakage
// stays below realistic noise floors), find the dominant peak skipping
// a few DC bins, then sum a "skirt" of `kSkirtBins` bins on each side
// as the fundamental power. Repeat for harmonics. Noise = remainder.

namespace {

PsdPair computePsdKaiser(const Value &x, keel::f64 fs, keel::Allocator *mr)
{
    // Build Kaiser(beta=38) window of length numel(x).
    Value win = kaiser(x.numel(), 38.0, mr);
    auto [Pxx, F] = periodogram(x, win, /*nfft=*/0, /*fs=*/keel::kTau, mr);
    const keel::usize n = Pxx.numel();
    PsdPair p;
    p.Pxx.resize(n);
    p.F.resize(n);
    const keel::usize nfft = (n > 1) ? 2 * (n - 1) : 1;
    if (fs > 0.0) {
        const keel::f64 fScale = fs / (2.0 * keel::kPi);
        const keel::f64 pScale = keel::as<keel::f64>(nfft) / fs;
        for (keel::usize i = 0; i < n; ++i) {
            p.F[i]   = F.doubleData()[i] * fScale;
            p.Pxx[i] = Pxx.doubleData()[i] * pScale;
        }
    } else {
        const keel::f64 pScale = keel::as<keel::f64>(nfft) / (2.0 * keel::kPi);
        for (keel::usize i = 0; i < n; ++i) {
            p.F[i]   = F.doubleData()[i];
            p.Pxx[i] = Pxx.doubleData()[i] * pScale;
        }
    }
    return p;
}

constexpr keel::i32 kSkirtBins  = 8;   // half-width of fundamental skirt
constexpr keel::i32 kHarmonics  = 6;   // number of harmonics to track
constexpr keel::i32 kDcBins     = 8;   // DC bins to ignore when finding peak

// Sum P over [lo, hi) inclusive of lo, exclusive of hi (uses raw bin
// values; for narrow peaks this is close to the trapezoidal estimate
// times df, scaled identically across all sums so the ratio in dB is
// unchanged).
[[maybe_unused]] keel::f64 sumBins(keel::ArrayRef<keel::f64> P, keel::i32 lo, keel::i32 hi)
{
    if (lo < 0) lo = 0;
    if (hi > keel::as<keel::i32>(P.size())) hi = keel::as<keel::i32>(P.size());
    keel::f64 s = 0.0;
    for (keel::i32 i = lo; i < hi; ++i) s += P[i];
    return s;
}

// Find the bin of the dominant peak, ignoring the first kDcBins.
keel::i32 findFundamentalBin(keel::ArrayRef<keel::f64> P)
{
    keel::i32 pk = kDcBins;
    keel::f64 pkV = -1.0;
    for (keel::i32 i = kDcBins; i < keel::as<keel::i32>(P.size()); ++i)
        if (P[i] > pkV) { pkV = P[i]; pk = i; }
    return pk;
}

struct HarmonicAnalysis {
    keel::f64 pSig;     // fundamental power
    keel::f64 pHarm;    // sum of harmonic powers
    keel::f64 pNoise;   // remainder (everything else, excluding DC bins)
    keel::f64 pSpurMax; // largest single spur OUTSIDE the fundamental skirt
    keel::i32    f0Bin;
};

HarmonicAnalysis analyseHarmonics(keel::ArrayRef<keel::f64> P)
{
    const keel::i32 nbins = keel::as<keel::i32>(P.size());
    HarmonicAnalysis r{0, 0, 0, 0, 0};
    if (nbins < 2) return r;

    r.f0Bin = findFundamentalBin(P);
    const keel::i32 f0 = r.f0Bin;

    // Mark which bins belong to the fundamental skirt or a harmonic
    // skirt. Everything else contributes to noise (after skipping DC
    // bins at the start).
    keel::Array<keel::u8> mark(nbins, 0); // 1=fund, 2=harm
    for (keel::i32 j = keel::max(0, f0 - kSkirtBins);
         j < keel::min(nbins, f0 + kSkirtBins + 1); ++j) mark[j] = 1;
    for (keel::i32 n = 2; n <= kHarmonics; ++n) {
        const keel::i32 bin = n * f0;
        if (bin >= nbins) break;
        for (keel::i32 j = keel::max(0, bin - kSkirtBins);
             j < keel::min(nbins, bin + kSkirtBins + 1); ++j) {
            if (mark[j] == 0) mark[j] = 2;
        }
    }

    for (keel::i32 i = 0; i < nbins; ++i) {
        if (i < kDcBins && mark[i] == 0) continue; // ignore DC noise contrib
        if      (mark[i] == 1) r.pSig  += P[i];
        else if (mark[i] == 2) r.pHarm += P[i];
        else                   r.pNoise += P[i];
    }

    // Largest spur: scan for local maxima OUTSIDE the fundamental skirt
    // and take the biggest peak value (not its surrounding bin sum —
    // sfdr is a single-bin / peak-to-peak metric).
    for (keel::i32 i = 1; i + 1 < nbins; ++i) {
        if (mark[i] == 1) continue;        // skip fundamental skirt
        if (i < kDcBins) continue;
        if (P[i] >= P[i - 1] && P[i] >= P[i + 1] && P[i] > r.pSpurMax)
            r.pSpurMax = P[i];
    }
    return r;
}

keel::f64 safeDb(keel::f64 num, keel::f64 den)
{
    if (num <= 0.0 || den <= 0.0)
        return keel::Limits<keel::f64>::quiet_nan();
    return 10.0 * keel::log10(num / den);
}

} // anonymous

Value snr(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsdKaiser(x, fsv, mr);
    auto h = analyseHarmonics(p.Pxx);
    return Value::scalar(safeDb(h.pSig, h.pNoise), mr);
}

Value sinad(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsdKaiser(x, fsv, mr);
    auto h = analyseHarmonics(p.Pxx);
    return Value::scalar(safeDb(h.pSig, h.pNoise + h.pHarm), mr);
}

Value thd(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::f64 fsv = scalarOr(fs, 0.0);
    auto p = computePsdKaiser(x, fsv, mr);
    auto h = analyseHarmonics(p.Pxx);
    return Value::scalar(safeDb(h.pHarm, h.pSig), mr);
}

Value sfdr(const Value &x, const Value &fs, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (n < 2)
        return Value::scalar(keel::Limits<keel::f64>::quiet_nan(), mr);
    const keel::f64 fsScalar = (!fs.isEmpty()) ? fs.toScalar() : 1.0;
    Value w = kaiser(n, 38.0, mr);
    const keel::f64 *wd = w.doubleData();
    keel::f64 sumW = 0.0, sumW2 = 0.0;
    Value xw = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    keel::f64 *xwd = xw.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 wi = wd[i];
        sumW += wi;
        sumW2 += wi * wi;
        xwd[i] = x.elemAsDouble(i) * wi;
    }
    const keel::f64 rbw = (sumW > 0.0) ? (fsScalar * sumW2 / (sumW * sumW)) : 0.0;
    Value Xv = fft(xw, keel::as<keel::i32>(n), 0, mr);
    const keel::c64 *X = Xv.complexData();
    const keel::usize nBins = n / 2 + 1;
    const keel::f64 df = fsScalar / keel::as<keel::f64>(n);
    keel::Array<keel::f64> Pxx(nBins, mr), F(nBins, mr);
    for (keel::usize k = 0; k < nBins; ++k) {
        keel::f64 mag2 = keel::norm(X[k]);
        if (k > 0 && 2 * k != n)
            mag2 *= 2.0;
        Pxx[k] = mag2 / (fsScalar * sumW2);
        F[k] = keel::as<keel::f64>(k) * df;
    }

    auto getTone = [&](keel::i32 idxTone, keel::i32 &outLeft, keel::i32 &outRight) -> keel::f64 {
        keel::i32 iL = idxTone - 1;
        keel::i32 iR = idxTone + 1;
        while (iL >= 0 && Pxx[iL] <= Pxx[iL + 1]) --iL;
        while (iR < keel::as<keel::i32>(nBins) && Pxx[iR - 1] >= Pxx[iR]) ++iR;
        ++iL;
        --iR;
        outLeft = iL;
        outRight = iR;
        keel::f64 power = 0.0;
        if (iL < iR) {
            keel::i32 len = iR - iL + 1;
            for (keel::i32 k = 0; k < len; ++k) {
                keel::f64 wBin = 0.0;
                if (k == 0) wBin = F[iL + 1] - F[iL];
                else if (k + 1 == len) wBin = F[iR] - F[iR - 1];
                else wBin = 0.5 * (F[iL + k + 1] - F[iL + k - 1]);
                power += Pxx[iL + k] * wBin;
            }
        } else {
            power = Pxx[iR] * df;
        }
        if (power < rbw * Pxx[idxTone])
            power = rbw * Pxx[idxTone];
        return power;
    };

    // Bump DC component by 3 dB and remove it
    Pxx[0] *= 2.0;
    keel::i32 dcL = 0, dcR = 0;
    getTone(0, dcL, dcR);
    for (keel::i32 k = dcL; k <= dcR; ++k) Pxx[k] = 0.0;

    // Fundamental peak
    keel::i32 iFund = keel::as<keel::i32>(keel::max_element(Pxx.begin(), Pxx.end()) - Pxx.begin());
    keel::i32 fL = 0, fR = 0;
    keel::f64 Pfund = getTone(iFund, fL, fR);
    for (keel::i32 k = fL; k <= fR; ++k) Pxx[k] = 0.0;

    // Max spur
    keel::i32 spurBin = keel::as<keel::i32>(keel::max_element(Pxx.begin(), Pxx.end()) - Pxx.begin());
    keel::i32 iLeftBin = keel::max(0, spurBin - 1);
    keel::i32 iRightBin = keel::min(keel::as<keel::i32>(nBins) - 1, spurBin + 1);
    keel::i32 iSpur = iLeftBin;
    for (keel::i32 k = iLeftBin + 1; k <= iRightBin; ++k)
        if (Pxx[k] > Pxx[iSpur]) iSpur = k;
    keel::i32 sL = 0, sR = 0;
    keel::f64 Pspur = getTone(iSpur, sL, sR);

    return Value::scalar(safeDb(Pfund, Pspur), mr);
}

// ════════════════════════════════════════════════════════════════════
// Power spectrum / spectrogram (pspectrum) & instantaneous metrics
// ════════════════════════════════════════════════════════════════════

namespace {

// Read complex Value (output of hilbert) into separate re/im arrays.
void readComplex(const Value &z, keel::Array<keel::f64> &re, keel::Array<keel::f64> &im)
{
    const keel::usize n = z.numel();
    re.resize(n); im.resize(n);
    if (z.type() == ValueType::COMPLEX) {
        const keel::c64 *c = z.complexData();
        for (keel::usize i = 0; i < n; ++i) { re[i] = c[i].real(); im[i] = c[i].imag(); }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            re[i] = z.elemAsDouble(i);
            im[i] = 0.0;
        }
    }
}

void pspectrum_argcheck(const Value &x, const Value &fsOrT,
                        const PspectrumOptions &opts, const char *caller)
{
    detail::requireFloatInput(x, caller);
    if (x.isEmpty() || x.numel() < 2)
        throw Error(keel::format("{}: input signal X must have at least 2 samples", caller),
                    0, 0, caller, "", keel::format("numkit:{}:invalidSignal", caller));
    if (x.dims().ndims() > 2)
        throw Error(keel::format("{}: input signal X must be a vector or 2-D matrix", caller),
                    0, 0, caller, "", keel::format("numkit:{}:expected2D", caller));
    if (opts.type != "power" && opts.type != "spectrogram")
        throw Error(keel::format("{}: spectrum type must be 'power' or 'spectrogram'", caller),
                    0, 0, caller, "", keel::format("numkit:{}:invalidType", caller));
    if (!keel::is_finite(opts.leakage) || opts.leakage < 0.0 || opts.leakage > 1.0)
        throw Error(keel::format("{}: 'Leakage' must be in [0, 1]", caller),
                    0, 0, caller, "", keel::format("numkit:{}:invalidLeakage", caller));
    if (opts.overlapPercent >= 100.0 || (opts.overlapPercent < 0.0 && opts.overlapPercent != -1.0))
        throw Error(keel::format("{}: 'OverlapPercent' must be in [0, 100)", caller),
                    0, 0, caller, "", keel::format("numkit:{}:invalidOverlap", caller));
    if (opts.timeResolution <= 0.0 && opts.timeResolution != -1.0)
        throw Error(keel::format("{}: 'TimeResolution' must be positive", caller),
                    0, 0, caller, "", keel::format("numkit:{}:invalidTimeResolution", caller));
    if (!fsOrT.isEmpty()) {
        detail::requireRealFloatInput(fsOrT, caller);
        if (fsOrT.numel() == 1) {
            const keel::f64 fsv = fsOrT.toScalar();
            if (!keel::is_finite(fsv) || fsv <= 0.0)
                throw Error(keel::format("{}: sample rate Fs must be finite and positive", caller),
                            0, 0, caller, "", keel::format("numkit:{}:invalidFs", caller));
        } else {
            const keel::usize N = x.dims().isVector() ? x.numel() : x.dims().rows();
            if (!fsOrT.dims().isVector() || fsOrT.numel() != N)
                throw Error(keel::format("{}: time vector T must match signal length", caller),
                            0, 0, caller, "", keel::format("numkit:{}:invalidTimeVector", caller));
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 ti = fsOrT.elemAsDouble(i);
                if (!keel::is_finite(ti) || (i > 0 && ti <= fsOrT.elemAsDouble(i - 1)))
                    throw Error(keel::format("{}: time vector T must be finite and strictly increasing", caller),
                                0, 0, caller, "", keel::format("numkit:{}:invalidTimeVector", caller));
            }
        }
    }
}

PspectrumOptions clonePspectrumOptions(const PspectrumOptions &inOpts, keel::StringRef overrideType)
{
    PspectrumOptions opts;
    opts.type.assign(overrideType);
    opts.leakage = inOpts.leakage;
    opts.overlapPercent = inOpts.overlapPercent;
    opts.timeResolution = inOpts.timeResolution;
    opts.fmin = inOpts.fmin;
    opts.fmax = inOpts.fmax;
    opts.twoSided = inOpts.twoSided;
    opts.hasTwoSided = inOpts.hasTwoSided;
    return opts;
}

keel::f64 leakageToOverlapRatio(keel::f64 leakage)
{
    static constexpr keel::f64 kTable[21] = {
        0.83872,  0.83390, 0.82868, 0.82290, 0.81662,
        0.80962,  0.80178, 0.79290, 0.78298, 0.77140,
        0.758072, 0.74220, 0.72306, 0.69930, 0.66912,
        0.62876,  0.57164, 0.48362, 0.33120, 0.08660,
        0.0
    };
    const keel::f64 clamped = keel::max(0.0, keel::min(1.0, leakage));
    const keel::f64 pos = clamped * 20.0;
    const keel::usize idx = keel::min<keel::usize>(19, keel::as<keel::usize>( keel::floor(pos) ));
    const keel::f64 frac = pos - keel::as<keel::f64>(idx);
    return kTable[idx] * (1.0 - frac) + kTable[idx + 1] * frac;
}

keel::f64 kaiserEnbwFactor(keel::f64 beta, keel::Allocator *mr)
{
    if (beta <= 0.0) return 1.0;
    constexpr keel::usize kRefLen = 1000;
    Value w = kaiser(kRefLen, beta, mr);
    const keel::f64 *wd = w.doubleData();
    keel::f64 sumW = 0.0, sumW2 = 0.0;
    for (keel::usize i = 0; i < kRefLen; ++i) {
        sumW += wd[i];
        sumW2 += wd[i] * wd[i];
    }
    return (sumW > 0.0) ? (keel::as<keel::f64>(kRefLen) * sumW2 / (sumW * sumW)) : 1.0;
}

keel::Array<keel::f64> normalizedKaiser(keel::usize len, keel::f64 beta, keel::Allocator *mr)
{
    if (len == 0) return keel::Array<keel::f64>(mr);
    if (len == 1) return keel::Array<keel::f64>({1.0}, mr);
    Value w = kaiser(len, beta, mr);
    const keel::f64 *wd = w.doubleData();
    keel::f64 sumW = 0.0;
    for (keel::usize i = 0; i < len; ++i) sumW += wd[i];
    keel::Array<keel::f64> out(len, mr);
    const keel::f64 inv = (sumW > 0.0) ? (1.0 / sumW) : (1.0 / keel::as<keel::f64>(len));
    for (keel::usize i = 0; i < len; ++i) out[i] = wd[i] * inv;
    return out;
}

// Evaluate zoom CZT on a matrix of column segments `segMat` (len x nSegs),
// writing `scale * |X(k, s)|^2` into `powerOut` (nF x nSegs).
void evalSegmentsCztPower(const Value &segMat, keel::usize len, keel::usize nSegs,
                          keel::usize nF, keel::c64 W, keel::c64 A, keel::f64 scale,
                          keel::f64 *powerOut, keel::Allocator *mr)
{
    if (nSegs == 0 || len == 0) return;
    if (len == 1) {
        for (keel::usize s = 0; s < nSegs; ++s) {
            keel::f64 p0 = 0.0;
            if (segMat.isComplex()) {
                p0 = scale * keel::norm(segMat.complexData()[s]);
            } else {
                const keel::f64 v = segMat.doubleData()[s];
                p0 = scale * v * v;
            }
            for (keel::usize k = 0; k < nF; ++k)
                powerOut[s * nF + k] = p0;
        }
        return;
    }
    Value X = czt(segMat, keel::as<keel::i32>(nF), W, A, mr);
    const keel::c64 *xd = X.complexData();
    for (keel::usize s = 0; s < nSegs; ++s) {
        for (keel::usize k = 0; k < nF; ++k) {
            powerOut[s * nF + k] = scale * keel::norm(xd[s * nF + k]);
        }
    }
}

PspectrumResult pspectrumImpl(const Value &x, const Value &fsOrT,
                              const PspectrumOptions &opts, const char *caller,
                              keel::Allocator *mr)
{
    pspectrum_argcheck(x, fsOrT, opts, caller);
    const bool isVec = x.dims().isVector();
    const keel::usize N = isVec ? x.numel() : x.dims().rows();
    const keel::usize nCh = isVec ? 1 : x.dims().cols();
    const bool isComplex = x.isComplex();

    keel::f64 fs = 2.0 * keel::kPi;
    keel::f64 t0 = 0.0;
    const bool isNormFreq = fsOrT.isEmpty();
    if (!isNormFreq) {
        if (fsOrT.numel() == 1) {
            fs = fsOrT.toScalar();
            t0 = 0.0;
        } else {
            const keel::f64 tStart = fsOrT.elemAsDouble(0);
            const keel::f64 tEnd   = fsOrT.elemAsDouble(N - 1);
            const keel::f64 dt     = (tEnd - tStart) / keel::as<keel::f64>(N - 1);
            fs = 1.0 / dt;
            t0 = tStart - 0.5 * dt;
        }
    }

    const keel::f64 beta = 40.0 * (1.0 - opts.leakage);
    const keel::f64 r = (opts.overlapPercent >= 0.0)
        ? (opts.overlapPercent / 100.0)
        : leakageToOverlapRatio(opts.leakage);

    bool twoSided = opts.hasTwoSided ? opts.twoSided : isComplex;
    keel::f64 fmin = twoSided ? (-0.5 * fs) : 0.0;
    keel::f64 fmax = 0.5 * fs;
    if (!keel::is_nan(opts.fmin) && !keel::is_nan(opts.fmax)) {
        if (opts.fmin >= opts.fmax || opts.fmin < -0.5 * fs - 1e-12 || opts.fmax > 0.5 * fs + 1e-12)
            throw Error(keel::format("{}: 'FrequencyLimits' must satisfy [fmin, fmax] within Nyquist interval", caller),
                        0, 0, caller, "", keel::format("numkit:{}:invalidFrequencyLimits", caller));
        fmin = opts.fmin;
        fmax = opts.fmax;
        if (fmin < 0.0) twoSided = true;
    }

    const bool isSpectrogram = (opts.type == "spectrogram");
    const keel::usize nF = isSpectrogram ? 1024 : 4096;
    const keel::f64 scale = twoSided ? 1.0 : 2.0;

    Value F = Value::matrix(nF, 1, ValueType::DOUBLE, mr);
    keel::f64 *fd = F.doubleDataMut();
    for (keel::usize k = 0; k < nF; ++k) {
        fd[k] = fmin + (fmax - fmin) * keel::as<keel::f64>(k) / keel::as<keel::f64>(nF - 1);
    }

    const keel::c64 A = keel::polar(1.0, 2.0 * keel::kPi * fmin / fs);
    const keel::c64 W = keel::polar(1.0, -2.0 * keel::kPi * (fmax - fmin) / (keel::as<keel::f64>(nF - 1) * fs));

    if (isSpectrogram) {
        if (nCh > 1)
            throw Error(keel::format("{}: 'spectrogram' mode requires a 1-D vector input", caller),
                        0, 0, caller, "", keel::format("numkit:{}:expectedVector", caller));
        keel::usize wlen = 0;
        if (opts.timeResolution > 0.0) {
            wlen = keel::min(N, keel::as<keel::usize>(keel::max(1.0, keel::round(opts.timeResolution * fs))));
        } else {
            keel::usize D = 2;
            if (N >= 64) {
                const keel::i32 exp2 = keel::as<keel::i32>(keel::floor(keel::log2(keel::as<keel::f64>(N)))) - 7;
                if (exp2 <= 3)      D = 8;
                else if (exp2 >= 7) D = 128;
                else                D = keel::usize(1) << exp2;
            }
            wlen = (N + D - 1) / D;
        }
        wlen = keel::max<keel::usize>(1, keel::min(N, wlen));
        keel::usize nov = keel::as<keel::usize>(keel::floor(r * keel::as<keel::f64>(wlen)));
        if (nov >= wlen) nov = wlen - 1;
        const keel::usize hop = keel::max<keel::usize>(1, wlen - nov);
        const keel::usize nT = (N > wlen) ? ((N - wlen + hop - 1) / hop + 1) : 1;

        Value P = Value::matrix(nF, nT, ValueType::DOUBLE, mr);
        Value T = Value::matrix(nT, 1, ValueType::DOUBLE, mr);
        keel::f64 *pd = P.doubleDataMut();
        keel::f64 *td = T.doubleDataMut();

        const keel::f64 timeStepScale = isNormFreq ? 1.0 : (1.0 / fs);
        for (keel::usize m = 0; m < nT; ++m) {
            td[m] = t0 + (0.5 * keel::as<keel::f64>(wlen) + keel::as<keel::f64>(m * hop)) * timeStepScale;
        }

        const keel::usize lastStart = (nT - 1) * hop;
        const keel::usize lastLen   = N - lastStart;
        const keel::usize nFull     = (lastLen == wlen) ? nT : (nT - 1);

        if (nFull > 0) {
            const auto wFull = normalizedKaiser(wlen, beta, mr);
            if (isComplex) {
                const keel::c64 *xc = x.complexData();
                Value segMat = Value::complexMatrix(wlen, nFull, mr);
                keel::c64 *sd = segMat.complexDataMut();
                for (keel::usize m = 0; m < nFull; ++m) {
                    const keel::usize s = m * hop;
                    for (keel::usize i = 0; i < wlen; ++i)
                        sd[m * wlen + i] = xc[s + i] * wFull[i];
                }
                evalSegmentsCztPower(segMat, wlen, nFull, nF, W, A, scale, pd, mr);
            } else {
                Value xD = detail::widenSingleToDouble(x, mr);
                const keel::f64 *xd = xD.doubleData();
                Value segMat = Value::matrix(wlen, nFull, ValueType::DOUBLE, mr);
                keel::f64 *sd = segMat.doubleDataMut();
                for (keel::usize m = 0; m < nFull; ++m) {
                    const keel::usize s = m * hop;
                    for (keel::usize i = 0; i < wlen; ++i)
                        sd[m * wlen + i] = xd[s + i] * wFull[i];
                }
                evalSegmentsCztPower(segMat, wlen, nFull, nF, W, A, scale, pd, mr);
            }
        }

        if (nFull < nT) {
            const auto wLast = normalizedKaiser(lastLen, beta, mr);
            if (isComplex) {
                const keel::c64 *xc = x.complexData();
                Value segLast = Value::complexMatrix(lastLen, 1, mr);
                keel::c64 *sd = segLast.complexDataMut();
                for (keel::usize i = 0; i < lastLen; ++i)
                    sd[i] = xc[lastStart + i] * wLast[i];
                evalSegmentsCztPower(segLast, lastLen, 1, nF, W, A, scale, pd + nFull * nF, mr);
            } else {
                Value xD = detail::widenSingleToDouble(x, mr);
                const keel::f64 *xd = xD.doubleData();
                Value segLast = Value::matrix(lastLen, 1, ValueType::DOUBLE, mr);
                keel::f64 *sd = segLast.doubleDataMut();
                for (keel::usize i = 0; i < lastLen; ++i)
                    sd[i] = xd[lastStart + i] * wLast[i];
                evalSegmentsCztPower(segLast, lastLen, 1, nF, W, A, scale, pd + nFull * nF, mr);
            }
        }

        return {keel::move(P), keel::move(F), keel::move(T)};
    }

    // 'power' mode
    keel::usize wlen = 0;
    if (opts.timeResolution > 0.0) {
        wlen = keel::min(N, keel::as<keel::usize>(keel::max(1.0, keel::round(opts.timeResolution * fs))));
    } else {
        const keel::usize target = keel::as<keel::usize>(keel::ceil(1024.0 * kaiserEnbwFactor(beta, mr)));
        wlen = keel::min(N, keel::max<keel::usize>(2, target));
    }
    wlen = keel::max<keel::usize>(1, keel::min(N, wlen));
    keel::usize nov = keel::as<keel::usize>(keel::floor(r * keel::as<keel::f64>(wlen)));
    if (nov >= wlen) nov = wlen - 1;
    const keel::usize hop = keel::max<keel::usize>(1, wlen - nov);
    const keel::usize nT = (N > wlen) ? ((N - wlen + hop - 1) / hop + 1) : 1;

    const auto wFull = normalizedKaiser(wlen, beta, mr);
    keel::Array<keel::usize> starts(nT, 0, mr);
    if (nT > 1) {
        for (keel::usize m = 0; m < nT; ++m) {
            starts[m] = keel::as<keel::usize>(keel::round_i64(
                keel::as<keel::f64>(m) * keel::as<keel::f64>(N - wlen) / keel::as<keel::f64>(nT - 1)));
        }
    }

    Value P = Value::matrix(nF, nCh, ValueType::DOUBLE, mr);
    keel::f64 *pd = P.doubleDataMut();
    keel::Array<keel::f64> segPow(nF * nT, 0.0, mr);

    if (isComplex) {
        const keel::c64 *xc = x.complexData();
        for (keel::usize ch = 0; ch < nCh; ++ch) {
            const keel::c64 *col = xc + ch * N;
            Value segMat = Value::complexMatrix(wlen, nT, mr);
            keel::c64 *sd = segMat.complexDataMut();
            for (keel::usize m = 0; m < nT; ++m) {
                const keel::usize s = starts[m];
                for (keel::usize i = 0; i < wlen; ++i)
                    sd[m * wlen + i] = col[s + i] * wFull[i];
            }
            evalSegmentsCztPower(segMat, wlen, nT, nF, W, A, scale, segPow.data(), mr);
            for (keel::usize k = 0; k < nF; ++k) {
                keel::f64 acc = 0.0;
                for (keel::usize m = 0; m < nT; ++m) acc += segPow[m * nF + k];
                pd[ch * nF + k] = acc / keel::as<keel::f64>(nT);
            }
        }
    } else {
        Value xD = detail::widenSingleToDouble(x, mr);
        const keel::f64 *xd = xD.doubleData();
        for (keel::usize ch = 0; ch < nCh; ++ch) {
            const keel::f64 *col = xd + ch * N;
            Value segMat = Value::matrix(wlen, nT, ValueType::DOUBLE, mr);
            keel::f64 *sd = segMat.doubleDataMut();
            for (keel::usize m = 0; m < nT; ++m) {
                const keel::usize s = starts[m];
                for (keel::usize i = 0; i < wlen; ++i)
                    sd[m * wlen + i] = col[s + i] * wFull[i];
            }
            evalSegmentsCztPower(segMat, wlen, nT, nF, W, A, scale, segPow.data(), mr);
            for (keel::usize k = 0; k < nF; ++k) {
                keel::f64 acc = 0.0;
                for (keel::usize m = 0; m < nT; ++m) acc += segPow[m * nF + k];
                pd[ch * nF + k] = acc / keel::as<keel::f64>(nT);
            }
        }
    }

    return {keel::move(P), keel::move(F), Value::Empty};
}

Value extractColumnValue(const Value &mat, keel::usize colIdx, keel::Allocator *mr)
{
    const keel::usize R = mat.dims().rows();
    if (mat.isComplex()) {
        Value out = Value::complexMatrix(R, 1, mr);
        keel::copy_n(out.complexDataMut(), mat.complexData() + colIdx * R, R);
        return out;
    }
    Value xD = detail::widenSingleToDouble(mat, mr);
    Value out = Value::matrix(R, 1, ValueType::DOUBLE, mr);
    keel::copy_n(out.doubleDataMut(), xD.doubleData() + colIdx * R, R);
    return out;
}

} // anonymous

PspectrumResult pspectrum(const Value &x, const Value &fsOrT,
                          const PspectrumOptions &opts,
                          keel::Allocator *mr)
{
    return pspectrumImpl(x, fsOrT, opts, "pspectrum", mr);
}

keel::Tuple<Value, Value>
instfreqHilbert(const Value &x, const Value &fs, keel::Allocator *mr)
{
    detail::requireFloatInput(x, "instfreq");
    keel::f64 fsv = 1.0;
    keel::f64 t0 = 0.0;
    if (!fs.isEmpty()) {
        detail::requireRealFloatInput(fs, "instfreq");
        if (fs.numel() == 1) {
            fsv = fs.toScalar();
            if (!keel::is_finite(fsv) || fsv <= 0.0)
                throw Error("instfreq: sample rate Fs must be finite and positive",
                            0, 0, "instfreq", "", "numkit:instfreq:invalidFs");
        } else {
            const keel::usize N = x.dims().isVector() ? x.numel() : x.dims().rows();
            if (!fs.dims().isVector() || fs.numel() != N)
                throw Error("instfreq: time vector T must match signal length",
                            0, 0, "instfreq", "", "numkit:instfreq:invalidTimeVector");
            const keel::f64 tStart = fs.elemAsDouble(0);
            const keel::f64 tEnd   = fs.elemAsDouble(N - 1);
            const keel::f64 dt     = (tEnd - tStart) / keel::as<keel::f64>(N - 1);
            fsv = 1.0 / dt;
            t0 = tStart;
        }
    }

    Value z = hilbert(x, mr);
    keel::Array<keel::f64> re(mr), im(mr);
    readComplex(z, re, im);
    const keel::usize n = re.size();
    if (n < 2) {
        return {Value::matrix(0, 1, ValueType::DOUBLE, mr),
                Value::matrix(0, 1, ValueType::DOUBLE, mr)};
    }

    auto out = Value::matrix(n - 1, 1, ValueType::DOUBLE, mr);
    auto tOut = Value::matrix(n - 1, 1, ValueType::DOUBLE, mr);
    keel::f64 *d = out.doubleDataMut();
    keel::f64 *td = tOut.doubleDataMut();
    constexpr keel::f64 kPi = keel::kPi;
    for (keel::usize i = 0; i + 1 < n; ++i) {
        const keel::f64 p1 = keel::atan2(im[i + 1], re[i + 1]);
        const keel::f64 p0 = keel::atan2(im[i], re[i]);
        keel::f64 dp = p1 - p0;
        while (dp >  kPi) dp -= 2.0 * kPi;
        while (dp <= -kPi) dp += 2.0 * kPi;
        d[i] = dp * fsv / (2.0 * kPi);
        td[i] = t0 + (keel::as<keel::f64>(i) + 0.5) / fsv;
    }
    return {keel::move(out), keel::move(tOut)};
}

keel::Tuple<Value, Value>
instfreqWithTime(const Value &x, const Value &fs,
                 const PspectrumOptions &inOpts,
                 keel::Allocator *mr)
{
    PspectrumOptions opts = clonePspectrumOptions(inOpts, "spectrogram");
    const Value effFs = fs.isEmpty() ? Value::scalar(1.0, mr) : fs;

    const bool isVec = x.dims().isVector();
    const keel::usize nCh = isVec ? 1 : x.dims().cols();

    if (nCh == 1) {
        auto spec = pspectrumImpl(x, effFs, opts, "instfreq", mr);
        const keel::usize nF = spec.p.dims().rows();
        const keel::usize nT = spec.p.dims().cols();
        const keel::f64 *pd = spec.p.doubleData();
        const keel::f64 *fd = spec.f.doubleData();

        Value ifr = Value::matrix(nT, 1, ValueType::DOUBLE, mr);
        keel::f64 *ifd = ifr.doubleDataMut();
        for (keel::usize m = 0; m < nT; ++m) {
            const keel::f64 *col = pd + m * nF;
            keel::f64 num = 0.0, den = 0.0;
            for (keel::usize k = 0; k < nF; ++k) {
                num += fd[k] * col[k];
                den += col[k];
            }
            ifd[m] = (den > 0.0) ? (num / den) : 0.0;
        }
        return {keel::move(ifr), keel::move(spec.t)};
    }

    Value ifrMat;
    Value tVec;
    for (keel::usize ch = 0; ch < nCh; ++ch) {
        Value colVal = extractColumnValue(x, ch, mr);
        auto spec = pspectrumImpl(colVal, effFs, opts, "instfreq", mr);
        const keel::usize nF = spec.p.dims().rows();
        const keel::usize nT = spec.p.dims().cols();
        if (ch == 0) {
            ifrMat = Value::matrix(nT, nCh, ValueType::DOUBLE, mr);
            tVec = keel::move(spec.t);
        }
        const keel::f64 *pd = spec.p.doubleData();
        const keel::f64 *fd = spec.f.doubleData();
        keel::f64 *ifd = ifrMat.doubleDataMut() + ch * nT;
        for (keel::usize m = 0; m < nT; ++m) {
            const keel::f64 *col = pd + m * nF;
            keel::f64 num = 0.0, den = 0.0;
            for (keel::usize k = 0; k < nF; ++k) {
                num += fd[k] * col[k];
                den += col[k];
            }
            ifd[m] = (den > 0.0) ? (num / den) : 0.0;
        }
    }
    return {keel::move(ifrMat), keel::move(tVec)};
}

Value instfreq(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto [ifr, t] = instfreqWithTime(x, fs, PspectrumOptions{}, mr);
    (void)t;
    return ifr;
}

keel::Tuple<Value, Value>
instbwWithTime(const Value &x, const Value &fs,
               const PspectrumOptions &inOpts,
               keel::Allocator *mr)
{
    PspectrumOptions opts = clonePspectrumOptions(inOpts, "spectrogram");
    const Value effFs = fs.isEmpty() ? Value::scalar(1.0, mr) : fs;

    const bool isVec = x.dims().isVector();
    const keel::usize nCh = isVec ? 1 : x.dims().cols();
    const keel::f64 scale = 2.0 * keel::sqrt(keel::kPi);

    if (nCh == 1) {
        auto spec = pspectrumImpl(x, effFs, opts, "instbw", mr);
        const keel::usize nF = spec.p.dims().rows();
        const keel::usize nT = spec.p.dims().cols();
        const keel::f64 *pd = spec.p.doubleData();
        const keel::f64 *fd = spec.f.doubleData();

        Value ibw = Value::matrix(nT, 1, ValueType::DOUBLE, mr);
        keel::f64 *ibd = ibw.doubleDataMut();
        for (keel::usize m = 0; m < nT; ++m) {
            const keel::f64 *col = pd + m * nF;
            keel::f64 num = 0.0, den = 0.0;
            for (keel::usize k = 0; k < nF; ++k) {
                num += fd[k] * col[k];
                den += col[k];
            }
            if (!(den > 0.0)) {
                ibd[m] = 0.0;
                continue;
            }
            const keel::f64 mu = num / den;
            keel::f64 varNum = 0.0;
            for (keel::usize k = 0; k < nF; ++k) {
                const keel::f64 df = fd[k] - mu;
                varNum += col[k] * df * df;
            }
            ibd[m] = scale * keel::sqrt(keel::max(0.0, varNum / den));
        }
        return {keel::move(ibw), keel::move(spec.t)};
    }

    Value ibwMat;
    Value tVec;
    for (keel::usize ch = 0; ch < nCh; ++ch) {
        Value colVal = extractColumnValue(x, ch, mr);
        auto spec = pspectrumImpl(colVal, effFs, opts, "instbw", mr);
        const keel::usize nF = spec.p.dims().rows();
        const keel::usize nT = spec.p.dims().cols();
        if (ch == 0) {
            ibwMat = Value::matrix(nT, nCh, ValueType::DOUBLE, mr);
            tVec = keel::move(spec.t);
        }
        const keel::f64 *pd = spec.p.doubleData();
        const keel::f64 *fd = spec.f.doubleData();
        keel::f64 *ibd = ibwMat.doubleDataMut() + ch * nT;
        for (keel::usize m = 0; m < nT; ++m) {
            const keel::f64 *col = pd + m * nF;
            keel::f64 num = 0.0, den = 0.0;
            for (keel::usize k = 0; k < nF; ++k) {
                num += fd[k] * col[k];
                den += col[k];
            }
            if (!(den > 0.0)) {
                ibd[m] = 0.0;
                continue;
            }
            const keel::f64 mu = num / den;
            keel::f64 varNum = 0.0;
            for (keel::usize k = 0; k < nF; ++k) {
                const keel::f64 df = fd[k] - mu;
                varNum += col[k] * df * df;
            }
            ibd[m] = scale * keel::sqrt(keel::max(0.0, varNum / den));
        }
    }
    return {keel::move(ibwMat), keel::move(tVec)};
}

Value instbw(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto [ibw, t] = instbwWithTime(x, fs, PspectrumOptions{}, mr);
    (void)t;
    return ibw;
}

} // namespace numkit::signal
