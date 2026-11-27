// toolboxes/signal/src/spectral_analysis/spectral_metrics.cpp
//
// Spectral measurement functions, harmonic metrics, pspectrum, instfreq/instbw
// (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/spectral_analysis/spectral_metrics.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

keel::f64 scalarOr(const Value &v, keel::f64 dflt)
{
    return v.isEmpty() ? dflt : v.toScalar();
}

keel::Array<keel::f64> readRealVec(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> out(n, mr);
    for (keel::usize i = 0; i < n; ++i) out[i] = x.elemAsDouble(i);
    return out;
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

keel::signal::PspectrumOptions toKeelPspecOpts(const Value &x, const Value &fsOrT,
                                               const PspectrumOptions &opts, const char *caller)
{
    const bool isVec = x.dims().isVector();
    const keel::usize N = isVec ? x.numel() : x.dims().rows();

    keel::signal::PspectrumOptions kopts;
    kopts.mode = (opts.type == "spectrogram") ? keel::signal::PspectrumMode::kSpectrogram
                                              : keel::signal::PspectrumMode::kPower;
    kopts.leakage = opts.leakage;
    kopts.overlap_percent = opts.overlapPercent;
    kopts.time_resolution = opts.timeResolution;
    kopts.fmin = opts.fmin;
    kopts.fmax = opts.fmax;
    kopts.two_sided = opts.twoSided;
    kopts.has_two_sided = opts.hasTwoSided;
    kopts.is_norm_freq = fsOrT.isEmpty();
    kopts.fs = keel::kTau;
    kopts.t0 = 0.0;

    if (!kopts.is_norm_freq) {
        if (fsOrT.numel() == 1) {
            kopts.fs = fsOrT.toScalar();
            kopts.t0 = 0.0;
        } else {
            const keel::f64 tStart = fsOrT.elemAsDouble(0);
            const keel::f64 tEnd = fsOrT.elemAsDouble(N - 1);
            const keel::f64 dt = (tEnd - tStart) / keel::as<keel::f64>(N - 1);
            kopts.fs = 1.0 / dt;
            kopts.t0 = tStart - 0.5 * dt;
        }
    }

    if (!keel::is_nan(opts.fmin) && !keel::is_nan(opts.fmax)) {
        if (opts.fmin >= opts.fmax || opts.fmin < -0.5 * kopts.fs - 1e-12 ||
            opts.fmax > 0.5 * kopts.fs + 1e-12)
            throw Error(keel::format("{}: 'FrequencyLimits' must satisfy [fmin, fmax] within Nyquist interval", caller),
                        0, 0, caller, "", keel::format("numkit:{}:invalidFrequencyLimits", caller));
    }
    return kopts;
}

PspectrumResult pspectrumImpl(const Value &x, const Value &fsOrT,
                              const PspectrumOptions &opts, const char *caller,
                              keel::Allocator *mr)
{
    pspectrum_argcheck(x, fsOrT, opts, caller);
    const bool isVec = x.dims().isVector();
    const keel::usize N = isVec ? x.numel() : x.dims().rows();
    const keel::usize nCh = isVec ? 1 : x.dims().cols();
    if (opts.type == "spectrogram" && nCh > 1)
        throw Error(keel::format("{}: 'spectrogram' mode requires a 1-D vector input", caller),
                    0, 0, caller, "", keel::format("numkit:{}:expectedVector", caller));

    const keel::signal::PspectrumOptions kopts = toKeelPspecOpts(x, fsOrT, opts, caller);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::signal::PspectrumResult kres;
    if (x.isComplex()) {
        (void)keel::signal::pspectrum_c64(
            keel::ArrayRef<keel::c64>(x.complexData(), N * nCh), N, nCh, kopts, kres,
            scratch.allocator());
    } else {
        auto xv = readRealVec(x, scratch.allocator());
        (void)keel::signal::pspectrum_real(xv.as_ref(), N, nCh, kopts, kres, scratch.allocator());
    }

    Value P = Value::matrix(kres.n_freq, kres.n_cols, ValueType::DOUBLE, mr);
    if (!kres.p_col_major.empty())
        keel::copy_n(P.doubleDataMut(), kres.p_col_major.data(), kres.p_col_major.size());
    Value F = Value::matrix(kres.n_freq, 1, ValueType::DOUBLE, mr);
    if (!kres.f.empty())
        keel::copy_n(F.doubleDataMut(), kres.f.data(), kres.f.size());
    if (kres.t.empty()) {
        return {keel::move(P), keel::move(F), Value::Empty};
    }
    Value T = Value::matrix(kres.t.size(), 1, ValueType::DOUBLE, mr);
    keel::copy_n(T.doubleDataMut(), kres.t.data(), kres.t.size());
    return {keel::move(P), keel::move(F), keel::move(T)};
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

Value bandpower(const Value &x, const Value &fs, const Value &freqrange, keel::Allocator *mr)
{
    const bool xSingle = (x.type() == ValueType::SINGLE);
    detail::requireFloatInput(x, "bandpower");
    if (fs.isEmpty()) {
        const keel::usize n = x.numel();
        if (n == 0) return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), xSingle, mr);
        keel::f64 s = 0.0;
        if (x.isComplex()) {
            const keel::c64 *xd = x.complexData();
            for (keel::usize i = 0; i < n; ++i) s += keel::norm(xd[i]);
        } else {
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 v = x.elemAsDouble(i);
                s += v * v;
            }
        }
        return detail::narrowDoubleToSingleIf(Value::scalar(s / keel::as<keel::f64>(n), mr), xSingle, mr);
    }
    if (freqrange.isEmpty())
        throw Error("bandpower: fs requires freqrange — specify both together (bandpower(x, fs, freqrange))",
                    0, 0, "bandpower", "", "numkit:bandpower:freqRangeRequired");
    if (freqrange.numel() != 2)
        throw Error(keel::format("bandpower: freqrange must be a 2-element vector [fmin, fmax] (got {} elements)",
                                 freqrange.numel()),
                    0, 0, "bandpower", "", "numkit:bandpower:freqRange");
    if (fs.numel() != 1 || fs.isComplex())
        throw Error("bandpower: fs must be a real scalar",
                    0, 0, "bandpower", "", "numkit:bandpower:fs");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::f64 pwr = 0.0;
    const keel::f64 fsv = fs.toScalar();
    const keel::f64 fLo = freqrange.elemAsDouble(0);
    const keel::f64 fHi = freqrange.elemAsDouble(1);
    if (x.isComplex()) {
        const keel::usize n = x.numel();
        const keel::f64 effFs = (fsv > 0.0) ? fsv : keel::kTau;
        keel::Array<keel::f64> pxx(n, 0.0, scratch.allocator());
        keel::Array<keel::f64> fv(n, 0.0, scratch.allocator());
        (void)keel::signal::periodogram_c64(
            keel::ArrayRef<keel::c64>(x.complexData(), n), {}, n, effFs,
            pxx.as_mut_ref(), fv.as_mut_ref(), scratch.allocator());
        (void)keel::signal::bandpower_psd(pxx.as_ref(), fv.as_ref(), fLo, fHi, true, pwr);
    } else {
        auto xv = readRealVec(x, scratch.allocator());
        (void)keel::signal::bandpower_time(xv.as_ref(), fsv, fLo, fHi, pwr, scratch.allocator());
    }
    return detail::narrowDoubleToSingleIf(Value::scalar(pwr, mr), xSingle, mr);
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
    if (!freqrange.isEmpty() && freqrange.numel() != 2)
        throw Error(keel::format("bandpower: freqrange must be a 2-element vector [fmin, fmax] (got {} elements)",
                                 freqrange.numel()),
                    0, 0, "bandpower", "", "numkit:bandpower:freqRange");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto pv = readRealVec(pxx, scratch.allocator());
    auto fv = readRealVec(f, scratch.allocator());
    const bool useRange = !freqrange.isEmpty();
    const keel::f64 fLo = useRange ? freqrange.elemAsDouble(0) : 0.0;
    const keel::f64 fHi = useRange ? freqrange.elemAsDouble(1) : 0.0;
    keel::f64 pwr = 0.0;
    (void)keel::signal::bandpower_psd(pv.as_ref(), fv.as_ref(), fLo, fHi, useRange, pwr);
    return detail::narrowDoubleToSingleIf(Value::scalar(pwr, mr), anySingle, mr);
}

Value meanfreq(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::meanfreq_time(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value medfreq(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::medfreq_time(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value enbw(const Value &window, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto wv = readRealVec(window, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::enbw(wv.as_ref(), scalarOr(fs, 0.0), out);
    return Value::scalar(out, mr);
}

keel::Tuple<Value, Value, Value, Value>
obw(const Value &x, const Value &fs, keel::f64 p, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::signal::ObwResult res;
    (void)keel::signal::obw_time(xv.as_ref(), scalarOr(fs, 0.0), p, res, scratch.allocator());
    return keel::make_tuple(Value::scalar(res.bw, mr),
                            Value::scalar(res.flo, mr),
                            Value::scalar(res.fhi, mr),
                            Value::scalar(res.power, mr));
}

Value powerbw(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::powerbw_time(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value spectralcrest(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::spectral_crest(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value spectralflatness(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::spectral_flatness(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value spectralentropy(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::spectral_entropy(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value spectralkurtosis(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::spectral_kurtosis(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value spectralskewness(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::spectral_skewness(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value snr(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::snr_time(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value sinad(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::sinad_time(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value thd(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::thd_time(xv.as_ref(), scalarOr(fs, 0.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

Value sfdr(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    keel::f64 out = 0.0;
    (void)keel::signal::sfdr_time(xv.as_ref(), scalarOr(fs, 1.0), out, scratch.allocator());
    return Value::scalar(out, mr);
}

keel::Tuple<Value, Value, Value>
sfdrFull(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto *sa = scratch.allocator();
    auto xv = readRealVec(x, sa);
    const keel::usize n = xv.size();
    if (n < 2) {
        const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
        return {Value::scalar(nan, mr), Value::scalar(nan, mr), Value::scalar(nan, mr)};
    }
    const keel::f64 fsIn = scalarOr(fs, 1.0);
    const keel::f64 fsEff = (fsIn > 0.0) ? fsIn : 1.0;

    keel::Array<keel::f64> w(n, 0.0, sa);
    keel::signal::kaiser(w.as_mut_ref(), 38.0);
    keel::f64 sumW = 0.0;
    keel::f64 sumW2 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        sumW += w[i];
        sumW2 += w[i] * w[i];
    }
    const keel::f64 rbw = (sumW > 0.0) ? (fsEff * sumW2 / (sumW * sumW)) : 0.0;

    const keel::usize nBins = n / 2 + 1;
    const keel::f64 df = fsEff / keel::as<keel::f64>(n);
    keel::Array<keel::f64> pxx(nBins, 0.0, sa);
    keel::Array<keel::f64> f(nBins, 0.0, sa);
    (void)keel::signal::periodogram(xv.as_ref(), w.as_ref(), n, fsEff,
                                    pxx.as_mut_ref(), f.as_mut_ref(), sa);

    auto getTone = [&](keel::i32 idxTone, keel::i32 &outL, keel::i32 &outR, keel::f64 &outFreq) noexcept -> keel::f64 {
        keel::i32 iL = idxTone - 1;
        keel::i32 iR = idxTone + 1;
        while (iL >= 0 && pxx[keel::as<keel::usize>(iL)] <= pxx[keel::as<keel::usize>(iL + 1)]) --iL;
        while (iR < keel::as<keel::i32>(nBins) && pxx[keel::as<keel::usize>(iR - 1)] >= pxx[keel::as<keel::usize>(iR)]) ++iR;
        ++iL;
        --iR;
        outL = iL;
        outR = iR;
        keel::f64 power = 0.0;
        keel::f64 fNum = 0.0;
        if (iL < iR) {
            const keel::i32 len = iR - iL + 1;
            for (keel::i32 k = 0; k < len; ++k) {
                keel::f64 wBin = 0.0;
                if (k == 0) {
                    wBin = f[keel::as<keel::usize>(iL + 1)] - f[keel::as<keel::usize>(iL)];
                } else if (k + 1 == len) {
                    wBin = f[keel::as<keel::usize>(iR)] - f[keel::as<keel::usize>(iR - 1)];
                } else {
                    wBin = 0.5 * (f[keel::as<keel::usize>(iL + k + 1)] - f[keel::as<keel::usize>(iL + k - 1)]);
                }
                const keel::f64 dp = pxx[keel::as<keel::usize>(iL + k)] * wBin;
                power += dp;
                fNum += f[keel::as<keel::usize>(iL + k)] * dp;
            }
        } else {
            power = pxx[keel::as<keel::usize>(iR)] * df;
            fNum = f[keel::as<keel::usize>(iR)] * power;
        }
        outFreq = (power > 0.0) ? (fNum / power) : f[keel::as<keel::usize>(idxTone)];
        if (power < rbw * pxx[keel::as<keel::usize>(idxTone)]) {
            power = rbw * pxx[keel::as<keel::usize>(idxTone)];
        }
        return power;
    };

    auto argmaxPxx = [&]() noexcept -> keel::i32 {
        keel::i32 best = 0;
        for (keel::usize i = 1; i < nBins; ++i) {
            if (pxx[i] > pxx[keel::as<keel::usize>(best)]) best = keel::as<keel::i32>(i);
        }
        return best;
    };

    pxx[0] *= 2.0;
    keel::i32 dcL = 0, dcR = 0;
    keel::f64 dummyF = 0.0;
    (void)getTone(0, dcL, dcR, dummyF);
    for (keel::i32 k = dcL; k <= dcR; ++k) pxx[keel::as<keel::usize>(k)] = 0.0;

    const keel::i32 iFund = argmaxPxx();
    keel::i32 fL = 0, fR = 0;
    const keel::f64 pFund = getTone(iFund, fL, fR, dummyF);
    for (keel::i32 k = fL; k <= fR; ++k) pxx[keel::as<keel::usize>(k)] = 0.0;

    const keel::i32 spurBin = argmaxPxx();
    const keel::i32 iLeftBin = keel::max(keel::i32{0}, spurBin - 1);
    const keel::i32 iRightBin = keel::min(keel::as<keel::i32>(nBins) - 1, spurBin + 1);
    keel::i32 iSpur = iLeftBin;
    for (keel::i32 k = iLeftBin + 1; k <= iRightBin; ++k) {
        if (pxx[keel::as<keel::usize>(k)] > pxx[keel::as<keel::usize>(iSpur)]) iSpur = k;
    }
    keel::i32 sL = 0, sR = 0;
    keel::f64 fSpur = 0.0;
    const keel::f64 pSpur = getTone(iSpur, sL, sR, fSpur);

    const keel::f64 inf = keel::Limits<keel::f64>::infinity();
    const keel::f64 rDb = (pFund <= 0.0) ? -inf
                        : (pSpur <= 0.0) ? inf
                        : 10.0 * keel::log10(pFund / pSpur);
    const keel::f64 spDb = (pSpur <= 0.0) ? -inf : 10.0 * keel::log10(pSpur);
    return {Value::scalar(rDb, mr), Value::scalar(spDb, mr), Value::scalar(fSpur, mr)};
}

PspectrumResult pspectrum(const Value &x, const Value &fsOrT,
                          const PspectrumOptions &opts, keel::Allocator *mr)
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

    const keel::usize n = x.numel();
    if (n < 2) {
        return {Value::matrix(0, 1, ValueType::DOUBLE, mr),
                Value::matrix(0, 1, ValueType::DOUBLE, mr)};
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readRealVec(x, scratch.allocator());
    auto out = Value::matrix(n - 1, 1, ValueType::DOUBLE, mr);
    auto tOut = Value::matrix(n - 1, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::instfreq_hilbert(
        xv.as_ref(), fsv, t0,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n - 1),
        keel::MutArrayRef<keel::f64>(tOut.doubleDataMut(), n - 1),
        scratch.allocator());
    return {keel::move(out), keel::move(tOut)};
}

keel::Tuple<Value, Value>
instfreqWithTime(const Value &x, const Value &fs,
                 const PspectrumOptions &inOpts, keel::Allocator *mr)
{
    PspectrumOptions opts = clonePspectrumOptions(inOpts, "spectrogram");
    const Value effFs = fs.isEmpty() ? Value::scalar(1.0, mr) : fs;
    const bool isVec = x.dims().isVector();
    const keel::usize nCh = isVec ? 1 : x.dims().cols();

    if (nCh == 1) {
        auto spec = pspectrumImpl(x, effFs, opts, "instfreq", mr);
        const keel::usize nF = spec.p.dims().rows();
        const keel::usize nT = spec.p.dims().cols();
        Value ifr = Value::matrix(nT, 1, ValueType::DOUBLE, mr);
        (void)keel::signal::instfreq_from_spec(
            keel::ArrayRef<keel::f64>(spec.p.doubleData(), nF * nT), nF, nT,
            keel::ArrayRef<keel::f64>(spec.f.doubleData(), nF),
            keel::MutArrayRef<keel::f64>(ifr.doubleDataMut(), nT));
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
        (void)keel::signal::instfreq_from_spec(
            keel::ArrayRef<keel::f64>(spec.p.doubleData(), nF * nT), nF, nT,
            keel::ArrayRef<keel::f64>(spec.f.doubleData(), nF),
            keel::MutArrayRef<keel::f64>(ifrMat.doubleDataMut() + ch * nT, nT));
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
               const PspectrumOptions &inOpts, keel::Allocator *mr)
{
    PspectrumOptions opts = clonePspectrumOptions(inOpts, "spectrogram");
    const Value effFs = fs.isEmpty() ? Value::scalar(1.0, mr) : fs;
    const bool isVec = x.dims().isVector();
    const keel::usize nCh = isVec ? 1 : x.dims().cols();

    if (nCh == 1) {
        auto spec = pspectrumImpl(x, effFs, opts, "instbw", mr);
        const keel::usize nF = spec.p.dims().rows();
        const keel::usize nT = spec.p.dims().cols();
        Value ibw = Value::matrix(nT, 1, ValueType::DOUBLE, mr);
        (void)keel::signal::instbw_from_spec(
            keel::ArrayRef<keel::f64>(spec.p.doubleData(), nF * nT), nF, nT,
            keel::ArrayRef<keel::f64>(spec.f.doubleData(), nF),
            keel::MutArrayRef<keel::f64>(ibw.doubleDataMut(), nT));
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
        (void)keel::signal::instbw_from_spec(
            keel::ArrayRef<keel::f64>(spec.p.doubleData(), nF * nT), nF, nT,
            keel::ArrayRef<keel::f64>(spec.f.doubleData(), nF),
            keel::MutArrayRef<keel::f64>(ibwMat.doubleDataMut() + ch * nT, nT));
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
