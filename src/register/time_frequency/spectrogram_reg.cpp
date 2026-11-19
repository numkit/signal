// toolboxes/signal/src/time_frequency/spectrogram_reg.cpp
//
// CallContext register half of time_frequency/spectrogram.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.

// The reg wrappers build Hamming/Hann windows + the radian frequency axis with
// M_PI â€” define it before any <cmath> include (MSVC needs _USE_MATH_DEFINES).
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/string.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <numkit/sys/plugin.hpp>
#include <numkit/signal/time_frequency/spectrogram.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

#include "../../signal_detail.hpp"

namespace numkit::signal {

namespace detail {

void iscola_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "iscola");
    if (args[1].numel() != 1)
        throw Error("iscola: noverlap must be a scalar",
                     0, 0, "iscola", "", "numkit:iscola:badOverlap");
    const double novS = args[1].toScalar();
    if (!(novS >= 0.0))
        throw Error("iscola: noverlap must be non-negative",
                     0, 0, "iscola", "", "numkit:iscola:badOverlap");
    const keel::usize noverlap = static_cast<keel::usize>(novS);

    keel::String method = keel::String::from("wola");  // MATLAB default
    if (args.size() >= 3) {
        if (!(args[2].isChar() || args[2].isString()))
            throw Error("iscola: method must be a string",
                         0, 0, "iscola", "", "numkit:iscola:badMethod");
        method = args[2].toString();
    }

    auto [tf, m, dev] = iscola(args[0], noverlap, method,
                                ctx.engine->allocator());
    outs[0] = std::move(tf);
    if (nargout > 1) outs[1] = std::move(m);
    if (nargout > 2) outs[2] = std::move(dev);
}

void spectrogram_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "spectrogram");  // no max: name-value tail in the reference

    // MATLAB's spectrogram(x, N) accepts a scalar-N (window length) and
    // builds a Hamming window of that length. If arg 1 is a vector, it's
    // the explicit window. Adapter handles both by synthesizing a Hamming
    // vector here when it sees a scalar.
    Value window = Value();
    if (args.size() >= 2 && !args[1].isChar()) {
        if (args[1].numel() == 1) {
            const keel::usize winLen = static_cast<keel::usize>(argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "window", "spectrogram"), 2, "window", "spectrogram"));
            auto w = Value::matrix(1, winLen, ValueType::DOUBLE, ctx.engine->allocator());
            if (winLen == 1) {
                w.doubleDataMut()[0] = 1.0;
            } else {
                for (keel::usize i = 0; i < winLen; ++i)
                    w.doubleDataMut()[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (winLen - 1));
            }
            window = std::move(w);
        } else {
            window = args[1];
        }
    }
    const keel::usize noverlap = (args.size() >= 3 && !args[2].isEmpty())
        ? static_cast<keel::usize>(argcheck::expectedInteger(argcheck::expectedReal(args[2], 3, "noverlap", "spectrogram"), 3, "noverlap", "spectrogram")) : 0;
    const keel::usize nfft = (args.size() >= 4 && !args[3].isEmpty())
        ? static_cast<keel::usize>(argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "nfft", "spectrogram"), 4, "nfft", "spectrogram")) : 0;

    auto [S, F, T] = spectrogram(args[0], window, noverlap, nfft, ctx.engine->allocator());

    // Sample-rate scaling of the frequency / time axes. MATLAB's
    // spectrogram returns f in Hz = k*fs/nfft and t in seconds = centre/fs
    // when fs is supplied (5th positional arg). With no fs the normalized
    // convention uses fs = 2*pi, so f spans [0, pi] and t = centre/(2*pi).
    // The core builds the normalized axes (f = k*2*pi/nfft, t = centre in
    // samples); rescale both here so f and t honour fs.
    double fs = 2.0 * M_PI;
    if (args.size() >= 5 && !args[4].isEmpty())
        fs = args[4].toScalar();
    {
        const keel::usize nFreqs  = F.numel();
        const keel::usize nfftEff = (nFreqs > 1) ? (nFreqs - 1) * 2 : 1;
        double *fd = F.doubleDataMut();
        for (keel::usize i = 0; i < nFreqs; ++i)
            fd[i] = static_cast<double>(i) * fs / static_cast<double>(nfftEff);
        double *td = T.doubleDataMut();
        for (keel::usize j = 0; j < T.numel(); ++j)
            td[j] /= fs;
    }

    // 4th output / Auto-plot: ps — one-sided power spectral density per (freq, time),
    // ps = c[k]·|S|² / (fs·Σwin²), with c = 2 on interior bins and 1 at DC
    // and (when present) Nyquist. Reuses the already-computed STFT S.
    // 5th & 6th outputs: fc (center of gravity frequency) and tc (center of
    // gravity time) via time-reassignment using the time-ramped window wt and
    // not-a-knot cubic-spline derivative window wd.
    Value PS, FC, TC;
    if (nargout == 0 || nargout > 3) {
        const keel::usize R = S.dims().rows();
        const keel::usize C = S.dims().cols();
        const keel::usize nx = args[0].numel();
        auto *mr = ctx.engine->allocator();
        // Window coefficients — replicate the core's default (hamming of
        // length floor(nx/4.5)) when none was supplied; keep in sync with
        // resolveWindow() in this TU.
        keel::Array<double> win(mr);
        if (!window.isEmpty() && window.numel() > 0) {
            const keel::usize L = window.numel();
            win.resize(L);
            for (keel::usize i = 0; i < L; ++i) win[i] = window.elemAsDouble(i);
        } else {
            keel::usize L = std::max<keel::usize>(
                1, static_cast<keel::usize>(std::floor(static_cast<double>(nx) / 4.5)));
            if (L > nx && nx > 0) L = nx;
            win.resize(L);
            if (L == 1) win[0] = 1.0;
            else for (keel::usize i = 0; i < L; ++i)
                win[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (L - 1));
        }
        double U = 0.0;
        for (double w : win) U += w * w;
        const keel::usize nfftEff = (R > 1) ? (R - 1) * 2 : 1;
        const bool nyquist = (nfftEff % 2 == 0);   // last bin is Nyquist
        const double denom = fs * U;
        PS = Value::matrix(R, C, ValueType::DOUBLE, mr);
        const keel::c64 *sd = S.complexData();
        double *pd = PS.doubleDataMut();
        for (keel::usize k = 0; k < R; ++k) {
            const double c = (k == 0 || (nyquist && k == R - 1)) ? 1.0 : 2.0;
            for (keel::usize m = 0; m < C; ++m) {
                const keel::c64 v = sd[m * R + k];
                pd[m * R + k] = c * keel::norm(v) / denom;
            }
        }

        if (nargout > 4) {
            const keel::usize L = win.size();
            keel::Array<double> wt(L, mr);
            keel::Array<double> wd(L, mr);
            const double halfL = 0.5 * static_cast<double>(L > 0 ? L - 1 : 0);
            for (keel::usize i = 0; i < L; ++i)
                wt[i] = (static_cast<double>(i) - halfL) * win[i];

            if (L <= 1) {
                if (L == 1) wd[0] = 0.0;
            } else if (L == 2) {
                wd[0] = wd[1] = win[1] - win[0];
            } else {
                // Not-a-knot cubic spline nodal first derivatives on 0..L-1.
                keel::Array<double> sub(L, 1.0, mr);
                keel::Array<double> diag(L, 4.0, mr);
                keel::Array<double> sup(L, 1.0, mr);
                keel::Array<double> rhs(L, 0.0, mr);
                diag[0] = 1.0;
                sup[0]  = 2.0;
                rhs[0]  = -2.5 * win[0] + 2.0 * win[1] + 0.5 * win[2];
                for (keel::usize i = 1; i + 1 < L; ++i)
                    rhs[i] = 3.0 * (win[i + 1] - win[i - 1]);
                sub[L - 1]  = 2.0;
                diag[L - 1] = 1.0;
                rhs[L - 1]  = 2.5 * win[L - 1] - 2.0 * win[L - 2] - 0.5 * win[L - 3];

                for (keel::usize i = 1; i < L; ++i) {
                    const double factor = sub[i] / diag[i - 1];
                    diag[i] -= factor * sup[i - 1];
                    rhs[i]  -= factor * rhs[i - 1];
                }
                wd[L - 1] = rhs[L - 1] / diag[L - 1];
                for (keel::usize i = L - 1; i-- > 0;)
                    wd[i] = (rhs[i] - sup[i] * wd[i + 1]) / diag[i];
            }

            FC = Value::matrix(R, C, ValueType::DOUBLE, mr);
            TC = Value::matrix(R, C, ValueType::DOUBLE, mr);
            double *fcd = FC.doubleDataMut();
            double *tcd = TC.doubleDataMut();
            const double *fd = F.doubleData();
            const double *td = T.doubleData();
            const double *xd = args[0].doubleData();
            const keel::usize novEff = (noverlap == 0) ? (L / 2) : noverlap;
            const keel::usize step   = (L > novEff) ? (L - novEff) : 1;

            Value frameT = Value::matrix(nfftEff, 1, ValueType::DOUBLE, mr);
            Value frameD = Value::matrix(nfftEff, 1, ValueType::DOUBLE, mr);
            double *fTd = frameT.doubleDataMut();
            double *fDd = frameD.doubleDataMut();

            for (keel::usize m = 0; m < C; ++m) {
                const keel::usize start = m * step;
                for (keel::usize i = 0; i < L; ++i) {
                    const double xv = xd[start + i];
                    fTd[i] = xv * wt[i];
                    fDd[i] = xv * wd[i];
                }
                for (keel::usize i = L; i < nfftEff; ++i) {
                    fTd[i] = 0.0;
                    fDd[i] = 0.0;
                }
                Value St = fft(frameT, static_cast<int>(nfftEff), 1, mr);
                Value Sd = fft(frameD, static_cast<int>(nfftEff), 1, mr);
                const keel::c64 *std_ptr = St.complexData();
                const keel::c64 *sdd_ptr = Sd.complexData();
                for (keel::usize k = 0; k < R; ++k) {
                    const keel::c64 s_val = sd[m * R + k];
                    const double mag2 = keel::norm(s_val);
                    if (mag2 > 0.0) {
                        const keel::c64 st_cs = std_ptr[k] * keel::conj(s_val);
                        const keel::c64 sd_cs = sdd_ptr[k] * keel::conj(s_val);
                        tcd[m * R + k] = td[m] + st_cs.real() / (mag2 * fs);
                        const double f_raw = fd[k] - (sd_cs.imag() / (2.0 * M_PI * mag2)) * fs;
                        fcd[m * R + k] = std::fmod(std::fmod(f_raw, fs) + fs, fs);
                    } else {
                        tcd[m * R + k] = td[m];
                        fcd[m * R + k] = fd[k];
                    }
                }
            }
        }
    }

    // Auto-plot when called without LHS (MATLAB spectrogram convention).
    if (nargout == 0) {
        const keel::usize R = S.dims().rows();
        const keel::usize C = S.dims().cols();
        Value pdb = Value::matrix(R, C, ValueType::DOUBLE, ctx.engine->allocator());
        const double *pd = PS.doubleData();
        double *pdb_data = pdb.doubleDataMut();
        for (keel::usize i = 0; i < R * C; ++i) {
            double v = pd[i];
            if (v < 1e-15) v = 1e-15;
            pdb_data[i] = 10.0 * std::log10(v);
        }
        Value imagescFn = Value::funcHandle("imagesc", ctx.engine->allocator());
        Value imgArgs[3] = { T, F, pdb };
        ctx.engine->callFunctionHandle(imagescFn, keel::ArrayRef<Value>(imgArgs, 3), ctx.env);
        Value xlabelFn = Value::funcHandle("xlabel", ctx.engine->allocator());
        Value xlblArg = Value::fromString("Time (s)", ctx.engine->allocator());
        ctx.engine->callFunctionHandle(xlabelFn, keel::ArrayRef<Value>(&xlblArg, 1), ctx.env);
        Value ylabelFn = Value::funcHandle("ylabel", ctx.engine->allocator());
        Value ylblArg = Value::fromString("Frequency (Hz)", ctx.engine->allocator());
        ctx.engine->callFunctionHandle(ylabelFn, keel::ArrayRef<Value>(&ylblArg, 1), ctx.env);
        return;
    }

    outs[0] = std::move(S);
    if (nargout > 1)
        outs[1] = std::move(F);
    if (nargout > 2)
        outs[2] = std::move(T);
    if (nargout > 3)
        outs[3] = std::move(PS);
    if (nargout > 4)
        outs[4] = std::move(FC);
    if (nargout > 5)
        outs[5] = std::move(TC);
}

// Parse MATLAB-style trailing name-value pairs into the stft/istft
// argument set. Recognised keys (case-insensitive): Window, OverlapLength,
// FFTLength, FrequencyRange. Unknown keys throw — better to fail fast
// than to silently misuse a guess.
static void parseStftNVPairs(keel::ArrayRef<Value> args, keel::usize startIdx,
                             Value &window, keel::usize &overlap,
                             keel::usize &fftLength, keel::String &range)
{
    auto eqIgnoreCase = [](keel::StringRef a, const char *b) {
        if (a.size() != keel::StringRef(b).size()) return false;
        for (keel::usize i = 0; i < a.size(); ++i)
            if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
        return true;
    };
    for (keel::usize i = startIdx; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar())
            throw Error("stft: name-value pair name must be a string",
                         0, 0, "stft", "", "numkit:stft:badNVName");
        const keel::String key = args[i].toString();
        const Value &val      = args[i + 1];
        if (eqIgnoreCase(key, "Window"))           window    = val;
        else if (eqIgnoreCase(key, "OverlapLength"))
            overlap = static_cast<keel::usize>(argcheck::expectedInteger(
                argcheck::expectedReal(val, int(i) + 2, "OverlapLength", "stft"),
                int(i) + 2, "OverlapLength", "stft"));
        else if (eqIgnoreCase(key, "FFTLength"))
            fftLength = static_cast<keel::usize>(argcheck::expectedPositiveInteger(
                argcheck::expectedReal(val, int(i) + 2, "FFTLength", "stft"),
                int(i) + 2, "FFTLength", "stft"));
        else if (eqIgnoreCase(key, "FrequencyRange")) range  = val.toString();
        else
            throw Error(keel::format("stft: unknown name-value key '{}'", key),
                         0, 0, "stft", "", "numkit:stft:badNVKey");
    }
}

// Optional positional `fs` between `x` and any NV-pairs. Returns the
// index where NV-pairs start (1 if no fs, 2 if fs consumed).
static keel::usize parseOptionalFs(keel::ArrayRef<Value> args, bool &fsGiven, double &fs)
{
    fsGiven = false;
    fs = 1.0;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()
        && args[1].numel() == 1) {
        fs = args[1].toScalar();
        fsGiven = true;
        return 2;
    }
    return 1;
}

// Build the frequency-axis vector that MATLAB returns from
// `[s, f, ...] = stft(...)`. The angular spacing is `f_scale / NFFT`
// where `f_scale = fs` when given, else `2*pi` (radians/sample default).
static Value buildFreqAxis(keel::usize NFFT, keel::StringRef range,
                           bool fsGiven, double fs,
                           keel::Allocator *mr)
{
    const double f_scale = fsGiven ? fs : (2.0 * M_PI);
    const double df      = f_scale / static_cast<double>(NFFT);
    if (range == "onesided") {
        const keel::usize nF = NFFT / 2 + 1;
        Value f = Value::matrix(nF, 1, ValueType::DOUBLE, mr);
        double *fd = f.doubleDataMut();
        for (keel::usize k = 0; k < nF; ++k)
            fd[k] = static_cast<double>(k) * df;
        return f;
    }
    if (range == "centered") {
        Value f = Value::matrix(NFFT, 1, ValueType::DOUBLE, mr);
        double *fd = f.doubleDataMut();
        // Even N: bins are [-N/2+1, ..., N/2] (Nyquist at end).
        // Odd  N: bins are [-(N-1)/2, ..., (N-1)/2].
        const long long Nll = static_cast<long long>(NFFT);
        const long long lo  = (Nll % 2 == 0) ? -(Nll / 2 - 1) : -((Nll - 1) / 2);
        for (long long k = 0; k < Nll; ++k)
            fd[k] = static_cast<double>(lo + k) * df;
        return f;
    }
    // twosided: bins 0, 1, ..., N-1.
    Value f = Value::matrix(NFFT, 1, ValueType::DOUBLE, mr);
    double *fd = f.doubleDataMut();
    for (keel::usize k = 0; k < NFFT; ++k)
        fd[k] = static_cast<double>(k) * df;
    return f;
}

// Build the time-axis vector for stft frame centres. MATLAB places
// each frame's centre at sample `(M/2 + k*hop)` (0-based), scaled by
// `1/fs_t` where `fs_t = fs` if given else 1 (samples).
static Value buildTimeAxisStft(keel::usize M, keel::usize hop, keel::usize K,
                               bool fsGiven, double fs,
                               keel::Allocator *mr)
{
    const double tscale = fsGiven ? fs : 1.0;
    const double half_M = 0.5 * static_cast<double>(M);
    Value t = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    double *td = t.doubleDataMut();
    for (keel::usize k = 0; k < K; ++k)
        td[k] = (half_M + static_cast<double>(k * hop)) / tscale;
    return t;
}

// Resolve effective window length M given an explicit/empty Window
// arg, matching `resolveWindow` defaults.
static keel::usize resolveWindowLen(const Value &window)
{
    return (window.numel() > 0) ? window.numel() : 128;
}

void stft_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "stft");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    bool fsGiven = false;
    double fs    = 1.0;
    const keel::usize nvStart = parseOptionalFs(args, fsGiven, fs);

    Value window           = Value();
    keel::usize overlap    = SIZE_MAX;
    keel::usize fftLength  = 0;
    keel::String range      = keel::String::from("centered");  // matches MATLAB R2019b+ default
    parseStftNVPairs(args, nvStart, window, overlap, fftLength, range);

    // Resolve sizes (mirrors stft() internals) so we can build f, t.
    const keel::usize M    = resolveWindowLen(window);
    const keel::usize OL   = (overlap == SIZE_MAX) ? (3 * M) / 4 : overlap;
    const keel::usize hop  = (OL < M) ? (M - OL) : 1;
    const keel::usize NFFT = (fftLength == 0) ? M : fftLength;
    const bool isMatrix    = (args[0].dims().rows() > 1 && args[0].dims().cols() > 1);
    const keel::usize N    = isMatrix ? args[0].dims().rows() : args[0].numel();
    const keel::usize K    = (N >= M) ? ((N - M) / hop + 1) : 0;

    const bool anySingle = (args[0].type() == ValueType::SINGLE) || (window.type() == ValueType::SINGLE);
    outs[0] = stft(args[0], window, overlap, fftLength, range, mr);
    if (nargout > 1)
        outs[1] = narrowDoubleToSingleIf(buildFreqAxis(NFFT, range, fsGiven, fs, mr), anySingle, mr);
    if (nargout > 2)
        outs[2] = narrowDoubleToSingleIf(buildTimeAxisStft(M, hop, K, fsGiven, fs, mr), anySingle, mr);
}

void istft_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "istft");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();

    bool fsGiven = false;
    double fs    = 1.0;
    const keel::usize nvStart = parseOptionalFs(args, fsGiven, fs);

    Value window           = Value();
    keel::usize overlap    = SIZE_MAX;
    keel::usize fftLength  = 0;
    keel::String range      = keel::String::from("centered");
    parseStftNVPairs(args, nvStart, window, overlap, fftLength, range);

    outs[0] = istft(args[0], window, overlap, fftLength, range, mr);
    if (nargout > 1) {
        // t = (0 : Nout-1) / fs_t. Reconstructed length is in outs[0]'s rows.
        const keel::usize Nout = outs[0].dims().rows();
        const double tscale    = fsGiven ? fs : 1.0;
        Value t = Value::matrix(Nout, 1, ValueType::DOUBLE, mr);
        double *td = t.doubleDataMut();
        for (keel::usize i = 0; i < Nout; ++i)
            td[i] = static_cast<double>(i) / tscale;
        outs[1] = std::move(t);
    }
}

} // namespace detail

} // namespace numkit::signal
