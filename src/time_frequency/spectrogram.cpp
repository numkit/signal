// toolboxes/signal/src/time_frequency/spectrogram.cpp
//
// spectrogram, stft, istft, and iscola delegating numerical kernels to keel::signal.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/time_frequency/spectrogram.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "../dsp_helpers.hpp"
#include "../signal_detail.hpp"

#include <algorithm>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace {

void fillHannPeriodic(keel::f64 *w, keel::usize N) {
    if (N == 1) { w[0] = 1.0; return; }
    for (keel::usize i = 0; i < N; ++i)
        w[i] = 0.5 * (1.0 - keel::cos(keel::kTau * keel::as<keel::f64>(i) / keel::as<keel::f64>(N)));
}

void resolveStftWindow(const Value &win, keel::Array<keel::f64> &out) {
    if (win.numel() > 0) {
        const keel::usize n = win.numel();
        out.resize(n);
        for (keel::usize i = 0; i < n; ++i) out[i] = win.elemAsDouble(i);
        return;
    }
    out.resize(128);
    fillHannPeriodic(out.data(), 128);
}

keel::signal::StftRange parseStftRange(keel::StringRef range, const char *fn) {
    if (range == "twosided") return keel::signal::StftRange::kTwoSided;
    if (range == "onesided") return keel::signal::StftRange::kOneSided;
    if (range == "centered") return keel::signal::StftRange::kCentered;
    throw Error(keel::String::from(fn) == "stft"
                    ? "stft: FrequencyRange must be 'twosided', 'centered' or 'onesided'"
                    : "istft: FrequencyRange must be 'twosided', 'centered' or 'onesided'",
                0, 0, fn, "", keel::String::from(fn) == "stft" ? "numkit:stft:badRange" : "numkit:istft:badRange");
}

} // anonymous namespace

std::tuple<Value, Value, Value>
spectrogram(const Value &x, const Value &window, keel::usize noverlap, keel::usize nfft,
            keel::Allocator *mr) {
    Value xD = detail::widenSingleToDouble(x, mr);
    const keel::usize nx = xD.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    keel::usize winLen;
    keel::Array<keel::f64> win(scratch.allocator());
    if (window.numel() > 0) {
        winLen = std::min(window.numel(), nx);
        win.resize(winLen);
        for (keel::usize i = 0; i < winLen; ++i) win[i] = window.elemAsDouble(i);
    } else {
        winLen = std::max<keel::usize>(
            1, keel::as<keel::usize>(keel::floor(keel::as<keel::f64>(nx) / 4.5)));
        if (winLen > nx) winLen = nx;
        win.resize(winLen);
        keel::signal::hamming(win.as_mut_ref());
    }
    if (noverlap == 0) noverlap = winLen / 2;
    if (nfft == 0)     nfft = std::max<keel::usize>(256, nextPow2(winLen));

    const keel::usize nFreqs = nfft / 2 + 1;
    const keel::usize step = winLen - noverlap;
    const keel::usize nSegments = (winLen <= nx) ? ((nx - winLen) / step + 1) : 0;

    auto S = Value::complexMatrix(nFreqs, nSegments, mr);
    auto F = Value::matrix(nFreqs, 1, ValueType::DOUBLE, mr);
    auto T = Value::matrix(1, nSegments, ValueType::DOUBLE, mr);

    (void)keel::signal::spectrogram(
        keel::ArrayRef<keel::f64>(xD.doubleData(), nx), win.as_ref(), noverlap, nfft,
        keel::MutArrayRef<keel::c64>(S.complexDataMut(), nFreqs * nSegments),
        keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nFreqs),
        keel::MutArrayRef<keel::f64>(T.doubleDataMut(), nSegments),
        scratch.allocator());

    return std::make_tuple(std::move(S), std::move(F), std::move(T));
}

Value stft(const Value &x, const Value &window, keel::usize overlap, keel::usize fftLength,
           keel::StringRef range, keel::Allocator *mr) {
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
    keel::Array<keel::f64> win(scratch.allocator());
    resolveStftWindow(window, win);
    const keel::usize M = win.size();
    if (M == 0)
        throw Error("stft: window must be non-empty", 0, 0, "stft", "", "numkit:stft:badWindow");
    if (M > N)
        throw Error("stft: signal shorter than window length", 0, 0, "stft", "", "numkit:stft:shortSignal");

    const keel::usize OL = (overlap == keel::Limits<keel::usize>::kMax) ? (3 * M) / 4 : overlap;
    if (OL >= M)
        throw Error("stft: OverlapLength must be < window length", 0, 0, "stft", "", "numkit:stft:badOverlap");
    const keel::usize NFFT = (fftLength == 0) ? M : fftLength;
    if (NFFT < M)
        throw Error("stft: FFTLength must be >= window length", 0, 0, "stft", "", "numkit:stft:badNfft");

    const keel::signal::StftRange rng = parseStftRange(range, "stft");
    const keel::usize hop = M - OL;
    const keel::usize K   = (N - M) / hop + 1;
    const keel::usize outRows = (rng == keel::signal::StftRange::kOneSided) ? (NFFT / 2 + 1) : NFFT;

    const bool xCplx = x.isComplex();
    Value xD = xCplx ? Value() : detail::widenSingleToDouble(x, mr);

    Value S = (nCh > 1)
        ? Value::matrix3d(outRows, K, nCh, ValueType::COMPLEX, mr)
        : Value::complexMatrix(outRows, K, mr);
    keel::c64 *Sd = S.complexDataMut();

    for (keel::usize ch = 0; ch < nCh; ++ch) {
        keel::MutArrayRef<keel::c64> sCh(Sd + ch * (outRows * K), outRows * K);
        if (xCplx) {
            (void)keel::signal::stft_c64(keel::ArrayRef<keel::c64>(x.complexData() + ch * N, N),
                                         win.as_ref(), OL, NFFT, rng, sCh, scratch.allocator());
        } else {
            (void)keel::signal::stft(keel::ArrayRef<keel::f64>(xD.doubleData() + ch * N, N),
                                     win.as_ref(), OL, NFFT, rng, sCh, scratch.allocator());
        }
    }
    return S;
}

Value istft(const Value &S, const Value &window, keel::usize overlap, keel::usize fftLength,
            keel::StringRef range, keel::Allocator *mr) {
    const auto &d = S.dims();
    const keel::usize inRows = d.rows();
    const keel::usize K      = d.cols();
    if (K == 0 || inRows == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> win(scratch.allocator());
    resolveStftWindow(window, win);
    const keel::usize M = win.size();
    if (M == 0)
        throw Error("istft: window must be non-empty", 0, 0, "istft", "", "numkit:istft:badWindow");

    const keel::usize OL = (overlap == keel::Limits<keel::usize>::kMax) ? (3 * M) / 4 : overlap;
    if (OL >= M)
        throw Error("istft: OverlapLength must be < window length", 0, 0, "istft", "", "numkit:istft:badOverlap");

    const keel::signal::StftRange rng = parseStftRange(range, "istft");
    const bool isOneSided = (rng == keel::signal::StftRange::kOneSided);
    const keel::usize NFFT = (fftLength == 0)
        ? (isOneSided ? 2 * (inRows - 1) : inRows)
        : fftLength;
    if (!isOneSided && inRows != NFFT)
        throw Error("istft: STFT row count does not match FFTLength", 0, 0, "istft", "", "numkit:istft:badShape");
    if (isOneSided && inRows != NFFT / 2 + 1)
        throw Error("istft: one-sided STFT row count must equal FFTLength/2 + 1", 0, 0, "istft", "", "numkit:istft:badShape");
    if (NFFT < M)
        throw Error("istft: FFTLength must be >= window length", 0, 0, "istft", "", "numkit:istft:badNfft");

    const keel::usize hop = M - OL;
    const keel::usize Nout = (K - 1) * hop + M;
    Value out = Value::matrix(Nout, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::istft(
        keel::ArrayRef<keel::c64>(S.complexData(), inRows * K), inRows, K,
        win.as_ref(), OL, NFFT, rng,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), Nout),
        scratch.allocator());
    return out;
}

std::tuple<Value, Value, Value>
iscola(const Value &window, keel::usize noverlap, keel::StringRef method, keel::Allocator *mr) {
    const keel::usize M = window.numel();
    if (M == 0)
        throw Error("iscola: window must be non-empty", 0, 0, "iscola", "", "numkit:iscola:badWindow");
    if (noverlap >= M)
        throw Error("iscola: noverlap must be < window length", 0, 0, "iscola", "", "numkit:iscola:badOverlap");
    if (method != "ola" && method != "wola")
        throw Error("iscola: method must be 'ola' or 'wola'", 0, 0, "iscola", "", "numkit:iscola:badMethod");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> w(M, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < M; ++i) w[i] = window.elemAsDouble(i);

    keel::signal::ColaResult res;
    (void)keel::signal::iscola(w.as_ref(), noverlap, method == "wola", res, scratch.allocator());
    return {Value::scalar(res.is_cola ? 1.0 : 0.0, mr),
            Value::scalar(res.median, mr),
            Value::scalar(res.max_dev, mr)};
}

} // namespace numkit::signal
