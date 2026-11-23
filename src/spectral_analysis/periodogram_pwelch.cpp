// toolboxes/signal/src/spectral_analysis/periodogram_pwelch.cpp
//
// periodogram, pwelch, cpsd, mscohere, tfestimate delegating numerical kernels to keel::signal.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/internal/shims/specfun.hpp>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "../dsp_helpers.hpp"

#include <algorithm>
#include <keel/tuple.h>

namespace numkit::signal {

namespace {

keel::Array<keel::f64> resolveWelchWindow(const Value &window, keel::usize nx,
                                          keel::Allocator *alloc) {
    if (window.numel() > 0) {
        const keel::usize winLen = window.numel();
        keel::Array<keel::f64> win(winLen, 0.0, alloc);
        const keel::f64 *w = window.doubleData();
        for (keel::usize i = 0; i < winLen; ++i) win[i] = w[i];
        return win;
    }
    keel::usize winLen = std::max<keel::usize>(
        1, keel::as<keel::usize>(keel::floor(keel::as<keel::f64>(nx) / 4.5)));
    if (winLen > nx) winLen = nx;
    keel::Array<keel::f64> win(winLen, 0.0, alloc);
    keel::signal::hamming(win.as_mut_ref());
    return win;
}

} // anonymous namespace

keel::Tuple<Value, Value>
periodogram(const Value &x, const Value &window, keel::usize nfft, keel::f64 fs,
            keel::Allocator *mr) {
    const keel::usize N = x.numel();
    const bool isCplx = x.isComplex();
    if (nfft == 0) nfft = std::max<keel::usize>(256, nextPow2(N));

    keel::ArrayRef<keel::f64> winRef(
        (window.numel() == N) ? window.doubleData() : nullptr,
        (window.numel() == N) ? N : 0);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (isCplx) {
        auto Pxx = Value::matrix(nfft, 1, ValueType::DOUBLE, mr);
        auto F   = Value::matrix(nfft, 1, ValueType::DOUBLE, mr);
        (void)keel::signal::periodogram_c64(
            keel::ArrayRef<keel::c64>(x.complexData(), N), winRef, nfft, fs,
            keel::MutArrayRef<keel::f64>(Pxx.doubleDataMut(), nfft),
            keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nfft),
            scratch.allocator());
        return keel::make_tuple(keel::move(Pxx), keel::move(F));
    }

    const keel::usize nOut = nfft / 2 + 1;
    auto Pxx = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    auto F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::periodogram(
        keel::ArrayRef<keel::f64>(x.doubleData(), N), winRef, nfft, fs,
        keel::MutArrayRef<keel::f64>(Pxx.doubleDataMut(), nOut),
        keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nOut),
        scratch.allocator());
    return keel::make_tuple(keel::move(Pxx), keel::move(F));
}

Value
periodogramConf(const Value &Pxx, keel::f64 confidenceLevel, bool realInput, bool nfftEven,
                keel::Allocator *mr) {
    const keel::usize nf = Pxx.numel();
    const keel::f64 *p = Pxx.doubleData();

    const keel::f64 alpha = 1.0 - confidenceLevel;
    const keel::f64 qLo   = 1.0 - alpha / 2.0;
    const keel::f64 qHi   = alpha / 2.0;

    const keel::f64 c2Lo = -2.0 * keel::log(1.0 - qLo);
    const keel::f64 c2Hi = -2.0 * keel::log(1.0 - qHi);
    const keel::f64 eLo  = numkit::builtin::erfinv(Value::scalar(qLo, mr)).toScalar();
    const keel::f64 eHi  = numkit::builtin::erfinv(Value::scalar(qHi, mr)).toScalar();
    const keel::f64 c1Lo = 2.0 * eLo * eLo;
    const keel::f64 c1Hi = 2.0 * eHi * eHi;

    const keel::f64 r2Lo = 2.0 / c2Lo, r2Hi = 2.0 / c2Hi;
    const keel::f64 r1Lo = 1.0 / c1Lo, r1Hi = 1.0 / c1Hi;

    Value out = Value::matrix(nf, 2, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < nf; ++i) {
        const bool oneDof = realInput && (i == 0 || (nfftEven && i + 1 == nf));
        o[i]      = p[i] * (oneDof ? r1Lo : r2Lo);
        o[nf + i] = p[i] * (oneDof ? r1Hi : r2Hi);
    }
    return out;
}

keel::Tuple<Value, Value>
pwelch(const Value &x, const Value &window, keel::usize noverlap, keel::usize nfft, keel::f64 fs,
       keel::Allocator *mr) {
    const keel::usize nx = x.numel();
    const bool isCplx = x.isComplex();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    const auto win = resolveWelchWindow(window, nx, scratch.allocator());
    const keel::usize winLen = win.size();
    if (noverlap == 0) noverlap = winLen / 2;
    if (nfft == 0)     nfft = std::max<keel::usize>(256, nextPow2(winLen));

    if (isCplx) {
        auto Pxx = Value::matrix(nfft, 1, ValueType::DOUBLE, mr);
        auto F   = Value::matrix(nfft, 1, ValueType::DOUBLE, mr);
        (void)keel::signal::pwelch_c64(
            keel::ArrayRef<keel::c64>(x.complexData(), nx), win.as_ref(), noverlap, nfft, fs,
            keel::MutArrayRef<keel::f64>(Pxx.doubleDataMut(), nfft),
            keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nfft),
            scratch.allocator());
        return keel::make_tuple(keel::move(Pxx), keel::move(F));
    }

    const keel::usize nOut = nfft / 2 + 1;
    auto Pxx = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    auto F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::pwelch(
        keel::ArrayRef<keel::f64>(x.doubleData(), nx), win.as_ref(), noverlap, nfft, fs,
        keel::MutArrayRef<keel::f64>(Pxx.doubleDataMut(), nOut),
        keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nOut),
        scratch.allocator());
    return keel::make_tuple(keel::move(Pxx), keel::move(F));
}

keel::Tuple<Value, Value>
cpsd(const Value &x, const Value &y, const Value &window, keel::usize noverlap, keel::usize nfft,
     keel::f64 fs, keel::Allocator *mr) {
    if (x.numel() != y.numel())
        throw Error("cpsd/mscohere: x and y must have the same length",
                    0, 0, "cpsd", "", "numkit:cpsd:size");
    const keel::usize nx = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto win = resolveWelchWindow(window, nx, scratch.allocator());
    if (noverlap == 0) noverlap = win.size() / 2;
    if (nfft == 0)     nfft = std::max<keel::usize>(256, nextPow2(win.size()));

    const keel::usize nOut = nfft / 2 + 1;
    Value Pxy = Value::matrix(nOut, 1, ValueType::COMPLEX, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::cpsd(
        keel::ArrayRef<keel::f64>(x.doubleData(), nx),
        keel::ArrayRef<keel::f64>(y.doubleData(), nx),
        win.as_ref(), noverlap, nfft, fs,
        keel::MutArrayRef<keel::c64>(Pxy.complexDataMut(), nOut),
        keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nOut),
        scratch.allocator());
    return keel::make_tuple(keel::move(Pxy), keel::move(F));
}

keel::Tuple<Value, Value>
mscohere(const Value &x, const Value &y, const Value &window, keel::usize noverlap,
         keel::usize nfft, keel::f64 fs, keel::Allocator *mr) {
    if (x.numel() != y.numel())
        throw Error("cpsd/mscohere: x and y must have the same length",
                    0, 0, "cpsd", "", "numkit:cpsd:size");
    const keel::usize nx = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto win = resolveWelchWindow(window, nx, scratch.allocator());
    if (noverlap == 0) noverlap = win.size() / 2;
    if (nfft == 0)     nfft = std::max<keel::usize>(256, nextPow2(win.size()));

    const keel::usize nOut = nfft / 2 + 1;
    Value Cxy = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::mscohere(
        keel::ArrayRef<keel::f64>(x.doubleData(), nx),
        keel::ArrayRef<keel::f64>(y.doubleData(), nx),
        win.as_ref(), noverlap, nfft, fs,
        keel::MutArrayRef<keel::f64>(Cxy.doubleDataMut(), nOut),
        keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nOut),
        scratch.allocator());
    return keel::make_tuple(keel::move(Cxy), keel::move(F));
}

keel::Tuple<Value, Value>
tfestimate(const Value &x, const Value &y, const Value &window, keel::usize noverlap,
           keel::usize nfft, keel::f64 fs, keel::Allocator *mr) {
    if (x.numel() != y.numel())
        throw Error("cpsd/mscohere: x and y must have the same length",
                    0, 0, "cpsd", "", "numkit:cpsd:size");
    const keel::usize nx = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto win = resolveWelchWindow(window, nx, scratch.allocator());
    if (noverlap == 0) noverlap = win.size() / 2;
    if (nfft == 0)     nfft = std::max<keel::usize>(256, nextPow2(win.size()));

    const keel::usize nOut = nfft / 2 + 1;
    Value Txy = Value::matrix(nOut, 1, ValueType::COMPLEX, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::tfestimate(
        keel::ArrayRef<keel::f64>(x.doubleData(), nx),
        keel::ArrayRef<keel::f64>(y.doubleData(), nx),
        win.as_ref(), noverlap, nfft, fs,
        keel::MutArrayRef<keel::c64>(Txy.complexDataMut(), nOut),
        keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nOut),
        scratch.allocator());
    return keel::make_tuple(keel::move(Txy), keel::move(F));
}

} // namespace numkit::signal
