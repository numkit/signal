// toolboxes/signal/src/spectral_analysis/periodogram_pwelch_reg.cpp
//
// CallContext register half of spectral_analysis/periodogram_pwelch.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

// MATLAB DSP-spectrum default fs = 2*pi (treats input as normalised
// radian frequency). Explicit fs as the last positional argument
// overrides this.
constexpr keel::f64 kDefaultFs = 2.0 * 3.14159265358979323846;

void periodogram_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "periodogram");  // no max: name-value tail in the reference

    // Positional args (x, window, nfft, fs) precede any string flag
    // ('onesided'/'twosided'/'power'/'psd'/'ConfidenceLevel', value).
    keel::usize firstStr = args.size();
    for (keel::usize i = 1; i < args.size(); ++i)
        if (args[i].isChar()) { firstStr = i; break; }

    Value window = Value();
    if (firstStr > 1 && !args[1].isEmpty())
        window = args[1];
    const keel::usize nfft = (firstStr > 2 && !args[2].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "periodogram"), 3, "nfft", "periodogram")) : 0;
    const keel::f64 fs   = (firstStr > 3 && !args[3].isEmpty()) ? args[3].toScalar() : kDefaultFs;

    // 'ConfidenceLevel', p name-value pair (case-insensitive); MATLAB default 0.95.
    keel::f64 conf = 0.95;
    for (keel::usize i = firstStr; i + 1 < args.size(); ++i) {
        if (!args[i].isChar()) continue;
        keel::String opt = args[i].toString();
        std::transform(opt.begin(), opt.end(), opt.begin(),
                       [](unsigned char c) { return keel::to_ascii_lower(c); });
        if (opt == "confidencelevel")
            conf = args[i + 1].toScalar();
    }

    auto [Pxx, F] = periodogram(args[0], window, nfft, fs, ctx.engine->allocator());

    if (nargout > 2) {
        const bool realInput = !args[0].isComplex();
        // Default NFFT (max(256, 2^nextpow2(N))) is always even â†’ Nyquist bin present.
        const bool nfftEven = (nfft != 0) ? (nfft % 2 == 0) : true;
        outs[2] = periodogramConf(Pxx, conf, realInput, nfftEven, ctx.engine->allocator());
    }
    if (nargout > 1)
        outs[1] = std::move(F);
    outs[0] = std::move(Pxx);
}

void pwelch_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 9, "pwelch");

    Value window = Value();
    if (args.size() >= 2 && !args[1].isChar())
        window = args[1];
    // Empty [] placeholders select the default (MATLAB pwelch(x,[],[],nfft)).
    const keel::usize noverlap = (args.size() >= 3 && !args[2].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "noverlap", "pwelch"),
            3, "noverlap", "pwelch")) : 0;
    const keel::usize nfft = (args.size() >= 4 && !args[3].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[3], 4, "nfft", "pwelch"),
            4, "nfft", "pwelch")) : 0;
    const keel::f64 fs   = (args.size() >= 5 && !args[4].isEmpty()) ? args[4].toScalar() : kDefaultFs;

    bool twoSided = args[0].isComplex();
    for (keel::usize i = 1; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            keel::String s = args[i].toString();
            for (char &c : s) c = keel::to_ascii_lower(c);
            if (s == "twosided" || s == "centered") twoSided = true;
        }
    }
    auto *mr = ctx.engine->allocator();
    auto [Pxx, F] = pwelch(args[0], window, noverlap, nfft, fs, mr);
    if (twoSided && !args[0].isComplex() && Pxx.numel() > 1) {
        const keel::usize nOne = Pxx.numel();
        const keel::usize fullN = (nfft > 0) ? nfft : ((nOne - 1) * 2);
        auto P2 = Value::matrix(fullN, 1, ValueType::DOUBLE, mr);
        auto F2 = Value::matrix(fullN, 1, ValueType::DOUBLE, mr);
        const keel::f64 *p1 = Pxx.doubleData();
        keel::f64 *p2 = P2.doubleDataMut();
        keel::f64 *f2 = F2.doubleDataMut();
        p2[0] = p1[0];
        for (keel::usize k = 1; k < nOne; ++k) {
            if (fullN % 2 == 0 && k == fullN / 2) {
                p2[k] = p1[k];
            } else {
                p2[k] = 0.5 * p1[k];
                if (fullN - k < fullN) p2[fullN - k] = 0.5 * p1[k];
            }
        }
        for (keel::usize k = 0; k < fullN; ++k)
            f2[k] = fs * keel::as<keel::f64>(k) / keel::as<keel::f64>(fullN);
        Pxx = std::move(P2);
        F = std::move(F2);
    }
    outs[0] = std::move(Pxx);
    if (nargout > 1)
        outs[1] = std::move(F);
}

void cpsd_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 8, "cpsd");
    Value window = Value();
    if (args.size() >= 3 && !args[2].isChar()) window = args[2];
    const keel::usize noverlap = (args.size() >= 4 && !args[3].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedInteger(
            argcheck::expectedReal(args[3], 4, "noverlap", "cpsd"),
            4, "noverlap", "cpsd")) : 0;
    const keel::usize nfft     = (args.size() >= 5 && !args[4].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[4], 5, "nfft", "cpsd"),
            5, "nfft", "cpsd")) : 0;
    const keel::f64 fs       = (args.size() >= 6 && !args[5].isEmpty()) ? args[5].toScalar() : kDefaultFs;
    auto [Pxy, F] = cpsd(args[0], args[1], window, noverlap, nfft, fs, ctx.engine->allocator());
    outs[0] = std::move(Pxy);
    if (nargout > 1) outs[1] = std::move(F);
}

void mscohere_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 8, "mscohere");
    Value window = Value();
    if (args.size() >= 3 && !args[2].isChar()) window = args[2];
    const keel::usize noverlap = (args.size() >= 4 && !args[3].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedInteger(
            argcheck::expectedReal(args[3], 4, "noverlap", "mscohere"),
            4, "noverlap", "mscohere")) : 0;
    const keel::usize nfft     = (args.size() >= 5 && !args[4].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[4], 5, "nfft", "mscohere"),
            5, "nfft", "mscohere")) : 0;
    const keel::f64 fs       = (args.size() >= 6 && !args[5].isEmpty()) ? args[5].toScalar() : kDefaultFs;
    auto [Cxy, F] = mscohere(args[0], args[1], window, noverlap, nfft, fs, ctx.engine->allocator());
    outs[0] = std::move(Cxy);
    if (nargout > 1) outs[1] = std::move(F);
}

void tfestimate_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "tfestimate");  // no max: name-value tail in the reference
    Value window = Value();
    if (args.size() >= 3 && !args[2].isChar()) window = args[2];
    const keel::usize noverlap = (args.size() >= 4 && !args[3].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedInteger(
            argcheck::expectedReal(args[3], 4, "noverlap", "tfestimate"),
            4, "noverlap", "tfestimate")) : 0;
    const keel::usize nfft     = (args.size() >= 5 && !args[4].isEmpty())
        ? keel::as<keel::usize>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[4], 5, "nfft", "tfestimate"),
            5, "nfft", "tfestimate")) : 0;
    const keel::f64 fs       = (args.size() >= 6 && !args[5].isEmpty()) ? args[5].toScalar() : kDefaultFs;
    auto [Txy, F] = tfestimate(args[0], args[1], window, noverlap, nfft, fs, ctx.engine->allocator());
    outs[0] = std::move(Txy);
    if (nargout > 1) outs[1] = std::move(F);
}

} // namespace detail

} // namespace numkit::signal
