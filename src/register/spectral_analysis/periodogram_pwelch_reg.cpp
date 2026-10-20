// toolboxes/signal/src/spectral_analysis/periodogram_pwelch_reg.cpp
//
// CallContext register half of spectral_analysis/periodogram_pwelch.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::signal {

namespace detail {

// MATLAB DSP-spectrum default fs = 2*pi (treats input as normalised
// radian frequency). Explicit fs as the last positional argument
// overrides this.
constexpr double kDefaultFs = 2.0 * 3.14159265358979323846;

void periodogram_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "periodogram");  // no max: name-value tail in the reference

    // Positional args (x, window, nfft, fs) precede any string flag
    // ('onesided'/'twosided'/'power'/'psd'/'ConfidenceLevel', value).
    size_t firstStr = args.size();
    for (size_t i = 1; i < args.size(); ++i)
        if (args[i].isChar()) { firstStr = i; break; }

    Value window = Value();
    if (firstStr > 1 && !args[1].isEmpty())
        window = args[1];
    const size_t nfft = (firstStr > 2 && !args[2].isEmpty())
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "periodogram"), 3, "nfft", "periodogram")) : 0;
    const double fs   = (firstStr > 3 && !args[3].isEmpty()) ? args[3].toScalar() : kDefaultFs;

    // 'ConfidenceLevel', p name-value pair (case-insensitive); MATLAB default 0.95.
    double conf = 0.95;
    for (size_t i = firstStr; i + 1 < args.size(); ++i) {
        if (!args[i].isChar()) continue;
        std::string opt = args[i].toString();
        std::transform(opt.begin(), opt.end(), opt.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (opt == "confidencelevel")
            conf = args[i + 1].toScalar();
    }

    auto [Pxx, F] = periodogram(args[0], window, nfft, fs, ctx.engine->resource());

    if (nargout > 2) {
        const bool realInput = !args[0].isComplex();
        // Default NFFT (max(256, 2^nextpow2(N))) is always even â†’ Nyquist bin present.
        const bool nfftEven = (nfft != 0) ? (nfft % 2 == 0) : true;
        outs[2] = periodogramConf(Pxx, conf, realInput, nfftEven, ctx.engine->resource());
    }
    if (nargout > 1)
        outs[1] = std::move(F);
    outs[0] = std::move(Pxx);
}

void pwelch_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 9, "pwelch");

    Value window = Value();
    if (args.size() >= 2 && !args[1].isChar())
        window = args[1];
    // Empty [] placeholders select the default (MATLAB pwelch(x,[],[],nfft)).
    const size_t noverlap = (args.size() >= 3 && !args[2].isEmpty())
        ? static_cast<size_t>(argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "noverlap", "pwelch"),
            3, "noverlap", "pwelch")) : 0;
    const size_t nfft = (args.size() >= 4 && !args[3].isEmpty())
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[3], 4, "nfft", "pwelch"),
            4, "nfft", "pwelch")) : 0;
    const double fs   = (args.size() >= 5 && !args[4].isEmpty()) ? args[4].toScalar() : kDefaultFs;

    bool twoSided = args[0].isComplex();
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            std::string s = args[i].toString();
            for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (s == "twosided" || s == "centered") twoSided = true;
        }
    }
    auto *mr = ctx.engine->resource();
    auto [Pxx, F] = pwelch(args[0], window, noverlap, nfft, fs, mr);
    if (twoSided && !args[0].isComplex() && Pxx.numel() > 1) {
        const size_t nOne = Pxx.numel();
        const size_t fullN = (nfft > 0) ? nfft : ((nOne - 1) * 2);
        auto P2 = Value::matrix(fullN, 1, ValueType::DOUBLE, mr);
        auto F2 = Value::matrix(fullN, 1, ValueType::DOUBLE, mr);
        const double *p1 = Pxx.doubleData();
        double *p2 = P2.doubleDataMut();
        double *f2 = F2.doubleDataMut();
        p2[0] = p1[0];
        for (size_t k = 1; k < nOne; ++k) {
            if (fullN % 2 == 0 && k == fullN / 2) {
                p2[k] = p1[k];
            } else {
                p2[k] = 0.5 * p1[k];
                if (fullN - k < fullN) p2[fullN - k] = 0.5 * p1[k];
            }
        }
        for (size_t k = 0; k < fullN; ++k)
            f2[k] = fs * static_cast<double>(k) / static_cast<double>(fullN);
        Pxx = std::move(P2);
        F = std::move(F2);
    }
    outs[0] = std::move(Pxx);
    if (nargout > 1)
        outs[1] = std::move(F);
}

void cpsd_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 8, "cpsd");
    Value window = Value();
    if (args.size() >= 3 && !args[2].isChar()) window = args[2];
    const size_t noverlap = (args.size() >= 4 && !args[3].isEmpty())
        ? static_cast<size_t>(argcheck::expectedInteger(
            argcheck::expectedReal(args[3], 4, "noverlap", "cpsd"),
            4, "noverlap", "cpsd")) : 0;
    const size_t nfft     = (args.size() >= 5 && !args[4].isEmpty())
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[4], 5, "nfft", "cpsd"),
            5, "nfft", "cpsd")) : 0;
    const double fs       = (args.size() >= 6 && !args[5].isEmpty()) ? args[5].toScalar() : kDefaultFs;
    auto [Pxy, F] = cpsd(args[0], args[1], window, noverlap, nfft, fs, ctx.engine->resource());
    outs[0] = std::move(Pxy);
    if (nargout > 1) outs[1] = std::move(F);
}

void mscohere_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 8, "mscohere");
    Value window = Value();
    if (args.size() >= 3 && !args[2].isChar()) window = args[2];
    const size_t noverlap = (args.size() >= 4 && !args[3].isEmpty())
        ? static_cast<size_t>(argcheck::expectedInteger(
            argcheck::expectedReal(args[3], 4, "noverlap", "mscohere"),
            4, "noverlap", "mscohere")) : 0;
    const size_t nfft     = (args.size() >= 5 && !args[4].isEmpty())
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[4], 5, "nfft", "mscohere"),
            5, "nfft", "mscohere")) : 0;
    const double fs       = (args.size() >= 6 && !args[5].isEmpty()) ? args[5].toScalar() : kDefaultFs;
    auto [Cxy, F] = mscohere(args[0], args[1], window, noverlap, nfft, fs, ctx.engine->resource());
    outs[0] = std::move(Cxy);
    if (nargout > 1) outs[1] = std::move(F);
}

void tfestimate_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "tfestimate");  // no max: name-value tail in the reference
    Value window = Value();
    if (args.size() >= 3 && !args[2].isChar()) window = args[2];
    const size_t noverlap = (args.size() >= 4 && !args[3].isEmpty())
        ? static_cast<size_t>(argcheck::expectedInteger(
            argcheck::expectedReal(args[3], 4, "noverlap", "tfestimate"),
            4, "noverlap", "tfestimate")) : 0;
    const size_t nfft     = (args.size() >= 5 && !args[4].isEmpty())
        ? static_cast<size_t>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[4], 5, "nfft", "tfestimate"),
            5, "nfft", "tfestimate")) : 0;
    const double fs       = (args.size() >= 6 && !args[5].isEmpty()) ? args[5].toScalar() : kDefaultFs;
    auto [Txy, F] = tfestimate(args[0], args[1], window, noverlap, nfft, fs, ctx.engine->resource());
    outs[0] = std::move(Txy);
    if (nargout > 1) outs[1] = std::move(F);
}

} // namespace detail

} // namespace numkit::signal
