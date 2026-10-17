// toolboxes/signal/src/spectral_analysis/spectral_metrics_reg.cpp
//
// CallContext register half of spectral_analysis/spectral_metrics.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>
#include <numkit/signal/spectral_analysis/spectral_metrics.hpp>
#include <numkit/signal/transforms/hilbert.hpp>
#include <numkit/signal/windows/windows.hpp>
#include <numkit/value/error.hpp>
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
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void bandpower_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "bandpower");
    auto *mr = ctx.engine->resource();
    if (args.size() >= 3 && (args[args.size() - 1].isChar() || args[args.size() - 1].isString())) {
        std::string opt = args[args.size() - 1].toString();
        for (char &c : opt) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (opt != "psd")
            throw Error("bandpower: unrecognized option '" + args[args.size() - 1].toString() +
                        "' ('psd' is the only supported option)",
                        0, 0, "bandpower", "", "numkit:bandpower:badOption");
        const Value &fr = (args.size() == 4) ? args[2] : Value::Empty;
        outs[0] = bandpowerPsd(args[0], args[1], fr, mr);
        return;
    }
    if (args.size() >= 4) {
        throw Error("bandpower: unrecognized 4th input option ('psd' is "
                    "the only supported option in the reference engine)",
                    0, 0, "bandpower", "", "numkit:bandpower:badOption");
    }
    const Value &fs = (args.size() >= 2) ? args[1] : Value::Empty;
    const Value &fr = (args.size() >= 3) ? args[2] : Value::Empty;
    outs[0] = bandpower(args[0], fs, fr, mr);
}

void obw_reg(Span<const Value> args, size_t nargout,
             Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "obw");
    const Value &fs = (args.size() >= 2) ? args[1] : Value::Empty;
    double p = 0.99;
    if (args.size() >= 3 && !args[2].isEmpty()) p = args[2].toScalar();
    auto [bw, flo, fhi, pw] = obw(args[0], fs, p, ctx.engine->resource());
    outs[0] = std::move(bw);
    if (nargout > 1) outs[1] = std::move(flo);
    if (nargout > 2) outs[2] = std::move(fhi);
    if (nargout > 3) outs[3] = std::move(pw);
}

#define NUMKIT_SPEC1_REG(name, fn, mn, mx)                                          \
    void name##_reg(Span<const Value> args, size_t /*nargout*/,                \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), mn, mx, #name);                      \
        const Value &fs = (args.size() >= 2) ? args[1] : Value::Empty;          \
        outs[0] = fn(args[0], fs, ctx.engine->resource());                      \
    }
#define NUMKIT_SPEC1_NV_REG(name, fn)  /* no max: name-value tail in the reference */\
    void name##_reg(Span<const Value> args, size_t /*nargout*/,                \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 1, #name);                           \
        const Value &fs = (args.size() >= 2) ? args[1] : Value::Empty;          \
        outs[0] = fn(args[0], fs, ctx.engine->resource());                      \
    }

NUMKIT_SPEC1_REG(meanfreq,         meanfreq, 1, 4)
NUMKIT_SPEC1_REG(medfreq,          medfreq, 1, 4)
NUMKIT_SPEC1_REG(enbw,             enbw, 1, 2)
NUMKIT_SPEC1_REG(powerbw,          powerbw, 1, 5)
NUMKIT_SPEC1_NV_REG(spectralcrest,    spectralcrest)
NUMKIT_SPEC1_NV_REG(spectralflatness, spectralflatness)
NUMKIT_SPEC1_NV_REG(spectralentropy,  spectralentropy)
NUMKIT_SPEC1_NV_REG(spectralkurtosis, spectralkurtosis)
NUMKIT_SPEC1_NV_REG(spectralskewness, spectralskewness)
NUMKIT_SPEC1_REG(snr,              snr, 1, 6)
NUMKIT_SPEC1_REG(sinad,            sinad, 1, 4)
NUMKIT_SPEC1_REG(thd,              thd, 1, 6)
NUMKIT_SPEC1_REG(sfdr,             sfdr, 1, 4)
static std::string toLowerCopy(std::string s)
{
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static void parsePspectrumTail(Span<const Value> args, size_t startIdx,
                               PspectrumOptions &opts, std::string *methodOut,
                               bool allowTypeFlag, const char *caller)
{
    size_t i = startIdx;
    if (allowTypeFlag && i < args.size() && (args[i].isChar() || args[i].isString())) {
        const std::string low = toLowerCopy(args[i].toString());
        if (low == "power" || low == "spectrogram") {
            opts.type = low;
            ++i;
        }
    }
    while (i < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(std::string(caller) + ": expected option name string at argument " +
                        std::to_string(i + 1),
                        0, 0, caller, "", std::string("numkit:") + caller + ":invalidOption");
        const std::string key = toLowerCopy(args[i].toString());
        if (i + 1 >= args.size())
            throw Error(std::string(caller) + ": missing value for option '" + args[i].toString() + "'",
                        0, 0, caller, "", std::string("numkit:") + caller + ":missingOptionValue");
        const Value &val = args[i + 1];
        if (key == "leakage") {
            opts.leakage = argcheck::expectedReal(val, static_cast<int>(i + 2), "Leakage", caller);
        } else if (key == "overlappercent") {
            opts.overlapPercent = argcheck::expectedReal(val, static_cast<int>(i + 2), "OverlapPercent", caller);
        } else if (key == "timeresolution") {
            opts.timeResolution = argcheck::expectedReal(val, static_cast<int>(i + 2), "TimeResolution", caller);
        } else if (key == "frequencylimits") {
            if (val.numel() != 2 || val.isComplex() || val.isChar() || val.isString())
                throw Error(std::string(caller) + ": 'FrequencyLimits' must be a 2-element real vector",
                            0, 0, caller, "", std::string("numkit:") + caller + ":invalidFrequencyLimits");
            opts.fmin = val.elemAsDouble(0);
            opts.fmax = val.elemAsDouble(1);
        } else if (key == "twosided") {
            opts.hasTwoSided = true;
            opts.twoSided = val.toBool();
        } else if (key == "method" && methodOut != nullptr) {
            if (!val.isChar() && !val.isString())
                throw Error(std::string(caller) + ": 'Method' value must be a string",
                            0, 0, caller, "", std::string("numkit:") + caller + ":method");
            *methodOut = toLowerCopy(val.toString());
        } else {
            throw Error(std::string(caller) + ": unrecognized option '" + args[i].toString() + "'",
                        0, 0, caller, "", std::string("numkit:") + caller + ":invalidOption");
        }
        i += 2;
    }
}

void pspectrum_reg(Span<const Value> args, size_t nargout,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 11, "pspectrum");
    PspectrumOptions opts;
    size_t idx = 1;
    Value fsOrT = Value::Empty;
    if (idx < args.size() && !args[idx].isChar() && !args[idx].isString()) {
        fsOrT = args[idx];
        ++idx;
    }
    parsePspectrumTail(args, idx, opts, nullptr, true, "pspectrum");
    auto res = pspectrum(args[0], fsOrT, opts, ctx.engine->resource());
    outs[0] = std::move(res.p);
    if (nargout > 1) outs[1] = std::move(res.f);
    if (nargout > 2) outs[2] = std::move(res.t);
}

void instfreq_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 7, "instfreq");
    PspectrumOptions opts;
    opts.type = "spectrogram";
    std::string method = "tfmoment";
    size_t idx = 1;
    Value fsOrT = Value::Empty;
    if (idx < args.size() && !args[idx].isChar() && !args[idx].isString()) {
        fsOrT = args[idx];
        ++idx;
    }
    parsePspectrumTail(args, idx, opts, &method, false, "instfreq");
    if (method == "hilbert") {
        auto [ifr, t] = instfreqHilbert(args[0], fsOrT, ctx.engine->resource());
        outs[0] = std::move(ifr);
        if (nargout > 1) outs[1] = std::move(t);
        return;
    }
    if (method != "tfmoment")
        throw Error("instfreq: 'Method' must be 'tfmoment' or 'hilbert' (got '" + method + "')",
                    0, 0, "instfreq", "", "numkit:instfreq:method");
    auto [ifr, t] = instfreqWithTime(args[0], fsOrT, opts, ctx.engine->resource());
    outs[0] = std::move(ifr);
    if (nargout > 1) outs[1] = std::move(t);
}

void instbw_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 7, "instbw");
    PspectrumOptions opts;
    opts.type = "spectrogram";
    size_t idx = 1;
    Value fsOrT = Value::Empty;
    if (idx < args.size() && !args[idx].isChar() && !args[idx].isString()) {
        fsOrT = args[idx];
        ++idx;
    }
    parsePspectrumTail(args, idx, opts, nullptr, false, "instbw");
    auto [ibw, t] = instbwWithTime(args[0], fsOrT, opts, ctx.engine->resource());
    outs[0] = std::move(ibw);
    if (nargout > 1) outs[1] = std::move(t);
}

#undef NUMKIT_SPEC1_REG

} // namespace detail

} // namespace numkit::signal
