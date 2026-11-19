// toolboxes/signal/src/spectral_analysis/spectral_metrics_reg.cpp
//
// CallContext register half of spectral_analysis/spectral_metrics.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>
#include <numkit/signal/spectral_analysis/spectral_metrics.hpp>
#include <numkit/signal/transforms/hilbert.hpp>
#include <numkit/signal/windows/windows.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void bandpower_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "bandpower");
    auto *mr = ctx.engine->allocator();
    if (args.size() >= 3 && (args[args.size() - 1].isChar() || args[args.size() - 1].isString())) {
        keel::String rawOpt = args[args.size() - 1].toString();
        keel::String opt = rawOpt.clone();
        for (char &c : opt) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (opt != "psd")
            throw Error(keel::format("bandpower: unrecognized option '{}' ('psd' is the only supported option)", rawOpt),
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

void obw_reg(keel::ArrayRef<Value> args, keel::usize nargout,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "obw");
    const Value &fs = (args.size() >= 2) ? args[1] : Value::Empty;
    double p = 0.99;
    if (args.size() >= 3 && !args[2].isEmpty()) p = args[2].toScalar();
    auto [bw, flo, fhi, pw] = obw(args[0], fs, p, ctx.engine->allocator());
    outs[0] = std::move(bw);
    if (nargout > 1) outs[1] = std::move(flo);
    if (nargout > 2) outs[2] = std::move(fhi);
    if (nargout > 3) outs[3] = std::move(pw);
}

#define NUMKIT_SPEC1_REG(name, fn, mn, mx)                                          \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), mn, mx, #name);                      \
        const Value &fs = (args.size() >= 2) ? args[1] : Value::Empty;          \
        outs[0] = fn(args[0], fs, ctx.engine->allocator());                      \
    }
#define NUMKIT_SPEC1_NV_REG(name, fn)  /* no max: name-value tail in the reference */\
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
        argcheck::checkNargin(args.size(), 1, #name);                           \
        const Value &fs = (args.size() >= 2) ? args[1] : Value::Empty;          \
        outs[0] = fn(args[0], fs, ctx.engine->allocator());                      \
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
static keel::String toLowerCopy(keel::StringRef sv)
{
    keel::String s = keel::String::from(sv);
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static void parsePspectrumTail(keel::ArrayRef<Value> args, keel::usize startIdx,
                               PspectrumOptions &opts, keel::String *methodOut,
                               bool allowTypeFlag, const char *caller)
{
    keel::usize i = startIdx;
    if (allowTypeFlag && i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String argStr = args[i].toString();
        keel::String low = toLowerCopy(argStr);
        if (low == "power" || low == "spectrogram") {
            opts.type = keel::move(low);
            ++i;
        }
    }
    while (i < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(keel::format("{}: expected option name string at argument {}", caller, i + 1),
                        0, 0, caller, "", keel::format("numkit:{}:invalidOption", caller));
        keel::String rawKey = args[i].toString();
        const keel::String key = toLowerCopy(rawKey);
        if (i + 1 >= args.size())
            throw Error(keel::format("{}: missing value for option '{}'", caller, rawKey),
                        0, 0, caller, "", keel::format("numkit:{}:missingOptionValue", caller));
        const Value &val = args[i + 1];
        if (key == "leakage") {
            opts.leakage = argcheck::expectedReal(val, static_cast<int>(i + 2), "Leakage", caller);
        } else if (key == "overlappercent") {
            opts.overlapPercent = argcheck::expectedReal(val, static_cast<int>(i + 2), "OverlapPercent", caller);
        } else if (key == "timeresolution") {
            opts.timeResolution = argcheck::expectedReal(val, static_cast<int>(i + 2), "TimeResolution", caller);
        } else if (key == "frequencylimits") {
            if (val.numel() != 2 || val.isComplex() || val.isChar() || val.isString())
                throw Error(keel::format("{}: 'FrequencyLimits' must be a 2-element real vector", caller),
                            0, 0, caller, "", keel::format("numkit:{}:invalidFrequencyLimits", caller));
            opts.fmin = val.elemAsDouble(0);
            opts.fmax = val.elemAsDouble(1);
        } else if (key == "twosided") {
            opts.hasTwoSided = true;
            opts.twoSided = val.toBool();
        } else if (key == "method" && methodOut != nullptr) {
            if (!val.isChar() && !val.isString())
                throw Error(keel::format("{}: 'Method' value must be a string", caller),
                            0, 0, caller, "", keel::format("numkit:{}:method", caller));
            keel::String valStr = val.toString();
            *methodOut = toLowerCopy(valStr);
        } else {
            throw Error(keel::format("{}: unrecognized option '{}'", caller, rawKey),
                        0, 0, caller, "", keel::format("numkit:{}:invalidOption", caller));
        }
        i += 2;
    }
}

void pspectrum_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 11, "pspectrum");
    PspectrumOptions opts;
    keel::usize idx = 1;
    Value fsOrT = Value::Empty;
    if (idx < args.size() && !args[idx].isChar() && !args[idx].isString()) {
        fsOrT = args[idx];
        ++idx;
    }
    parsePspectrumTail(args, idx, opts, nullptr, true, "pspectrum");
    auto res = pspectrum(args[0], fsOrT, opts, ctx.engine->allocator());
    outs[0] = std::move(res.p);
    if (nargout > 1) outs[1] = std::move(res.f);
    if (nargout > 2) outs[2] = std::move(res.t);
}

void instfreq_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 7, "instfreq");
    PspectrumOptions opts;
    opts.type.assign("spectrogram");
    keel::String method = keel::String::from("tfmoment");
    keel::usize idx = 1;
    Value fsOrT = Value::Empty;
    if (idx < args.size() && !args[idx].isChar() && !args[idx].isString()) {
        fsOrT = args[idx];
        ++idx;
    }
    parsePspectrumTail(args, idx, opts, &method, false, "instfreq");
    if (method == "hilbert") {
        auto [ifr, t] = instfreqHilbert(args[0], fsOrT, ctx.engine->allocator());
        outs[0] = std::move(ifr);
        if (nargout > 1) outs[1] = std::move(t);
        return;
    }
    if (method != "tfmoment")
        throw Error(keel::format("instfreq: 'Method' must be 'tfmoment' or 'hilbert' (got '{}')", method),
                    0, 0, "instfreq", "", "numkit:instfreq:method");
    auto [ifr, t] = instfreqWithTime(args[0], fsOrT, opts, ctx.engine->allocator());
    outs[0] = std::move(ifr);
    if (nargout > 1) outs[1] = std::move(t);
}

void instbw_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 7, "instbw");
    PspectrumOptions opts;
    opts.type.assign("spectrogram");
    keel::usize idx = 1;
    Value fsOrT = Value::Empty;
    if (idx < args.size() && !args[idx].isChar() && !args[idx].isString()) {
        fsOrT = args[idx];
        ++idx;
    }
    parsePspectrumTail(args, idx, opts, nullptr, false, "instbw");
    auto [ibw, t] = instbwWithTime(args[0], fsOrT, opts, ctx.engine->allocator());
    outs[0] = std::move(ibw);
    if (nargout > 1) outs[1] = std::move(t);
}

#undef NUMKIT_SPEC1_REG

} // namespace detail

} // namespace numkit::signal
