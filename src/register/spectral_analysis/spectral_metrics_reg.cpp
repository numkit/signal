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
// instfreq: x, fs, plus MATLAB's 'Method' name-value ('hilbert' exact;
// 'tfmoment' â€” MATLAB's DEFAULT â€” pending the pspectrum port, honest
// not-supported error; bugs/opened/signal/instfreq-instbw.md).
void instfreq_reg(Span<const Value> args, size_t, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 7, "instfreq");
    std::string method = "hilbert";
    for (size_t i = 1; i < args.size(); ++i) {
        if ((args[i].isChar() || args[i].isString())
            && args[i].toString() == "Method" && i + 1 < args.size()) {
            if (!(args[i + 1].isChar() || args[i + 1].isString()))
                throw Error("instfreq: 'Method' value must be a string",
                             0, 0, "instfreq", "", "numkit:instfreq:method");
            method = args[i + 1].toString();
            ++i;
        }
    }
    if (method != "hilbert")
        throw Error("instfreq: method '" + method + "' is not yet supported "
                    "(the default tfmoment needs a pspectrum port â€” see "
                    "bugs/opened/signal/instfreq-instbw.md)",
                     0, 0, "instfreq", "", "numkit:instfreq:method");
    const Value &fs = (args.size() >= 2 && !args[1].isChar() && !args[1].isString())
                          ? args[1] : Value::Empty;
    outs[0] = instfreq(args[0], fs, ctx.engine->resource());
}
NUMKIT_SPEC1_REG(instbw,           instbw, 1, 7)

#undef NUMKIT_SPEC1_REG

} // namespace detail

} // namespace numkit::signal
