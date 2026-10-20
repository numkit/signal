// toolboxes/signal/src/filter_analysis/frequency_response_reg.cpp
//
// Register half of the signal frequency_response builtins: the CallContext wrappers
// delegating to the engine-free compute in frequency_response.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <numkit/signal/filter_analysis/frequency_response.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::signal {

namespace detail {

void freqz_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "freqz");
    auto *mr = ctx.engine->resource();
    Value a = (args.size() >= 2 && !args[1].isEmpty()) ? args[1] : Value::scalar(1.0, mr);
    // Parse the trailing args: 'whole' (string, any position) and up to two
    // numerics after b,a — the first is n (npts), the second is fs (the
    // freqz(b,a,n,fs) / freqz(b,a,n,'whole',fs) sample-rate form).
    size_t npts = 512;
    double fs   = 0.0;
    bool   whole = false;
    int    numericSeen = 0;
    const Value *wvec = nullptr;
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            std::string s = args[i].toString();
            for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (s == "whole") whole = true;
        } else if (!args[i].isEmpty()) {
            if (numericSeen == 0 && args[i].numel() > 1)
                wvec = &args[i];           // freqz(b,a,W): vector-w form
            else if (numericSeen == 0)
                npts = static_cast<size_t>(argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i], int(i) + 1, "n", "freqz"),
                    int(i) + 1, "n", "freqz"));
            else if (numericSeen == 1) fs   = args[i].toScalar();
            ++numericSeen;
        }
    }

    if (wvec) {
        auto [Hv, Wv] = freqz(args[0], a, *wvec, mr);
        outs[0] = std::move(Hv);
        if (nargout > 1)
            outs[1] = std::move(Wv);
        return;
    }
    auto [H, W] = freqz(args[0], a, npts, whole, fs, mr);
    outs[0] = std::move(H);
    if (nargout > 1)
        outs[1] = std::move(W);
}

void phasez_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 6, "phasez");
    auto *mr = ctx.engine->resource();
    Value a = (args.size() >= 2 && !args[1].isEmpty()) ? args[1] : Value::scalar(1.0, mr);
    // n (1st numeric after b,a) and optional fs (2nd) — phasez(b,a,n,fs).
    size_t npts = 512;
    double fs = 0.0;
    int numericSeen = 0;
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString() || args[i].isEmpty()) continue;
            if (numericSeen == 0)
                npts = static_cast<size_t>(argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i], int(i) + 1, "n", "phasez"),
                    int(i) + 1, "n", "phasez"));
        else if (numericSeen == 1) fs   = args[i].toScalar();
        ++numericSeen;
    }
    auto [phi, W] = phasez(args[0], a, npts, fs, mr);
    outs[0] = std::move(phi);
    if (nargout > 1) outs[1] = std::move(W);
}

void grpdelay_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 6, "grpdelay");
    auto *mr = ctx.engine->resource();
    Value a = (args.size() >= 2 && !args[1].isEmpty()) ? args[1] : Value::scalar(1.0, mr);
    // n (1st numeric after b,a) and optional fs (2nd) — grpdelay(b,a,n,fs).
    size_t npts = 512;
    double fs = 0.0;
    int numericSeen = 0;
    const Value *fvec = nullptr;
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString() || args[i].isEmpty()) continue;
        if (numericSeen == 0 && args[i].numel() > 1)
            fvec = &args[i];
        else if (numericSeen == 0)
            npts = static_cast<size_t>(argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[i], int(i) + 1, "n", "grpdelay"),
                int(i) + 1, "n", "grpdelay"));
        else if (numericSeen == 1) fs = args[i].toScalar();
        ++numericSeen;
    }
    if (fvec) {
        const size_t nf = fvec->numel();
        const size_t nb = args[0].numel(), na = a.numel();
        auto gd = Value::matrix(nf, 1, ValueType::DOUBLE, mr);
        if (nf > 0 && nb > 0 && na > 0) {
            ScratchArena scratch(mr);
            const size_t nc = nb + na - 1;
            auto c = ScratchVec<double>(nc, &scratch);
            for (size_t i = 0; i < nc; ++i) c[i] = 0.0;
            for (size_t i = 0; i < nb; ++i)
                for (size_t j = 0; j < na; ++j)
                    c[i + j] += args[0].elemAsDouble(i) * a.elemAsDouble(na - 1 - j);
            double *g = gd.doubleDataMut();
            const double offset = static_cast<double>(na) - 1.0;
            constexpr double kPi = 3.14159265358979323846;
            for (size_t k = 0; k < nf; ++k) {
                const double fk = fvec->elemAsDouble(k);
                const double w = (fs > 0.0) ? (2.0 * kPi * fk / fs) : fk;
                const Complex ejw(std::cos(w), -std::sin(w));
                Complex C(0, 0), CR(0, 0), ejwk(1, 0);
                for (size_t n = 0; n < nc; ++n) {
                    C  += c[n] * ejwk;
                    CR += (static_cast<double>(n) * c[n]) * ejwk;
                    ejwk *= ejw;
                }
                const double cmag2 = C.real() * C.real() + C.imag() * C.imag();
                g[k] = (cmag2 < 1e-300) ? 0.0 : ((CR / C).real() - offset);
            }
        }
        outs[0] = std::move(gd);
        if (nargout > 1) {
            auto fv = Value::matrix(nf, 1, ValueType::DOUBLE, mr);
            for (size_t k = 0; k < nf; ++k) fv.doubleDataMut()[k] = fvec->elemAsDouble(k);
            outs[1] = std::move(fv);
        }
        return;
    }
    auto [gd, W] = grpdelay(args[0], a, npts, fs, mr);
    outs[0] = std::move(gd);
    if (nargout > 1) outs[1] = std::move(W);
}

} // namespace detail

} // namespace numkit::signal
