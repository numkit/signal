// toolboxes/signal/src/convolution/convolution_reg.cpp
//
// Register half of the signal convolution builtins: the CallContext wrappers
// delegating to the engine-free compute in convolution.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <numkit/signal/convolution/convolution.hpp>

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

// Shape options must be strings (the reference rejects a numeric shape
// argument with unknownShapeParameter). conv2's numeric third argument
// is instead the SEPARABLE form conv2(u, v, A) â€” unimplemented, voiced
// honestly instead of being silently ignored.
static std::string shapeArg(const Value &v, std::string_view fn)
{
    if (v.isChar() || v.isString())
        return v.toString();
    throw Error(std::string(fn) + ": shape option must be 'full', 'same', "
                    "or 'valid' (got a non-string argument)",
                0, 0, std::string(fn), "",
                "numkit:" + std::string(fn) + ":badShape");
}

void conv_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "conv");

    std::string shape = "full";
    if (args.size() >= 3 && !args[2].isEmpty())
        shape = shapeArg(args[2], "conv");

    outs[0] = conv(args[0], args[1], shape, ctx.engine->resource());
}

void deconv_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "deconv");

    auto [q, r] = deconv(args[0], args[1], ctx.engine->resource());
    outs[0] = std::move(q);
    if (nargout > 1)
        outs[1] = std::move(r);
}

void xcorr_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "xcorr");
    auto *mr = ctx.engine->resource();
    const Value &x = args[0];
    Value y = x;                 // default: autocorrelation
    bool haveY = false;
    int maxlag = -1;             // -1 => full
    std::string scaleopt = "none";

    // Disambiguate (MATLAB): a string is scaleopt; a scalar numeric in the
    // y-slot is maxlag (autocorr); a vector numeric is y. Trailing args:
    // numeric => maxlag, string => scaleopt.
    size_t idx = 1;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString())
            scaleopt = args[1].toString();
        else if (args[1].numel() == 1)
            maxlag = std::abs(argcheck::expectedInteger(
                argcheck::expectedReal(args[1], 2, "maxlag", "xcorr"),
                2, "maxlag", "xcorr"));
        else { y = args[1]; haveY = true; }
        idx = 2;
    }
    for (; idx < args.size(); ++idx) {
        if (args[idx].isEmpty()) continue;
        if (args[idx].isChar() || args[idx].isString())
            scaleopt = args[idx].toString();
        else
            maxlag = std::abs(argcheck::expectedInteger(
                argcheck::expectedReal(args[idx], int(idx) + 1, "maxlag", "xcorr"),
                int(idx) + 1, "maxlag", "xcorr"));
    }

    auto [cfull, lagsfull] = haveY ? xcorr(x, y, mr) : xcorr(x, mr);
    const size_t nc = cfull.numel();
    double *cd = cfull.doubleDataMut();
    const double *ld = lagsfull.doubleData();
    const size_t nx = x.numel(), ny = y.numel();
    const size_t N  = std::max(nx, ny);

    // scaleopt (case-insensitive). Previously accepted-and-ignored.
    std::string opt = scaleopt;
    for (char &ch : opt) if (ch >= 'A' && ch <= 'Z') ch = char(ch + 32);

    if (opt == "biased") {
        const double inv = (N > 0) ? 1.0 / static_cast<double>(N) : 0.0;
        for (size_t i = 0; i < nc; ++i) cd[i] *= inv;
    } else if (opt == "unbiased") {
        for (size_t i = 0; i < nc; ++i) {
            const double div = static_cast<double>(N) - std::abs(ld[i]);
            cd[i] = (div > 0.0) ? cd[i] / div : 0.0;
        }
    } else if (opt == "coeff" || opt == "normalized") {
        // Normalize so an autocorrelation has 1.0 at lag 0:
        // divide by sqrt(Rxx(0) * Ryy(0)) = sqrt(sum x^2 * sum y^2).
        double c0x = 0.0, c0y = 0.0;
        const double *xd = x.doubleData();
        const double *yd = y.doubleData();
        for (size_t i = 0; i < nx; ++i) c0x += xd[i] * xd[i];
        for (size_t i = 0; i < ny; ++i) c0y += yd[i] * yd[i];
        const double denom = std::sqrt(c0x * c0y);
        if (denom > 0.0)
            for (size_t i = 0; i < nc; ++i) cd[i] /= denom;
    } else if (!(opt.empty() || opt == "none")) {
        throw Error("xcorr: scaleopt must be 'none', 'biased', 'unbiased', or 'coeff'",
                     0, 0, "xcorr", "", "numkit:xcorr:badScaleopt");
    }

    // maxlag crop (or zero-pad) about lag 0 at index fullMaxLag.
    const int fullMaxLag = (N > 0) ? static_cast<int>(N) - 1 : 0;
    if (maxlag < 0) maxlag = fullMaxLag;
    if (maxlag != fullMaxLag) {
        const int center0 = fullMaxLag;
        const int outLen = 2 * maxlag + 1;
        Value cOut = Value::matrix(1, outLen, ValueType::DOUBLE, mr);
        Value lOut = Value::matrix(1, outLen, ValueType::DOUBLE, mr);
        double *co = cOut.doubleDataMut();
        double *lo = lOut.doubleDataMut();
        for (int m = -maxlag; m <= maxlag; ++m) {
            const int src = center0 + m;
            const int dst = m + maxlag;
            lo[dst] = static_cast<double>(m);
            co[dst] = (src >= 0 && src < static_cast<int>(nc)) ? cd[src] : 0.0;
        }
        cfull = std::move(cOut);
        lagsfull = std::move(lOut);
    }

    outs[0] = std::move(cfull);
    if (nargout > 1) outs[1] = std::move(lagsfull);
}

void xcov_reg(Span<const Value> args, size_t nargout, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "xcov");
    auto *mr = ctx.engine->resource();
    const Value &x = args[0];
    Value y = x;                 // default: auto-covariance
    int maxlag = -1;             // -1 => full
    std::string scaleopt = "none";

    // MATLAB disambiguation: a string arg is scaleopt; a scalar numeric
    // arg in the y-slot is maxlag (auto-cov); a vector numeric is y.
    size_t idx = 1;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString()) {
            scaleopt = args[1].toString();
        } else if (args[1].numel() == 1) {
            maxlag = std::abs(argcheck::expectedInteger(
                argcheck::expectedReal(args[1], 2, "maxlag", "xcov"),
                2, "maxlag", "xcov"));
        } else {
            y = args[1];
        }
        idx = 2;
    }
    // Trailing args: numeric => maxlag, string => scaleopt.
    for (; idx < args.size(); ++idx) {
        if (args[idx].isEmpty()) continue;
        if (args[idx].isChar() || args[idx].isString())
            scaleopt = args[idx].toString();
        else
            maxlag = std::abs(argcheck::expectedInteger(
                argcheck::expectedReal(args[idx], int(idx) + 1, "maxlag", "xcov"),
                int(idx) + 1, "maxlag", "xcov"));
    }

    auto result = xcov(x, y, maxlag, scaleopt, mr);
    outs[0] = std::move(std::get<0>(result));
    if (nargout > 1) outs[1] = std::move(std::get<1>(result));
}

void conv2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "conv2");
    std::string shape = "full";
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (args[2].isChar() || args[2].isString()) {
            shape = args[2].toString();
        } else {
            // conv2(u, v, A[, shape]): separable outer-product-kernel
            // form â€” not implemented (bugs/opened/signal/conv2-separable-form.md).
            throw Error("conv2: the separable form conv2(u, v, A) is not "
                        "supported yet â€” build the kernel with "
                        "conv2(u, v) and convolve with that",
                        0, 0, "conv2", "", "numkit:conv2:separableUnsupported");
        }
    }
    outs[0] = conv2(args[0], args[1], shape, ctx.engine->resource());
}

void filter2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "filter2");
    std::string shape = "same";
    if (args.size() >= 3 && !args[2].isEmpty())
        shape = shapeArg(args[2], "filter2");
    outs[0] = filter2(args[0], args[1], shape, ctx.engine->resource());
}

void convn_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "convn");
    std::string shape = "full";
    if (args.size() >= 3 && !args[2].isEmpty())
        shape = shapeArg(args[2], "convn");
    outs[0] = convn(args[0], args[1], shape, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
