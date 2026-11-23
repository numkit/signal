// toolboxes/signal/src/convolution/convolution_reg.cpp
//
// Register half of the signal convolution builtins: the CallContext wrappers
// delegating to the engine-free compute in convolution.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/convolution/convolution.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <keel/tuple.h>

namespace numkit::signal {

namespace detail {

// Shape options must be strings (the reference rejects a numeric shape
// argument with unknownShapeParameter). conv2's numeric third argument
// is instead the SEPARABLE form conv2(u, v, A) â€” unimplemented, voiced
// honestly instead of being silently ignored.
static keel::String shapeArg(const Value &v, keel::StringRef fn)
{
    if (v.isChar() || v.isString())
        return v.toString();
    throw Error(keel::format("{}: shape option must be 'full', 'same', or 'valid' (got a non-string argument)", fn),
                0, 0, fn, "",
                keel::format("numkit:{}:badShape", fn));
}

void conv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "conv");

    keel::String shape = keel::String::from("full");
    if (args.size() >= 3 && !args[2].isEmpty())
        shape = shapeArg(args[2], "conv");

    outs[0] = conv(args[0], args[1], shape, ctx.engine->allocator());
}

void deconv_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "deconv");

    auto [q, r] = deconv(args[0], args[1], ctx.engine->allocator());
    outs[0] = keel::move(q);
    if (nargout > 1)
        outs[1] = keel::move(r);
}

void xcorr_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "xcorr");
    auto *mr = ctx.engine->allocator();
    const Value &x = args[0];
    Value y = x;                 // default: autocorrelation
    bool haveY = false;
    keel::i32 maxlag = -1;             // -1 => full
    keel::String scaleopt = keel::String::from("none");

    // Disambiguate (MATLAB): a string is scaleopt; a scalar numeric in the
    // y-slot is maxlag (autocorr); a vector numeric is y. Trailing args:
    // numeric => maxlag, string => scaleopt.
    keel::usize idx = 1;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString())
            scaleopt = args[1].toString();
        else if (args[1].numel() == 1)
            maxlag = keel::abs(argcheck::expectedInteger(
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
            maxlag = keel::abs(argcheck::expectedInteger(
                argcheck::expectedReal(args[idx], keel::i32(idx) + 1, "maxlag", "xcorr"),
                keel::i32(idx) + 1, "maxlag", "xcorr"));
    }

    auto [cfull, lagsfull] = haveY ? xcorr(x, y, mr) : xcorr(x, mr);
    const keel::usize nc = cfull.numel();
    const bool isCplx = cfull.isComplex();
    keel::f64 *cd = isCplx ? nullptr : cfull.doubleDataMut();
    keel::c64 *ccd = isCplx ? cfull.complexDataMut() : nullptr;
    const keel::f64 *ld = lagsfull.doubleData();
    const keel::usize nx = x.numel(), ny = y.numel();
    const keel::usize N  = keel::max(nx, ny);

    // scaleopt (case-insensitive). Previously accepted-and-ignored.
    keel::String opt = scaleopt.clone();
    for (char &ch : opt) if (ch >= 'A' && ch <= 'Z') ch = char(ch + 32);

    auto scaleElem = [&](keel::usize i, keel::f64 s) {
        if (isCplx) ccd[i] *= s;
        else cd[i] *= s;
    };

    if (opt == "biased") {
        const keel::f64 inv = (N > 0) ? 1.0 / keel::as<keel::f64>(N) : 0.0;
        for (keel::usize i = 0; i < nc; ++i) scaleElem(i, inv);
    } else if (opt == "unbiased") {
        for (keel::usize i = 0; i < nc; ++i) {
            const keel::f64 div = keel::as<keel::f64>(N) - keel::abs(ld[i]);
            scaleElem(i, (div > 0.0) ? (1.0 / div) : 0.0);
        }
    } else if (opt == "coeff" || opt == "normalized") {
        keel::f64 c0x = 0.0, c0y = 0.0;
        for (keel::usize i = 0; i < nx; ++i) {
            if (x.isComplex()) c0x += keel::norm(x.complexData()[i]);
            else { keel::f64 v = x.elemAsDouble(i); c0x += v * v; }
        }
        for (keel::usize i = 0; i < ny; ++i) {
            if (y.isComplex()) c0y += keel::norm(y.complexData()[i]);
            else { keel::f64 v = y.elemAsDouble(i); c0y += v * v; }
        }
        const keel::f64 denom = keel::sqrt(c0x * c0y);
        if (denom > 0.0)
            for (keel::usize i = 0; i < nc; ++i) scaleElem(i, 1.0 / denom);
    } else if (!(opt.empty() || opt == "none")) {
        throw Error("xcorr: scaleopt must be 'none', 'biased', 'unbiased', or 'coeff'",
                     0, 0, "xcorr", "", "numkit:xcorr:badScaleopt");
    }

    // maxlag crop (or zero-pad) about lag 0 at index fullMaxLag.
    const keel::i32 fullMaxLag = (N > 0) ? keel::as<keel::i32>(N) - 1 : 0;
    if (maxlag < 0) maxlag = fullMaxLag;
    if (maxlag != fullMaxLag) {
        const keel::i32 center0 = fullMaxLag;
        const keel::i32 outLen = 2 * maxlag + 1;
        const bool colOut = (x.dims().rows() > 1 && x.dims().cols() == 1);
        ValueType vt = isCplx ? ValueType::COMPLEX : ValueType::DOUBLE;
        Value cOut = colOut ? Value::matrix(outLen, 1, vt, mr)
                            : Value::matrix(1, outLen, vt, mr);
        Value lOut = Value::matrix(1, outLen, ValueType::DOUBLE, mr);
        keel::f64 *lo = lOut.doubleDataMut();
        for (keel::i32 m = -maxlag; m <= maxlag; ++m) {
            const keel::i32 src = center0 + m;
            const keel::i32 dst = m + maxlag;
            lo[dst] = keel::as<keel::f64>(m);
            if (isCplx) {
                cOut.complexDataMut()[dst] = (src >= 0 && src < keel::as<keel::i32>(nc)) ? ccd[src] : keel::c64(0.0, 0.0);
            } else {
                cOut.doubleDataMut()[dst] = (src >= 0 && src < keel::as<keel::i32>(nc)) ? cd[src] : 0.0;
            }
        }
        cfull = keel::move(cOut);
        lagsfull = keel::move(lOut);
    }

    outs[0] = keel::move(cfull);
    if (nargout > 1) outs[1] = keel::move(lagsfull);
}

void xcov_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "xcov");
    auto *mr = ctx.engine->allocator();
    const Value &x = args[0];
    Value y = x;                 // default: auto-covariance
    keel::i32 maxlag = -1;             // -1 => full
    keel::String scaleopt = keel::String::from("none");

    // MATLAB disambiguation: a string arg is scaleopt; a scalar numeric
    // arg in the y-slot is maxlag (auto-cov); a vector numeric is y.
    keel::usize idx = 1;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString()) {
            scaleopt = args[1].toString();
        } else if (args[1].numel() == 1) {
            maxlag = keel::abs(argcheck::expectedInteger(
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
            maxlag = keel::abs(argcheck::expectedInteger(
                argcheck::expectedReal(args[idx], keel::i32(idx) + 1, "maxlag", "xcov"),
                keel::i32(idx) + 1, "maxlag", "xcov"));
    }

    auto result = xcov(x, y, maxlag, scaleopt, mr);
    outs[0] = keel::move(keel::get<0>(result));
    if (nargout > 1) outs[1] = keel::move(keel::get<1>(result));
}

void conv2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "conv2");
    auto *mr = ctx.engine->allocator();
    keel::String shape = keel::String::from("full");
    if (args.size() >= 3 && !args[2].isEmpty() &&
        !(args[2].isChar() || args[2].isString())) {
        if ((!args[0].isEmpty() && !args[0].dims().isVector()) ||
            (!args[1].isEmpty() && !args[1].dims().isVector())) {
            throw Error("conv2: first two arguments u and v must be vectors in separable form conv2(u, v, A)",
                        0, 0, "conv2", "", "numkit:conv2:firstTwoInputsNotVectors");
        }
        if (args.size() >= 4 && !args[3].isEmpty()) {
            if (!args[3].isChar() && !args[3].isString())
                throw Error("conv2: shape option must be 'full', 'same', or 'valid'",
                            0, 0, "conv2", "", "numkit:conv2:badShape");
            shape = args[3].toString();
        }
        const keel::usize m = args[0].numel();
        const keel::usize n = args[1].numel();
        Value k = Value::matrix(m, n, ValueType::DOUBLE, mr);
        keel::f64 *kd = k.doubleDataMut();
        for (keel::usize c = 0; c < n; ++c) {
            const keel::f64 vc = args[1].elemAsDouble(c);
            for (keel::usize r = 0; r < m; ++r)
                kd[r + c * m] = args[0].elemAsDouble(r) * vc;
        }
        outs[0] = conv2(args[2], k, shape, mr);
        return;
    }
    if (args.size() >= 3 && !args[2].isEmpty()) {
        shape = args[2].toString();
    }
    outs[0] = conv2(args[0], args[1], shape, mr);
}

void filter2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "filter2");
    keel::String shape = keel::String::from("same");
    if (args.size() >= 3 && !args[2].isEmpty())
        shape = shapeArg(args[2], "filter2");
    outs[0] = filter2(args[0], args[1], shape, ctx.engine->allocator());
}

void convn_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "convn");
    keel::String shape = keel::String::from("full");
    if (args.size() >= 3 && !args[2].isEmpty())
        shape = shapeArg(args[2], "convn");
    outs[0] = convn(args[0], args[1], shape, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
