// toolboxes/signal/src/transforms/fft_reg.cpp
//
// CallContext register half of transforms/fft.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.

// czt_reg below references M_PI â€” define it before any <cmath> include (MSVC
// does not expose M_PI without _USE_MATH_DEFINES).
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <numkit/sys/plugin.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/value/argcheck.hpp>
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

namespace numkit::signal {

namespace detail {

void fft_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "fft");

    int n = -1;
    int dim = 0;   // 0 = auto (first non-singleton) â€” resolved in public fft()
    if (args.size() >= 2 && !args[1].isEmpty())
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "n", "fft"), 2, "n", "fft");
    if (args.size() >= 3)
        dim = argcheck::dimArg(args[2], "fft");

    outs[0] = fft(args[0], n, dim, ctx.engine->resource());
}

void ifft_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "ifft");

    // A trailing 'symmetric'/'nonsymmetric' string flag (MATLAB): forces a
    // real result by treating the input as conjugate-symmetric. Strip it
    // before parsing the numeric n / dim args.
    std::size_t nargs = args.size();
    bool symmetric = false;
    if (nargs >= 2 && (args[nargs - 1].isChar() || args[nargs - 1].isString())) {
        std::string flag = args[nargs - 1].toString();
        for (char &c : flag) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (flag == "symmetric")
            symmetric = true;
        else if (flag == "nonsymmetric")
            symmetric = false;
        else
            throw Error("ifft: unknown option '" + args[nargs - 1].toString() +
                            "' (expected 'symmetric' or 'nonsymmetric')",
                         0, 0, "ifft", "", "numkit:ifft:badOption");
        --nargs;   // consume the flag
    }

    int n = -1;
    int dim = 0;   // 0 = auto (first non-singleton) â€” resolved in public fft()
    if (nargs >= 2 && !args[1].isEmpty())
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "n", "ifft"), 2, "n", "ifft");
    if (nargs >= 3)
        dim = argcheck::dimArg(args[2], "ifft");

    auto *mr = ctx.engine->resource();
    outs[0] = symmetric ? ifftSymmetric(args[0], n, dim, mr)
                        : ifft(args[0], n, dim, mr);
}

void fft2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "fft2");
    int m = -1, n = -1;
    if (args.size() >= 2 && !args[1].isEmpty())
        m = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "m", "fft2"), 2, "m", "fft2");
    if (args.size() >= 3 && !args[2].isEmpty())
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "n", "fft2"), 3, "n", "fft2");
    outs[0] = fft2(args[0], m, n, ctx.engine->resource());
}

void ifft2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "ifft2");

    // A trailing 'symmetric'/'nonsymmetric' string flag (MATLAB) forces a
    // real result by treating X as conjugate-symmetric. Strip it first.
    std::size_t nargs = args.size();
    bool symmetric = false;
    if (nargs >= 2 && (args[nargs - 1].isChar() || args[nargs - 1].isString())) {
        std::string flag = args[nargs - 1].toString();
        for (char &c : flag) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (flag == "symmetric")
            symmetric = true;
        else if (flag == "nonsymmetric")
            symmetric = false;
        else
            throw Error("ifft2: unknown option '" + args[nargs - 1].toString() +
                            "' (expected 'symmetric' or 'nonsymmetric')",
                         0, 0, "ifft2", "", "numkit:ifft2:badOption");
        --nargs;
    }

    int m = -1, n = -1;
    if (nargs >= 2 && !args[1].isEmpty())
        m = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "m", "ifft2"), 2, "m", "ifft2");
    if (nargs >= 3 && !args[2].isEmpty())
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "n", "ifft2"), 3, "n", "ifft2");

    auto *mr = ctx.engine->resource();
    if (symmetric) {
        if (m > 0 || n > 0)
            throw Error("ifft2: the 'symmetric' option with explicit size args "
                        "(m, n) is not supported in this revision",
                         0, 0, "ifft2", "", "numkit:ifft2:symmetricResize");
        outs[0] = ifft2Symmetric(args[0], mr);
        return;
    }
    outs[0] = ifft2(args[0], m, n, mr);
}

void interpft_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "interpft");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "interpft"), 2, "n", "interpft");
    int dim = 0;
    if (args.size() >= 3) dim = argcheck::dimArg(args[2], "interpft");
    outs[0] = interpft(args[0], n, dim, ctx.engine->resource());
}

// Shared helper for fftn_reg / ifftn_reg: unpack the optional `sz`
// vector into a std::vector<size_t>. MATLAB accepts an empty / missing
// 2nd arg.
static void extractSizeArg(const Value &v, std::vector<std::size_t> &dst)
{
    const std::size_t n = v.numel();
    dst.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double d = v.elemAsDouble(i);
        if (d <= 0 || d != std::floor(d))
            throw Error("fftn: size entries must be positive integers",
                         0, 0, "fftn", "", "numkit:fftn:badSize");
        dst[i] = static_cast<std::size_t>(d);
    }
}

void fftn_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "fftn");
    std::vector<std::size_t> sz;
    if (args.size() >= 2 && !args[1].isEmpty())
        extractSizeArg(args[1], sz);
    outs[0] = fftn(args[0], Span<const std::size_t>(sz.data(), sz.size()), ctx.engine->resource());
}

void ifftn_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "ifftn");
    std::vector<std::size_t> sz;
    if (args.size() >= 2 && !args[1].isEmpty())
        extractSizeArg(args[1], sz);
    outs[0] = ifftn(args[0], Span<const std::size_t>(sz.data(), sz.size()), ctx.engine->resource());
}

void czt_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
             CallContext &ctx)
{
    using Cd = std::complex<double>;
    argcheck::checkNargin(args.size(), 1, 4, "czt");
    const Value &x = args[0];

    // MATLAB: czt([]) returns empty without complaining about defaults.
    if (x.isEmpty()) {
        outs[0] = czt(x, 0, Cd(1, 0), Cd(1, 0), ctx.engine->resource());
        return;
    }

    // MATLAB defaults: m = length(x), w = exp(-2Ï€Â·j/m), a = 1.
    int m = static_cast<int>(x.numel());
    if (args.size() >= 2 && !args[1].isEmpty())
        m = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "m", "czt"), 2, "m", "czt");
    if (m <= 0)
        throw Error("czt: m must be a positive integer",
                     0, 0, "czt", "", "numkit:czt:badM");

    Cd w(std::cos(-2.0 * M_PI / m), std::sin(-2.0 * M_PI / m));
    if (args.size() >= 3 && !args[2].isEmpty()) {
        const Value &wv = args[2];
        if (wv.numel() != 1)
            throw Error("czt: input 3 (W) must be a scalar",
                         0, 0, "czt", "", "numkit:czt:wNotScalar");
        if (wv.isComplex()) w = wv.complexData()[0];
        else                w = Cd(wv.toScalar(), 0.0);
    }

    Cd a(1.0, 0.0);
    if (args.size() >= 4 && !args[3].isEmpty()) {
        const Value &av = args[3];
        if (av.numel() != 1)
            throw Error("czt: input 4 (A) must be a scalar",
                         0, 0, "czt", "", "numkit:czt:aNotScalar");
        if (av.isComplex()) a = av.complexData()[0];
        else                a = Cd(av.toScalar(), 0.0);
    }

    outs[0] = czt(x, m, w, a, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
