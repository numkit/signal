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
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/string.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <numkit/sys/plugin.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

void fft_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "fft");

    int n = -1;
    int dim = 0;   // 0 = auto (first non-singleton) â€” resolved in public fft()
    if (args.size() >= 2 && !args[1].isEmpty())
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "n", "fft"), 2, "n", "fft");
    if (args.size() >= 3)
        dim = argcheck::dimArg(args[2], "fft");

    outs[0] = fft(args[0], n, dim, ctx.engine->allocator());
}

void ifft_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "ifft");

    // A trailing 'symmetric'/'nonsymmetric' string flag (MATLAB): forces a
    // real result by treating the input as conjugate-symmetric. Strip it
    // before parsing the numeric n / dim args.
    keel::usize nargs = args.size();
    bool symmetric = false;
    if (nargs >= 2 && (args[nargs - 1].isChar() || args[nargs - 1].isString())) {
        keel::String rawFlag = args[nargs - 1].toString();
        keel::String flag = rawFlag.clone();
        for (char &c : flag) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (flag == "symmetric")
            symmetric = true;
        else if (flag == "nonsymmetric")
            symmetric = false;
        else
            throw Error(keel::format("ifft: unknown option '{}' (expected 'symmetric' or 'nonsymmetric')", rawFlag),
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

    auto *mr = ctx.engine->allocator();
    outs[0] = symmetric ? ifftSymmetric(args[0], n, dim, mr)
                        : ifft(args[0], n, dim, mr);
}

void fft2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "fft2");
    if (args.size() == 2)
        throw Error("fft2: requires 1 or 3 arguments (got 2 — specify both m and n)",
                    0, 0, "fft2", "", "numkit:fft2:nargin");
    int m = -1, n = -1;
    if (args.size() >= 2 && !args[1].isEmpty())
        m = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "m", "fft2"), 2, "m", "fft2");
    if (args.size() >= 3 && !args[2].isEmpty())
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "n", "fft2"), 3, "n", "fft2");
    outs[0] = fft2(args[0], m, n, ctx.engine->allocator());
}

void ifft2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "ifft2");

    // A trailing 'symmetric'/'nonsymmetric' string flag (MATLAB) forces a
    // real result by treating X as conjugate-symmetric. Strip it first.
    keel::usize nargs = args.size();
    bool symmetric = false;
    if (nargs >= 2 && (args[nargs - 1].isChar() || args[nargs - 1].isString())) {
        keel::String rawFlag = args[nargs - 1].toString();
        keel::String flag = rawFlag.clone();
        for (char &c : flag) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (flag == "symmetric")
            symmetric = true;
        else if (flag == "nonsymmetric")
            symmetric = false;
        else
            throw Error(keel::format("ifft2: unknown option '{}' (expected 'symmetric' or 'nonsymmetric')", rawFlag),
                         0, 0, "ifft2", "", "numkit:ifft2:badOption");
        --nargs;
    }

    if (nargs == 2)
        throw Error("ifft2: requires 1 or 3 numeric arguments (got 2 — specify both m and n)",
                    0, 0, "ifft2", "", "numkit:ifft2:nargin");

    int m = -1, n = -1;
    if (nargs >= 2 && !args[1].isEmpty())
        m = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "m", "ifft2"), 2, "m", "ifft2");
    if (nargs >= 3 && !args[2].isEmpty())
        n = argcheck::expectedInteger(
            argcheck::expectedReal(args[2], 3, "n", "ifft2"), 3, "n", "ifft2");

    auto *mr = ctx.engine->allocator();
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

void interpft_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "interpft");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "interpft"), 2, "n", "interpft");
    int dim = 0;
    if (args.size() >= 3) dim = argcheck::dimArg(args[2], "interpft");
    outs[0] = interpft(args[0], n, dim, ctx.engine->allocator());
}

// Shared helper for fftn_reg / ifftn_reg: unpack the optional `sz`
// vector into a keel::Array<keel::usize>. MATLAB accepts an empty / missing
// 2nd arg.
static void extractSizeArg(const Value &v, keel::Array<keel::usize> &dst)
{
    const keel::usize n = v.numel();
    dst.resize(n);
    for (keel::usize i = 0; i < n; ++i) {
        const double d = v.elemAsDouble(i);
        if (d <= 0 || d != std::floor(d))
            throw Error("fftn: size entries must be positive integers",
                         0, 0, "fftn", "", "numkit:fftn:badSize");
        dst[i] = static_cast<keel::usize>(d);
    }
}

void fftn_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "fftn");
    keel::Array<keel::usize> sz(ctx.engine->allocator());
    if (args.size() >= 2 && !args[1].isEmpty())
        extractSizeArg(args[1], sz);
    outs[0] = fftn(args[0], keel::ArrayRef<keel::usize>(sz.data(), sz.size()), ctx.engine->allocator());
}

void ifftn_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "ifftn");
    keel::Array<keel::usize> sz(ctx.engine->allocator());
    if (args.size() >= 2 && !args[1].isEmpty())
        extractSizeArg(args[1], sz);
    outs[0] = ifftn(args[0], keel::ArrayRef<keel::usize>(sz.data(), sz.size()), ctx.engine->allocator());
}

void czt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
             CallContext &ctx)
{
argcheck::checkNargin(args.size(), 1, 4, "czt");
    const Value &x = args[0];

    // MATLAB: czt([]) returns empty without complaining about defaults.
    if (x.isEmpty()) {
        outs[0] = czt(x, 0, keel::c64(1, 0), keel::c64(1, 0), ctx.engine->allocator());
        return;
    }

    // Reference defaults: m = length(x) for vectors, size(x, 1) for matrices/N-D,
    // w = exp(-2*pi*j/m), a = 1.
    const bool isRowVec = (x.dims().ndim() == 2 && x.dims().rows() == 1 && x.dims().cols() > 1);
    int m = static_cast<int>(isRowVec ? x.dims().cols() : x.dims().rows());
    if (args.size() >= 2 && !args[1].isEmpty())
        m = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "m", "czt"), 2, "m", "czt");
    if (m <= 0)
        throw Error("czt: m must be a positive integer",
                     0, 0, "czt", "", "numkit:czt:badM");

    keel::c64 w(std::cos(-2.0 * M_PI / m), std::sin(-2.0 * M_PI / m));
    if (args.size() >= 3 && !args[2].isEmpty()) {
        const Value &wv = args[2];
        if (wv.numel() != 1)
            throw Error("czt: input 3 (W) must be a scalar",
                         0, 0, "czt", "", "numkit:czt:wNotScalar");
        if (wv.isComplex()) w = wv.complexData()[0];
        else                w = keel::c64(wv.toScalar(), 0.0);
    }

    keel::c64 a(1.0, 0.0);
    if (args.size() >= 4 && !args[3].isEmpty()) {
        const Value &av = args[3];
        if (av.numel() != 1)
            throw Error("czt: input 4 (A) must be a scalar",
                         0, 0, "czt", "", "numkit:czt:aNotScalar");
        if (av.isComplex()) a = av.complexData()[0];
        else                a = keel::c64(av.toScalar(), 0.0);
    }

    outs[0] = czt(x, m, w, a, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
