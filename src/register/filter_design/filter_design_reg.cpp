// toolboxes/signal/src/filter_design/filter_design_reg.cpp
//
// CallContext register half of filter_design/filter_design.cpp (Phase 2b compute/register split).
// The detail{{}} *_reg fns were interleaved with compute in the original
// TU; collected here verbatim (reg-side arg parsers nested in their anon
// namespaces ride along). See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/signal/internal/shims/specfun.hpp>
#include <numkit/signal/filter_design/filter_design.hpp>
#include <numkit/signal/filter_implementation/conversions_extras.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::signal {

namespace detail {

namespace {

// FIR-family order extraction â€” the Value-boundary half of the order
// contract (a fractional order is only expressible as a Value; the
// positive/domain half lives in each compute's fir1_argcheck).
int firOrder(const Value &v, int argNum, const char *name, const char *fn)
{
    const double n = argcheck::expectedReal(v, argNum, name, fn);
    if (n != std::floor(n))
        throw Error(std::string(fn) + ": filter order " + name
                        + " must be a positive integer (got "
                        + argcheck::detail::num(n) + ").",
                    0, 0, fn, "", std::string("numkit:") + fn + ":orderNotInteger");
    return static_cast<int>(n);
}

} // namespace

void fir1_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 6, "fir1");
    const int N = firOrder(args[0], 1, "n", "fir1");
    if (args[1].isScalar() && args.size() <= 3 && (args.size() < 3 || args[2].isChar() || args[2].isString())) {
        const double Wn = argcheck::expectedReal(args[1], 2, "Wn", "fir1");
        std::string type = "low";
        if (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
            type = args[2].toString();
        outs[0] = fir1(N, Wn, type, ctx.engine->resource());
        return;
    }
    std::string type;
    Value window = Value::Empty;
    bool scale = true;
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            std::string s = args[i].toString();
            std::string sl = s;
            for (char &ch : sl)
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (sl == "noscale")
                scale = false;
            else if (sl == "scale")
                scale = true;
            else
                type = s;
        } else if (args[i].isNumeric()) {
            window = args[i];
        }
    }
    outs[0] = fir1(N, args[1], type, window, scale, ctx.engine->resource());
}

void firls_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "firls");
    const int N = firOrder(args[0], 1, "N", "firls");
    Value W = Value::Empty;
    std::string ftype;
    for (std::size_t i = 3; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString())
            ftype = args[i].toString();
        else
            W = args[i];
    }
    outs[0] = firls(N, args[1], args[2], W, ftype, ctx.engine->resource());
}

void fir2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "fir2");
    const int N = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[0], 1, "N", "fir2"), 1, "N", "fir2");

    // Optional trailing arguments fir2(N,F,A[,npt][,lap][,window]):
    // scalars are npt then lap (in order); a vector argument is the
    // output window.
    Fir2Options opts;
    int scalarsSeen = 0;
    for (size_t i = 3; i < args.size(); ++i) {
        const Value &a = args[i];
        if (a.numel() == 1) {
            if (scalarsSeen == 0)
                opts.npt = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(a, 0, "npt", "fir2"), 0, "npt", "fir2");
            else if (scalarsSeen == 1)
                opts.lap = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(a, 0, "lap", "fir2"), 0, "lap", "fir2");
            else
                throw Error("fir2: too many scalar arguments",
                            0, 0, "fir2", "", "numkit:fir2:nargin");
            ++scalarsSeen;
        } else {
            opts.window = a;
        }
    }
    outs[0] = fir2(N, args[1], args[2], opts, ctx.engine->resource());
}

} // namespace detail

namespace detail {

void cell2sos_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "cell2sos");
    auto [S, g] = cell2sos(args[0], ctx.engine->resource());
    outs[0] = std::move(S);
    if (nargout >= 2 && outs.size() >= 2) outs[1] = std::move(g);
}

} // namespace detail

namespace detail {

void firpm_reg(Span<const Value> args, std::size_t nargout, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "firpm");
    const int N = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[0], 1, "N", "firpm"), 1, "N", "firpm");

    // Trailing args: weights vector (numeric) or ftype string. MATLAB
    // accepts them in either order (W [, ftype] or ftype only).
    Value W;
    std::string ftype;
    for (std::size_t i = 3; i < args.size(); ++i) {
        if (args[i].isCell())
            throw Error("firpm: cell-form lgrid argument deferred",
                        0, 0, "firpm", "", "numkit:firpm:unsupportedLgrid");
        if (args[i].isChar())
            ftype = args[i].toString();
        else
            W = args[i];
    }

    auto [b, err] = firpm(N, args[1], args[2], W, ftype,
                          ctx.engine->resource());
    outs[0] = std::move(b);
    if (nargout >= 2 && outs.size() >= 2)
        outs[1] = Value::scalar(err, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
