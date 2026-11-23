// toolboxes/signal/src/filter_design/filter_design_reg.cpp
//
// CallContext register half of filter_design/filter_design.cpp (Phase 2b compute/register split).
// The detail{{}} *_reg fns were interleaved with compute in the original
// TU; collected here verbatim (reg-side arg parsers nested in their anon
// namespaces ride along). See project_layering_refactor.
#include <keel/core.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/signal/internal/shims/specfun.hpp>
#include <numkit/signal/filter_design/filter_design.hpp>
#include <numkit/signal/filter_implementation/conversions_extras.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>

namespace numkit::signal {

namespace detail {

namespace {

// FIR-family order extraction â€” the Value-boundary half of the order
// contract (a fractional order is only expressible as a Value; the
// positive/domain half lives in each compute's fir1_argcheck).
keel::i32 firOrder(const Value &v, keel::i32 argNum, const char *name, const char *fn)
{
    const keel::f64 n = argcheck::expectedReal(v, argNum, name, fn);
    if (n != keel::floor(n)) {
        const auto numStr = argcheck::detail::num(n);
        throw Error(keel::format("{}: filter order {} must be a positive integer (got {}).", fn, name, numStr),
                    0, 0, fn, "", keel::format("numkit:{}:orderNotInteger", fn));
    }
    return keel::as<keel::i32>(n);
}

} // namespace

void fir1_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 6, "fir1");
    const keel::i32 N = firOrder(args[0], 1, "n", "fir1");
    if (args[1].isScalar() && args.size() <= 3 && (args.size() < 3 || args[2].isChar() || args[2].isString())) {
        const keel::f64 Wn = argcheck::expectedReal(args[1], 2, "Wn", "fir1");
        keel::String type = keel::String::from("low");
        if (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
            type = args[2].toString();
        outs[0] = fir1(N, Wn, type, ctx.engine->allocator());
        return;
    }
    keel::String type;
    Value window = Value::Empty;
    bool scale = true;
    for (keel::usize i = 2; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            keel::String s = args[i].toString();
            keel::String sl = s.clone();
            sl.to_ascii_lower_in_place();
            if (sl == "noscale")
                scale = false;
            else if (sl == "scale")
                scale = true;
            else
                type = keel::move(s);
        } else if (args[i].isNumeric()) {
            window = args[i];
        }
    }
    outs[0] = fir1(N, args[1], type, window, scale, ctx.engine->allocator());
}

void firls_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "firls");
    const keel::i32 N = firOrder(args[0], 1, "N", "firls");
    Value W = Value::Empty;
    keel::String ftype;
    for (keel::usize i = 3; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString())
            ftype = args[i].toString();
        else
            W = args[i];
    }
    outs[0] = firls(N, args[1], args[2], W, ftype, ctx.engine->allocator());
}

void fir2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "fir2");
    const keel::i32 N = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[0], 1, "N", "fir2"), 1, "N", "fir2");

    // Optional trailing arguments fir2(N,F,A[,npt][,lap][,window]):
    // scalars are npt then lap (in order); a vector argument is the
    // output window.
    Fir2Options opts;
    keel::i32 scalarsSeen = 0;
    for (keel::usize i = 3; i < args.size(); ++i) {
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
    outs[0] = fir2(N, args[1], args[2], opts, ctx.engine->allocator());
}

} // namespace detail

namespace detail {

void cell2sos_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "cell2sos");
    auto [S, g] = cell2sos(args[0], ctx.engine->allocator());
    outs[0] = keel::move(S);
    if (nargout >= 2 && outs.size() >= 2) outs[1] = keel::move(g);
}

} // namespace detail

namespace detail {

void firpm_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "firpm");
    const keel::i32 N = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[0], 1, "N", "firpm"), 1, "N", "firpm");

    // Trailing args: weights vector (numeric) or ftype string. MATLAB
    // accepts them in either order (W [, ftype] or ftype only).
    Value W;
    keel::String ftype;
    for (keel::usize i = 3; i < args.size(); ++i) {
        if (args[i].isCell())
            throw Error("firpm: cell-form lgrid argument deferred",
                        0, 0, "firpm", "", "numkit:firpm:unsupportedLgrid");
        if (args[i].isChar())
            ftype = args[i].toString();
        else
            W = args[i];
    }

    auto [b, err] = firpm(N, args[1], args[2], W, ftype,
                          ctx.engine->allocator());
    outs[0] = keel::move(b);
    if (nargout >= 2 && outs.size() >= 2)
        outs[1] = Value::scalar(err, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
