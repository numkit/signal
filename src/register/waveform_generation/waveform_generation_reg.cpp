// toolboxes/signal/src/waveform_generation/waveform_generation_reg.cpp
//
// CallContext register half of waveform_generation/waveform_generation.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/digital_filtering/filter.hpp>        // filtfilt
#include <numkit/signal/filter_design/filter_design.hpp>     // butter
#include <numkit/signal/transforms/hilbert.hpp>              // hilbert
#include <numkit/signal/waveform_generation/waveform_generation.hpp>
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

// Fc/Fs marshalling for modulate/demod: real scalars. expectedReal alone
// would pass a CHAR scalar through as its ASCII code â€” the reference
// rejects non-numeric scalars, so gate the class first.
static double carrierArg(const Value &v, int argNum, const char *name,
                         const char *fn)
{
    if (v.isChar() || v.isString() || v.isCell() || v.isStruct())
        throw Error(keel::format("{}: input {} ({}) must be a real scalar (argument is not numeric)", fn, argNum, name),
                    0, 0, fn, "",
                    keel::format("numkit:{}:{}NotReal", fn, name));
    return argcheck::expectedReal(v, argNum, name, fn);
}

void demod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "demod");
    const double Fc = carrierArg(args[1], 2, "Fc", "demod");
    const double Fs = carrierArg(args[2], 3, "Fs", "demod");
    if (!args[3].isChar() && !args[3].isString())
        throw Error("demod: method must be a string",
                    0, 0, "demod", "", "numkit:demod:BadMethodType");
    keel::String method = args[3].toString();
    const Value &opt = (args.size() >= 5) ? args[4] : Value::Empty;
    outs[0] = demod(args[0], Fc, Fs, method, opt, ctx.engine->allocator());
}

void modulate_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "modulate");
    const double Fc = carrierArg(args[1], 2, "Fc", "modulate");
    const double Fs = carrierArg(args[2], 3, "Fs", "modulate");
    if (!args[3].isChar() && !args[3].isString())
        throw Error("modulate: method must be a string",
                    0, 0, "modulate", "", "numkit:modulate:BadMethodType");
    keel::String method = args[3].toString();
    const Value &opt = (args.size() >= 5) ? args[4] : Value::Empty;
    outs[0] = modulate(args[0], Fc, Fs, method, opt, ctx.engine->allocator());
}

void vco_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "vco");
    auto *mr = ctx.engine->allocator();
    double fs = 1.0;
    if (args.size() >= 3 && !args[2].isEmpty()) fs = args[2].toScalar();
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &rng = args[1];
        if (rng.numel() == 1)
            outs[0] = vco(args[0], rng.toScalar(), fs, mr);
        else if (rng.numel() == 2)
            outs[0] = vco(args[0], rng.elemAsDouble(0), rng.elemAsDouble(1),
                          fs, mr);
        else
            throw Error("vco: range must be scalar Fc or [Fmin Fmax]",
                        0, 0, "vco", "", "numkit:vco:BadRange");
    } else {
        outs[0] = vco(args[0], fs / 4.0, fs, mr);
    }
}

void rectpuls_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "rectpuls");
    const double w = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    outs[0] = rectpuls(args[0], w, ctx.engine->allocator());
}

void tripuls_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "tripuls");
    const double w = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    outs[0] = tripuls(args[0], w, ctx.engine->allocator());
}

void gauspuls_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 5, "gauspuls");
    const double fc = (args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : 1000.0;
    const double bw = (args.size() >= 3 && !args[2].isEmpty()) ? args[2].toScalar() : 0.5;
    outs[0] = gauspuls(args[0], fc, bw, ctx.engine->allocator());
}

void pulstran_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "pulstran");  // variadic by design: trailing pulse-function parameters
    keel::Allocator *mr = ctx.engine->allocator();
    if (args[2].isFuncHandle()) {
        const auto &handle = args[2];
        auto cb = [&ctx, &handle](keel::ArrayRef<Value> ar, keel::MutArrayRef<Value> ou,
                                   keel::Allocator * /*mr*/) {
            auto r = ctx.engine->callFunctionHandleMulti(handle, ar, ou.size());
            for (keel::usize i = 0; i < ou.size() && i < r.size(); ++i)
                ou[i] = std::move(r[i]);
        };
        // Extract t and d into double buffers; library takes Span.
        const Value &tv = args[0];
        const Value &dv = args[1];
        const keel::usize nt = tv.numel();
        const keel::usize nd = dv.numel();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<double> tbuf(nt, scratch.allocator());
        keel::Array<double> dbuf(nd, scratch.allocator());
        for (keel::usize i = 0; i < nt; ++i) tbuf[i] = tv.elemAsDouble(i);
        for (keel::usize i = 0; i < nd; ++i) dbuf[i] = dv.elemAsDouble(i);

        Value r = pulstranHandle(
            keel::ArrayRef<double>(tbuf.data(), nt),
            keel::ArrayRef<double>(dbuf.data(), nd),
            cb, mr);

        // MATLAB convention: output shape mirrors t's shape.
        // Library returns a column; if t was a row, reshape.
        if (tv.dims().rows() == 1 && tv.dims().cols() >= 1) {
            Value row = Value::matrix(1, nt, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < nt; ++i)
                row.doubleDataMut()[i] = r.doubleData()[i];
            r = std::move(row);
        }
        outs[0] = std::move(r);
        return;
    }
    if (!args[2].isChar() && !args[2].isString())
        throw Error("pulstran: 3rd argument must be a string name or a function handle",
                     0, 0, "pulstran", "", "numkit:pulstran:fnType");
    const keel::String fnName = args[2].toString();
    const double fcOrW = (args.size() >= 4) ? args[3].toScalar() : 1.0;
    const double bw    = (args.size() >= 5) ? args[4].toScalar() : 0.5;
    outs[0] = pulstran(args[0], args[1], fnName, fcOrW, bw, mr);
}

void square_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "square");
    const double duty = (args.size() >= 2) ? args[1].toScalar() : 50.0;
    outs[0] = square(args[0], duty, ctx.engine->allocator());
}

void sawtooth_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sawtooth");
    const double width = (args.size() >= 2) ? args[1].toScalar() : 1.0;
    outs[0] = sawtooth(args[0], width, ctx.engine->allocator());
}

void sinc_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "sinc");
    outs[0] = sinc(args[0], ctx.engine->allocator());
}

void gmonopuls_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "gmonopuls");
    const double fc = (args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : 1000.0;
    outs[0] = gmonopuls(args[0], fc, ctx.engine->allocator());
}

void diric_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "diric");
    const int n = argcheck::expectedInteger(argcheck::expectedReal(args[1], 2, "n", "diric"), 2, "n", "diric");
    outs[0] = diric(args[0], n, ctx.engine->allocator());
}

void chirp_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 8, "chirp");
    const double f0 = (args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : 0.0;
    const double t1 = (args.size() >= 3 && !args[2].isEmpty()) ? args[2].toScalar() : 1.0;
    const double f1 = (args.size() >= 4 && !args[3].isEmpty()) ? args[3].toScalar() : 100.0;
    keel::String method = keel::String::from("linear");
    if (args.size() >= 5) {
        if (!args[4].isChar() && !args[4].isString())
            throw Error("chirp: method must be a string",
                         0, 0, "chirp", "", "numkit:chirp:badMethodType");
        method = args[4].toString();
    }
    outs[0] = chirp(args[0], f0, t1, f1, method, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
