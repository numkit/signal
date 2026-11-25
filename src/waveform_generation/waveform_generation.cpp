// toolboxes/signal/src/waveform_generation/waveform_generation.cpp
//
// rectpuls, tripuls, gauspuls, pulstran (+ pulstranHandle), square,
// sawtooth, sinc, gmonopuls, diric, chirp, modulate, demod, vco
// (delegates numeric kernels to keel::signal).

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/signal/waveform_generation/waveform_generation.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

keel::Array<keel::f64> readReal(const Value &v, keel::Allocator *mr)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, mr);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void rectpuls_argcheck(keel::f64 w)
{
    if (w <= 0)
        throw Error("rectpuls: width w must be positive",
                     0, 0, "rectpuls", "", "numkit:rectpuls:badWidth");
}

void tripuls_argcheck(keel::f64 w)
{
    if (w <= 0)
        throw Error("tripuls: width w must be positive",
                     0, 0, "tripuls", "", "numkit:tripuls:badWidth");
}

void gauspuls_argcheck(keel::f64 fc, keel::f64 bw)
{
    if (fc <= 0 || bw <= 0)
        throw Error("gauspuls: fc and bw must be positive",
                     0, 0, "gauspuls", "", "numkit:gauspuls:badArg");
}

} // namespace

Value rectpuls(const Value &t, keel::f64 w, keel::Allocator *mr)
{
    rectpuls_argcheck(w);
    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::rectpuls(tv.as_ref(), w,
                                 keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value tripuls(const Value &t, keel::f64 w, keel::Allocator *mr)
{
    tripuls_argcheck(w);
    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::tripuls(tv.as_ref(), w,
                                keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value gauspuls(const Value &t, keel::f64 fc, keel::f64 bw, keel::Allocator *mr)
{
    gauspuls_argcheck(fc, bw);
    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::gauspuls(tv.as_ref(), fc, bw,
                                 keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value pulstranHandle(keel::ArrayRef<keel::f64> t, keel::ArrayRef<keel::f64> d,
                     FnHandle fn, keel::Allocator *mr)
{
    const keel::usize n  = t.size();
    const keel::usize nd = d.size();

    Value out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    if (n) keel::mem_zero(dst, n * sizeof(keel::f64));

    Value shifted = Value::matrix(1, n, ValueType::DOUBLE, mr);
    keel::f64 *sh = shifted.doubleDataMut();
    for (keel::usize k = 0; k < nd; ++k) {
        const keel::f64 dk = d[k];
        for (keel::usize i = 0; i < n; ++i) sh[i] = t[i] - dk;
        Value r;
        Value args[1] = { shifted };
        keel::ArrayRef<Value> ar(args, 1);
        keel::MutArrayRef<Value> ou(&r, 1);
        fn(ar, ou, mr);
        if (r.numel() != n)
            throw Error("pulstran: handle must return a vector of the same length as t",
                         0, 0, "pulstran", "", "numkit:pulstran:badHandleOutput");
        for (keel::usize i = 0; i < n; ++i) dst[i] += r.elemAsDouble(i);
    }
    return out;
}

Value pulstran(const Value &t, const Value &d, keel::StringRef fnName,
               keel::f64 fcOrW, keel::f64 bw, keel::Allocator *mr)
{
    keel::String lower;
    lower.reserve(fnName.size());
    for (char c : fnName) lower.push_back(keel::to_ascii_lower(c));

    keel::signal::PulstranKind kind;
    if (lower == "rectpuls") {
        kind = keel::signal::PulstranKind::kRectpuls;
    } else if (lower == "tripuls") {
        kind = keel::signal::PulstranKind::kTripuls;
    } else if (lower == "gauspuls") {
        kind = keel::signal::PulstranKind::kGauspuls;
    } else {
        throw Error(keel::format("pulstran: unsupported pulse function '{}' (built-ins: 'rectpuls'/'tripuls'/'gauspuls'). Custom handles need the engine callback API (planned).", fnName),
                     0, 0, "pulstran", "", "numkit:pulstran:fnUnsupported");
    }

    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    auto dv = readReal(d, scratch.allocator());
    (void)keel::signal::pulstran_builtin(
        tv.as_ref(), dv.as_ref(), kind, fcOrW, bw,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value square(const Value &t, keel::f64 duty, keel::Allocator *mr)
{
    if (duty < 0.0 || duty > 100.0)
        throw Error("square: duty cycle must be in [0, 100]",
                     0, 0, "square", "", "numkit:square:badDuty");
    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::square(tv.as_ref(), duty,
                               keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value sawtooth(const Value &t, keel::f64 width, keel::Allocator *mr)
{
    if (width < 0.0 || width > 1.0)
        throw Error("sawtooth: width must be in [0, 1]",
                     0, 0, "sawtooth", "", "numkit:sawtooth:badWidth");
    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::sawtooth(tv.as_ref(), width,
                                 keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value sinc(const Value &t, keel::Allocator *mr)
{
    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::sinc(tv.as_ref(),
                             keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value gmonopuls(const Value &t, keel::f64 fc, keel::Allocator *mr)
{
    if (fc <= 0)
        throw Error("gmonopuls: fc must be positive",
                     0, 0, "gmonopuls", "", "numkit:gmonopuls:badFc");
    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::gmonopuls(tv.as_ref(), fc,
                                  keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value diric(const Value &x, keel::i32 n, keel::Allocator *mr)
{
    if (n < 1)
        throw Error("diric: n must be a positive integer",
                     0, 0, "diric", "", "numkit:diric:badN");
    auto out = createLike(x, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readReal(x, scratch.allocator());
    (void)keel::signal::diric(xv.as_ref(), n,
                              keel::MutArrayRef<keel::f64>(out.doubleDataMut(), xv.size()));
    return out;
}

Value chirp(const Value &t, keel::f64 f0, keel::f64 t1, keel::f64 f1,
            keel::StringRef method, keel::f64 phi_deg, keel::StringRef quad_shape,
            keel::Allocator *mr)
{
    if (t1 <= 0)
        throw Error("chirp: t1 must be positive",
                     0, 0, "chirp", "", "numkit:chirp:badT1");

    keel::String m = keel::String::from(method);
    m.to_ascii_lower_in_place();
    keel::signal::ChirpMethod mode;
    if      (m.empty() || keel::StringRef("linear").starts_with(m) || m == "li")        mode = keel::signal::ChirpMethod::kLinear;
    else if (keel::StringRef("quadratic").starts_with(m) || m == "q")                   mode = keel::signal::ChirpMethod::kQuadratic;
    else if (keel::StringRef("logarithmic").starts_with(m) || m == "lo")                mode = keel::signal::ChirpMethod::kLogarithmic;
    else
        throw Error("chirp: method must be 'linear', 'quadratic', or 'logarithmic'",
                     0, 0, "chirp", "", "numkit:chirp:badMethod");

    if (mode == keel::signal::ChirpMethod::kLogarithmic) {
        if (f0 <= 0 || f1 <= 0)
            throw Error("chirp: 'logarithmic' requires f0 > 0 and f1 > 0",
                         0, 0, "chirp", "", "numkit:chirp:badFreq");
        if (f0 == f1)
            throw Error("chirp: 'logarithmic' requires f0 != f1",
                         0, 0, "chirp", "", "numkit:chirp:badFreq");
    }

    keel::signal::ChirpQuadShape qshape = keel::signal::ChirpQuadShape::kDefault;
    if (mode == keel::signal::ChirpMethod::kQuadratic) {
        keel::String sh = keel::String::from(quad_shape);
        sh.to_ascii_lower_in_place();
        if (!sh.empty()) {
            if (keel::StringRef("convex").starts_with(sh))       qshape = keel::signal::ChirpQuadShape::kConvex;
            else if (keel::StringRef("concave").starts_with(sh)) qshape = keel::signal::ChirpQuadShape::kConcave;
            else
                throw Error("chirp: shape must be 'convex' or 'concave'",
                            0, 0, "chirp", "", "numkit:chirp:badShape");
        }
    }

    auto out = createLike(t, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto tv = readReal(t, scratch.allocator());
    (void)keel::signal::chirp(tv.as_ref(), f0, t1, f1, mode, phi_deg, qshape,
                              keel::MutArrayRef<keel::f64>(out.doubleDataMut(), tv.size()));
    return out;
}

Value demod(const Value &y, keel::f64 Fc, keel::f64 Fs,
            keel::StringRef method, const Value &opt, keel::Allocator *mr)
{
    const ValueType vt = y.type();
    if (vt != ValueType::DOUBLE && vt != ValueType::SINGLE && vt != ValueType::COMPLEX)
        detail::throwInputType(y, "demod", "y", 1,
                               "a double, single, or complex double array");
    const bool ySingle = (vt == ValueType::SINGLE);
    if (Fs <= 0.0)
        throw Error("demod: Fs must be positive",
                    0, 0, "demod", "", "numkit:demod:BadFs");
    if (y.numel() == 0)
        return Value::matrix(0, 0, ySingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    keel::String m = keel::String::from(method);
    m.to_ascii_lower_in_place();
    keel::signal::ModulationMethod km;
    if (m == "am" || m == "amdsb-sc")      km = keel::signal::ModulationMethod::kAm;
    else if (m == "amdsb-tc")              km = keel::signal::ModulationMethod::kAmdsbTc;
    else if (m == "fm")                    km = keel::signal::ModulationMethod::kFm;
    else if (m == "pm")                    km = keel::signal::ModulationMethod::kPm;
    else
        throw Error("demod: only am/amdsb-sc/amdsb-tc/fm/pm supported "
                    "(amssb/pwm/ptm/ppm/qam deferred)",
                    0, 0, "demod", "", "numkit:demod:UnsupportedMethod");

    const bool isRowVec = (y.dims().rows() == 1 && y.dims().cols() > 1);
    const keel::usize len = isRowVec ? y.dims().cols() : y.dims().rows();
    const keel::usize cols = isRowVec ? 1 : y.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto yv = readReal(y, scratch.allocator());
    Value out = isRowVec ? Value::matrix(1, len, ValueType::DOUBLE, mr)
                         : Value::matrix(len, cols, ValueType::DOUBLE, mr);
    const bool hasOpt = !opt.isEmpty();
    const keel::f64 optVal = hasOpt ? opt.toScalar() : 0.0;
    (void)keel::signal::demod(
        yv.as_ref(), len, cols, Fc, Fs, km, hasOpt, optVal,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), len * cols), scratch.allocator());
    return detail::narrowDoubleToSingleIf(keel::move(out), ySingle, mr);
}

Value modulate(const Value &x, keel::f64 Fc, keel::f64 Fs,
               keel::StringRef method, const Value &opt, keel::Allocator *mr)
{
    const ValueType vt = x.type();
    if (vt != ValueType::DOUBLE && vt != ValueType::SINGLE && vt != ValueType::COMPLEX)
        detail::throwInputType(x, "modulate", "x", 1,
                               "a double, single, or complex double array");
    const bool xSingle = (vt == ValueType::SINGLE);
    if (Fs <= 0.0)
        throw Error("modulate: Fs must be positive",
                    0, 0, "modulate", "", "numkit:modulate:BadFs");
    if (x.numel() == 0)
        return Value::matrix(0, 0, xSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    keel::String m = keel::String::from(method);
    m.to_ascii_lower_in_place();
    keel::signal::ModulationMethod km;
    if (m == "am" || m == "amdsb-sc")      km = keel::signal::ModulationMethod::kAm;
    else if (m == "amssb")                 km = keel::signal::ModulationMethod::kAmssb;
    else if (m == "amdsb-tc")              km = keel::signal::ModulationMethod::kAmdsbTc;
    else if (m == "fm")                    km = keel::signal::ModulationMethod::kFm;
    else if (m == "pm")                    km = keel::signal::ModulationMethod::kPm;
    else
        throw Error("modulate: method must be 'am'/'amdsb-sc'/'amdsb-tc'/'amssb'/'fm'/'pm'",
                    0, 0, "modulate", "", "numkit:modulate:UnsupportedMethod");

    const bool isRowVec = (x.dims().rows() == 1 && x.dims().cols() > 1);
    const keel::usize len = isRowVec ? x.dims().cols() : x.dims().rows();
    const keel::usize cols = isRowVec ? 1 : x.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readReal(x, scratch.allocator());
    Value out = isRowVec ? Value::matrix(1, len, ValueType::DOUBLE, mr)
                         : Value::matrix(len, cols, ValueType::DOUBLE, mr);
    const bool hasOpt = !opt.isEmpty();
    const keel::f64 optVal = hasOpt ? opt.toScalar() : 0.0;
    (void)keel::signal::modulate(
        xv.as_ref(), len, cols, Fc, Fs, km, hasOpt, optVal,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), len * cols), scratch.allocator());
    return detail::narrowDoubleToSingleIf(keel::move(out), xSingle, mr);
}

namespace {

Value vcoImpl(const Value &x, keel::f64 Fc, keel::f64 range1, keel::f64 fs, keel::Allocator *mr)
{
    if (fs <= 0.0)
        throw Error("vco: fs must be positive",
                    0, 0, "vco", "", "numkit:vco:BadFs");
    const keel::usize N = x.numel();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (v > 1.0 || v < -1.0)
            throw Error("vco: x values must be in [-1, 1]",
                        0, 0, "vco", "", "numkit:vco:InvalidRange");
    }

    Value out = x.dims().is3D()
                    ? Value::matrix3d(x.dims().rows(), x.dims().cols(), x.dims().pages(), ValueType::DOUBLE, mr)
                    : Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
    if (N == 0) return out;

    const bool rowVec = (x.dims().rows() == 1 && x.dims().cols() > 1);
    const keel::usize R = rowVec ? x.dims().cols() : x.dims().rows();
    const keel::usize C = rowVec ? 1 : x.dims().cols();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readReal(x, scratch.allocator());
    (void)keel::signal::vco(xv.as_ref(), R, C, Fc, range1, fs,
                            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), R * C));
    return out;
}

} // anon

Value vco(const Value &x, keel::f64 fc, keel::f64 fs, keel::Allocator *mr)
{
    const keel::f64 range1 = (fc / fs) * keel::kTau;
    return vcoImpl(x, fc, range1, fs, mr);
}

Value vco(const Value &x, keel::f64 fmin, keel::f64 fmax, keel::f64 fs, keel::Allocator *mr)
{
    const keel::f64 fc = 0.5 * (fmin + fmax);
    const keel::f64 range1 = (fmax - fc) / fs * keel::kTau;
    return vcoImpl(x, fc, range1, fs, mr);
}

} // namespace numkit::signal
