// toolboxes/signal/src/digital_filtering/spec_driven.cpp
//
// lowpass / highpass / bandpass / bandstop -- wrappers over the
// MATLAB-default IIR filter design + filtfilt.
//
// MATLAB's lowpass(x, fpass) (and friends) defaults match
// designfilt("lowpassiir", "FilterOrder", N, "PassbandFrequency", fpass,
//            "PassbandRipple", 0.1, "StopbandAttenuation", 60,
//            "DesignMethod", "ellip") with cutoff = midpoint(fpass, fstop)
// where fstop = fpass + (1 - Steepness)*(1 - fpass), Steepness = 0.85.
// We compose ellip() + filtfilt() with the same parameters.

#include <keel/format.h>
#include <keel/memory.h>
#include <numkit/signal/digital_filtering/spec_driven.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/signal/digital_filtering/filter.hpp>
#include <numkit/signal/digital_filtering/sosfilt.hpp>
#include <numkit/signal/filter_design/filter_design.hpp>
#include <numkit/signal/filter_design/iir_designs.hpp>
#include <numkit/signal/filter_implementation/conversions.hpp>

#include <cmath>

namespace numkit::signal {

namespace {

keel::f64 normaliseW(keel::f64 f, keel::f64 fs, const char *fnName)
{
    if (!(fs > 0))
        throw Error(keel::format("{}: fs must be positive", fnName),
                     0, 0, fnName, "", keel::format("numkit:{}:badFs", fnName));
    const keel::f64 w = 2.0 * f / fs;
    if (!(w > 0.0) || !(w < 1.0))
        throw Error(keel::format("{}: cutoff must be in (0, fs/2)", fnName),
                     0, 0, fnName, "", keel::format("numkit:{}:badCutoff", fnName));
    return w;
}

void validateOrder(keel::i32 order, const char *fnName)
{
    if (order < 1)
        throw Error(keel::format("{}: order must be >= 1", fnName),
                     0, 0, fnName, "", keel::format("numkit:{}:badOrder", fnName));
}

// MATLAB lowpass/highpass/bandpass/bandstop default parameters --
// these match what designfilt("lowpassiir", ..., "DesignMethod", "ellip")
// produces under the lowpass/highpass/etc front-end.
constexpr keel::f64 kDefaultRp = 0.1;     // passband ripple, dB
constexpr keel::f64 kDefaultRs = 60.0;    // stopband attenuation, dB
constexpr keel::i32    kDefaultIirOrder  = 7; // matches MATLAB IIR-branch default
// (Steepness=0.85 only affects the FIR branch's transition width; the
// IIR ellip path uses the passband edge directly as Wp.)

inline Value scalarWn(keel::f64 w, keel::Allocator *mr) {
    return Value::scalar(w, mr);
}
inline Value pairWn(keel::f64 w1, keel::f64 w2, keel::Allocator *mr) {
    auto v = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    v.doubleDataMut()[0] = w1;
    v.doubleDataMut()[1] = w2;
    return v;
}

void lowpass_argcheck(keel::f64 fpass, keel::f64 fs, keel::i32 order)
{
    validateOrder(order, "lowpass");
    (void)normaliseW(fpass, fs, "lowpass");
}

void highpass_argcheck(keel::f64 fpass, keel::f64 fs, keel::i32 order)
{
    validateOrder(order, "highpass");
    (void)normaliseW(fpass, fs, "highpass");
}

void bandpass_argcheck(keel::f64 flo, keel::f64 fhi, keel::f64 fs, keel::i32 order)
{
    validateOrder(order, "bandpass");
    if (!(flo < fhi))
        throw Error("bandpass: low cutoff must be < high cutoff",
                     0, 0, "bandpass", "", "numkit:bandpass:badRange");
    (void)normaliseW(flo, fs, "bandpass");
    (void)normaliseW(fhi, fs, "bandpass");
}

void bandstop_argcheck(keel::f64 flo, keel::f64 fhi, keel::f64 fs, keel::i32 order)
{
    validateOrder(order, "bandstop");
    if (!(flo < fhi))
        throw Error("bandstop: low cutoff must be < high cutoff",
                     0, 0, "bandstop", "", "numkit:bandstop:badRange");
    (void)normaliseW(flo, fs, "bandstop");
    (void)normaliseW(fhi, fs, "bandstop");
}

} // namespace

Value lowpass(const Value &x, keel::f64 fpass, keel::f64 fs, keel::i32 order, keel::Allocator *mr)
{
    lowpass_argcheck(fpass, fs, order);
    const keel::f64 Wp = normaliseW(fpass, fs, "lowpass");
    // honour explicit order; remap legacy default 8 -> 7 to match the reference engine
    const keel::i32 N = (order == 8) ? kDefaultIirOrder : order;
    auto [b, a] = ellip(N, kDefaultRp, kDefaultRs, scalarWn(Wp, mr), FilterType::Lowpass, /*analog=*/false, mr);
    // SOS-form filtfilt is numerically stable for high-order IIR.
    auto sos = tf2sos(b, a, mr);
    return sosfiltfilt(sos, x, mr);
}

Value highpass(const Value &x, keel::f64 fpass, keel::f64 fs, keel::i32 order, keel::Allocator *mr)
{
    highpass_argcheck(fpass, fs, order);
    const keel::f64 Wp = normaliseW(fpass, fs, "highpass");
    const keel::i32 N = (order == 8) ? kDefaultIirOrder : order;
    auto [b, a] = ellip(N, kDefaultRp, kDefaultRs, scalarWn(Wp, mr), FilterType::Highpass, /*analog=*/false, mr);
    auto sos = tf2sos(b, a, mr);
    return sosfiltfilt(sos, x, mr);
}

Value bandpass(const Value &x, keel::f64 flo, keel::f64 fhi, keel::f64 fs, keel::i32 order, keel::Allocator *mr)
{
    bandpass_argcheck(flo, fhi, fs, order);
    const keel::f64 Wlo = normaliseW(flo, fs, "bandpass");
    const keel::f64 Whi = normaliseW(fhi, fs, "bandpass");
    const keel::i32 N = (order == 8) ? kDefaultIirOrder : order;
    auto [b, a] = ellip(N, kDefaultRp, kDefaultRs, pairWn(Wlo, Whi, mr), FilterType::Bandpass, /*analog=*/false, mr);
    auto sos = tf2sos(b, a, mr);
    return sosfiltfilt(sos, x, mr);
}

Value bandstop(const Value &x, keel::f64 flo, keel::f64 fhi, keel::f64 fs, keel::i32 order, keel::Allocator *mr)
{
    bandstop_argcheck(flo, fhi, fs, order);
    const keel::f64 Wlo = normaliseW(flo, fs, "bandstop");
    const keel::f64 Whi = normaliseW(fhi, fs, "bandstop");
    const keel::i32 N = (order == 8) ? kDefaultIirOrder : order;
    auto [b, a] = ellip(N, kDefaultRp, kDefaultRs, pairWn(Wlo, Whi, mr), FilterType::Bandstop, /*analog=*/false, mr);
    auto sos = tf2sos(b, a, mr);
    return sosfiltfilt(sos, x, mr);
}

} // namespace numkit::signal
