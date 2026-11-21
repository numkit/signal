// toolboxes/signal/src/filter_design/iir_designs.cpp
//
// Top-level cheby1 / cheby2 / besself: compose analog prototype +
// lp2X + zp2tf + bilinear.
//
// Wn convention (matches MATLAB):
//   • Digital (analog == false):  Wn ∈ (0, 1) normalised to Nyquist;
//     pre-warp via Ωₐ = 2·fs·tan(π·Wn / 2) with fs = 1.
//   • Analog  (analog == true ): Wn already in rad/s.
//
// For bandpass/bandstop, Wn is a 2-vector [Wlo, Whi]; the prototype is
// scaled to centre Ω₀ = √(Wlo·Whi), bandwidth Bw = Whi - Wlo.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/math.h>
#include <numkit/signal/filter_design/iir_designs.hpp>

#include <numkit/signal/filter_design/analog_filters.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/signal/internal/shims/specfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace {

// IIR-family compute-side order contract, first line of every design.
// The scalar/integer/complex halves are Value-boundary-only and live in
// iirOrder in the reg layer.
void iirOrder_argcheck(keel::i32 N, const char *fn)
{
    if (N <= 0)
        throw Error(keel::format("{}: filter order N must be a positive integer (got {}).", fn, N),
                    0, 0, fn, "",
                    keel::format("numkit:{}:orderNotPositive", fn));
}

// Convert Wn (Value) → 1 or 2 doubles.
keel::Array<keel::f64> readWn(const Value &Wn) {
    const keel::usize n = Wn.numel();
    keel::Array<keel::f64> out(n);
    for (keel::usize i = 0; i < n; ++i) out[i] = Wn.elemAsDouble(i);
    return out;
}

// Pre-warp digital Wn ∈ (0, 1) → analog Ω = 2·tan(π·Wn / 2). With fs = 1
// (we'll undo via bilinear with fs = 1), this is the standard prewarp.
inline keel::f64 prewarp(keel::f64 Wn_norm) {
    return 2.0 * std::tan(M_PI * Wn_norm / 2.0);
}

// Apply lp2X transform per ftype to (z, p, k).
std::tuple<Value, Value, Value>
applyLp2X(Value z, Value p, keel::f64 k, FilterType ftype, keel::ArrayRef<keel::f64> Wo_analog, keel::Allocator *mr)
{
    switch (ftype) {
        case FilterType::Lowpass:
            return lp2lp(z, p, k, Wo_analog[0], mr);
        case FilterType::Highpass:
            return lp2hp(z, p, k, Wo_analog[0], mr);
        case FilterType::Bandpass: {
            const keel::f64 Wo = std::sqrt(Wo_analog[0] * Wo_analog[1]);
            const keel::f64 Bw = Wo_analog[1] - Wo_analog[0];
            return lp2bp(z, p, k, Wo, Bw, mr);
        }
        case FilterType::Bandstop: {
            const keel::f64 Wo = std::sqrt(Wo_analog[0] * Wo_analog[1]);
            const keel::f64 Bw = Wo_analog[1] - Wo_analog[0];
            return lp2bs(z, p, k, Wo, Bw, mr);
        }
    }
    throw std::runtime_error("iir_designs: bad ftype");
}

// Validate Wn arity vs ftype.
void validateWn(FilterType ftype, keel::ArrayRef<keel::f64> Wn) {
    const bool isBand = (ftype == FilterType::Bandpass || ftype == FilterType::Bandstop);
    if (isBand && Wn.size() != 2)
        throw std::runtime_error("bandpass/bandstop requires 2-element Wn");
    if (!isBand && Wn.size() != 1)
        throw std::runtime_error("lowpass/highpass requires scalar Wn");
}

// Common end-of-pipe: zpk → tf → bilinear (or stay analog).
std::tuple<Value, Value>
finishDesign(Value z, Value p, Value k_v, bool analog, keel::Allocator *mr)
{
    const keel::f64 k = k_v.toScalar();
    auto [b, a] = ::numkit::builtin::zp2tf(z, p, k, mr);
    if (analog) return std::make_tuple(std::move(b), std::move(a));
    // bilinear with fs = 1 reverses the pre-warp we applied earlier
    // (Ω = 2·tan(π·Wn/2) with fs = 1).
    return bilinear(b, a, /*fs=*/1.0, /*fp=*/0.0, mr);
}

// Translate raw Wn (digital normalised or analog rad/s) into the
// analog-domain Wo's used by the lp2X step.
keel::Array<keel::f64> wnToAnalog(keel::ArrayRef<keel::f64> Wn, bool analog) {
    if (analog) return keel::Array<keel::f64>(Wn);
    keel::Array<keel::f64> out(Wn.size());
    for (keel::usize i = 0; i < Wn.size(); ++i) out[i] = prewarp(Wn[i]);
    return out;
}

} // anonymous

std::tuple<Value, Value>
cheby1(keel::i32 N, keel::f64 Rp, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "cheby1");
    auto wn = readWn(Wn);
    validateWn(ftype, wn);
    auto [z, p, k_v] = cheb1ap(N, Rp, mr);
    auto Wo = wnToAnalog(wn, analog);
    auto [z2, p2, k2] = applyLp2X(std::move(z), std::move(p), k_v.toScalar(), ftype, Wo, mr);
    return finishDesign(std::move(z2), std::move(p2), std::move(k2), analog, mr);
}

std::tuple<Value, Value>
cheby2(keel::i32 N, keel::f64 Rs, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "cheby2");
    auto wn = readWn(Wn);
    validateWn(ftype, wn);
    auto [z, p, k_v] = cheb2ap(N, Rs, mr);
    auto Wo = wnToAnalog(wn, analog);
    auto [z2, p2, k2] = applyLp2X(std::move(z), std::move(p), k_v.toScalar(), ftype, Wo, mr);
    return finishDesign(std::move(z2), std::move(p2), std::move(k2), analog, mr);
}

std::tuple<Value, Value>
besself(keel::i32 N, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "besself");
    auto wn = readWn(Wn);
    validateWn(ftype, wn);
    auto [z, p, k_v] = besselap(N, mr);
    auto Wo = wnToAnalog(wn, analog);
    auto [z2, p2, k2] = applyLp2X(std::move(z), std::move(p), k_v.toScalar(), ftype, Wo, mr);
    return finishDesign(std::move(z2), std::move(p2), std::move(k2), analog, mr);
}

std::tuple<Value, Value>
butter(keel::i32 N, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "butter");
    // Same pipeline as cheby1/besself: Butterworth prototype, LP→X
    // transform, bilinear only for digital. The analog flag keeps Wn in
    // rad/s (any positive value) — the classic `butter(N, Wn, 's')` form
    // (bugs/closed/signal/butter-analog-flag-wn-domain.md).
    auto wn = readWn(Wn);
    validateWn(ftype, wn);
    auto [z, p, k_v] = buttap(N, mr);
    auto Wo = wnToAnalog(wn, analog);
    auto [z2, p2, k2] = applyLp2X(std::move(z), std::move(p), k_v.toScalar(), ftype, Wo, mr);
    return finishDesign(std::move(z2), std::move(p2), std::move(k2), analog, mr);
}

std::tuple<Value, Value>
ellip(keel::i32 N, keel::f64 Rp, keel::f64 Rs, const Value &Wn, FilterType ftype, bool analog, keel::Allocator *mr)
{
    iirOrder_argcheck(N, "ellip");
    auto wn = readWn(Wn);
    validateWn(ftype, wn);
    auto [z, p, k_v] = ellipap(N, Rp, Rs, mr);
    auto Wo = wnToAnalog(wn, analog);
    auto [z2, p2, k2] = applyLp2X(std::move(z), std::move(p), k_v.toScalar(), ftype, Wo, mr);
    return finishDesign(std::move(z2), std::move(p2), std::move(k2), analog, mr);
}

// ════════════════════════════════════════════════════════════════════
// Order estimators
// ════════════════════════════════════════════════════════════════════
//
// All four take (Wp, Ws, Rp, Rs) and return the minimum order N plus a
// natural-frequency vector Wn for the chosen design. For digital
// inputs Wp / Ws are normalised to Nyquist (∈ (0, 1)); we pre-warp via
// Ω = tan(π·W/2) before working in the analog domain. The final Wn is
// post-warped back via W = (2/π)·atan(Ω). For analog (analog == true)
// the inputs are already in rad/s and no warping is done.

namespace {

inline keel::f64 prewarpWp(keel::f64 w, bool analog) {
    return analog ? w : std::tan(M_PI * w / 2.0);
}
inline keel::f64 dewarpWp(keel::f64 Omega, bool analog) {
    return analog ? Omega : (2.0 / M_PI) * std::atan(Omega);
}

// Normalised analog stopband / passband ratio for a generic spec.
// For lowpass : Ωs / Ωp.   For highpass : Ωp / Ωs.
// For bandpass / bandstop : choose the worst-case ratio over the two
// stopband edges given the geometric centre Ω0 = √(Ωp1·Ωp2). MATLAB
// uses the standard prewarp + frequency-transform identities.
struct OrdNormalised {
    keel::f64 ratio;       // Ωs_norm / Ωp_norm; always > 1 for valid specs
    keel::f64 Wo_passband; // analog-domain passband for the prototype
    keel::f64 Wo_stopband; // analog-domain stopband edge (for buttord cutoff)
    FilterType ftype;
    keel::f64 Wo_low{0.0}; // for band designs: lower & upper analog edges
    keel::f64 Wo_high{0.0};
    keel::f64 Bw{0.0};
    keel::f64 Wo_centre{0.0};
};

OrdNormalised normaliseOrd(keel::ArrayRef<keel::f64> Wp,
                           keel::ArrayRef<keel::f64> Ws,
                           bool analog)
{
    OrdNormalised n{};
    if (Wp.size() == 1 && Ws.size() == 1) {
        const keel::f64 Op = prewarpWp(Wp[0], analog);
        const keel::f64 Os = prewarpWp(Ws[0], analog);
        if (Op == Os) throw std::runtime_error("ord: Wp and Ws must differ");
        if (Op < Os) {
            n.ftype = FilterType::Lowpass;
            n.ratio = Os / Op;
            n.Wo_passband = Op;
            n.Wo_stopband = Os;
        } else {
            n.ftype = FilterType::Highpass;
            n.ratio = Op / Os;
            n.Wo_passband = Op;
            n.Wo_stopband = Os;
        }
    } else if (Wp.size() == 2 && Ws.size() == 2) {
        const keel::f64 Op1 = prewarpWp(Wp[0], analog), Op2 = prewarpWp(Wp[1], analog);
        const keel::f64 Os1 = prewarpWp(Ws[0], analog), Os2 = prewarpWp(Ws[1], analog);
        // bandpass:  Ws1 < Wp1 < Wp2 < Ws2
        // bandstop:  Wp1 < Ws1 < Ws2 < Wp2
        const bool bp = (Os1 < Op1 && Op2 < Os2);
        n.ftype = bp ? FilterType::Bandpass : FilterType::Bandstop;
        const keel::f64 Wo = std::sqrt(Op1 * Op2);
        const keel::f64 Bw = Op2 - Op1;
        n.Wo_centre = Wo;
        n.Bw = Bw;
        n.Wo_low = Op1; n.Wo_high = Op2;
        if (bp) {
            // Bandpass→lowpass map: Ω_LP = (Ω² - Ω0²) / (Bw·Ω).
            // Take absolute value; ratio = |Ω_LP_stopband| (≥ 1 for valid spec).
            auto mp = [&](keel::f64 O){ return keel::abs((O * O - Wo * Wo) / (Bw * O)); };
            n.ratio = std::min(mp(Os1), mp(Os2));
            n.Wo_passband = 1.0;  // unit prototype
        } else {
            // Bandstop→lowpass: Ω_LP = (Bw·Ω) / (Ω² - Ω0²)  (reciprocal form).
            auto mp = [&](keel::f64 O){ return keel::abs((Bw * O) / (O * O - Wo * Wo)); };
            n.ratio = std::min(mp(Os1), mp(Os2));
            n.Wo_passband = 1.0;
        }
    } else {
        throw std::runtime_error("ord: Wp and Ws must both be scalar or both 2-vectors");
    }
    return n;
}

// Convert the prototype Ω back into the requested Wn vector.
Value wnFromPrototype(const OrdNormalised &n, keel::f64 Wn_analog, bool analog, keel::Allocator *mr)
{
    switch (n.ftype) {
        case FilterType::Lowpass:
        case FilterType::Highpass: {
            const keel::f64 w = dewarpWp(Wn_analog, analog);
            return Value::scalar(w, mr);
        }
        case FilterType::Bandpass:
        case FilterType::Bandstop: {
            // For band designs, MATLAB's *ord typically returns the
            // original passband edges (no recomputation needed).
            auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
            keel::f64 *od = out.doubleDataMut();
            od[0] = dewarpWp(n.Wo_low,  analog);
            od[1] = dewarpWp(n.Wo_high, analog);
            return out;
        }
    }
    throw std::runtime_error("wnFromPrototype: bad ftype");
}

} // anonymous

std::tuple<keel::i32, Value>
buttord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    auto Wp = readWn(Wp_v);
    auto Ws = readWn(Ws_v);
    auto n = normaliseOrd(Wp, Ws, analog);
    const keel::f64 GsLin = std::pow(10.0, Rs / 10.0) - 1.0;
    const keel::f64 GpLin = std::pow(10.0, Rp / 10.0) - 1.0;
    if (GpLin <= 0.0 || GsLin <= 0.0)
        throw std::runtime_error("buttord: Rp, Rs must be positive");
    const keel::i32 N = keel::as<keel::i32>(std::ceil(
        0.5 * std::log10(GsLin / GpLin) / std::log10(n.ratio)));
    // MATLAB convention: Wn computed from STOPBAND spec, so the resulting
    // filter meets stopband attenuation exactly. For Butterworth:
    //   lowpass:  |H|² = 1/(1+(Ω/Ωc)^{2N})  ⇒ Ωc = Ωs / GsLin^{1/(2N)}
    //   highpass: |H|² = 1/(1+(Ωc/Ω)^{2N})  ⇒ Ωc = Ωs · GsLin^{1/(2N)}
    const keel::f64 r = std::pow(GsLin, 1.0 / (2.0 * N));
    keel::f64 Wn_analog = 0.0;
    switch (n.ftype) {
        case FilterType::Lowpass:  Wn_analog = n.Wo_stopband / r; break;
        case FilterType::Highpass: Wn_analog = n.Wo_stopband * r; break;
        case FilterType::Bandpass:
        case FilterType::Bandstop: Wn_analog = 1.0; break;
    }
    Value Wn = wnFromPrototype(n, Wn_analog, analog, mr);
    return std::make_tuple(N, std::move(Wn));
}

std::tuple<keel::i32, Value>
cheb1ord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    auto Wp = readWn(Wp_v);
    auto Ws = readWn(Ws_v);
    auto n = normaliseOrd(Wp, Ws, analog);
    const keel::f64 GsLin = std::pow(10.0, Rs / 10.0) - 1.0;
    const keel::f64 GpLin = std::pow(10.0, Rp / 10.0) - 1.0;
    if (GpLin <= 0.0 || GsLin <= 0.0)
        throw std::runtime_error("cheb1ord: Rp, Rs must be positive");
    const keel::i32 N = keel::as<keel::i32>(std::ceil(
        std::acosh(std::sqrt(GsLin / GpLin)) / std::acosh(n.ratio)));
    // Natural cutoff for cheby1 = passband edge — return Wp directly.
    if (n.ftype == FilterType::Lowpass || n.ftype == FilterType::Highpass)
        return std::make_tuple(N, Value::scalar(Wp[0], mr));
    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    od[0] = Wp[0]; od[1] = Wp[1];
    return std::make_tuple(N, std::move(out));
}

std::tuple<keel::i32, Value>
cheb2ord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    auto Wp = readWn(Wp_v);
    auto Ws = readWn(Ws_v);
    auto n = normaliseOrd(Wp, Ws, analog);
    const keel::f64 GsLin = std::pow(10.0, Rs / 10.0) - 1.0;
    const keel::f64 GpLin = std::pow(10.0, Rp / 10.0) - 1.0;
    if (GpLin <= 0.0 || GsLin <= 0.0)
        throw std::runtime_error("cheb2ord: Rp, Rs must be positive");
    const keel::i32 N = keel::as<keel::i32>(std::ceil(
        std::acosh(std::sqrt(GsLin / GpLin)) / std::acosh(n.ratio)));
    // Natural cutoff for cheby2 = stopband edge — return Ws directly.
    if (n.ftype == FilterType::Lowpass || n.ftype == FilterType::Highpass)
        return std::make_tuple(N, Value::scalar(Ws[0], mr));
    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    od[0] = Ws[0]; od[1] = Ws[1];
    return std::make_tuple(N, std::move(out));
}

// ════════════════════════════════════════════════════════════════════
// Engine adapters
// ════════════════════════════════════════════════════════════════════


// ── ellipord (Phase 4.6) ───────────────────────────────────────────────
//
// Matches MATLAB R2025b ellipord.m. Algorithm:
//   1. Determine ftype: 1=low, 2=high, 3=stop, 4=pass.
//   2. Prewarp digital → analog: WP=tan(π wp/2), WS=tan(π ws/2).
//   3. Per ftype compute analog passband-edge ratio WA, then
//      findelliporder(WA, Rp, Rs):
//        WA = min(|WA|);  ε = sqrt(10^(0.1·Rp)-1);  k1 = ε/sqrt(10^(0.1·Rs)-1)
//        k = 1/WA
//        capk  = ellipke([k², 1-k²])
//        capk1 = ellipke([k1², 1-k1²])
//        N = ceil(K·E1 / (E·K1))  where K, E from capk and K1, E1 from capk1
//   4. wn = wp (digital) or WP (analog).
//
// KNOWN GAP: bandstop (ftype=3) deferred — needs digital→analog branch
// with sin/cos centroid + recursive analog call.

namespace {

keel::f64 findElliporderImpl(keel::ArrayRef<keel::f64> WA, keel::f64 Rp, keel::f64 Rs, keel::Allocator *mr)
{
    keel::f64 WAmin = keel::abs(WA[0]);
    for (keel::usize i = 1; i < WA.size(); ++i)
        if (keel::abs(WA[i]) < WAmin) WAmin = keel::abs(WA[i]);
    const keel::f64 epsilon = std::sqrt(std::pow(10.0, 0.1 * Rp) - 1.0);
    const keel::f64 k1 = epsilon / std::sqrt(std::pow(10.0, 0.1 * Rs) - 1.0);
    const keel::f64 k  = 1.0 / WAmin;

    // capk = ellipke([k², 1-k²])
    Value mk = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    mk.doubleDataMut()[0] = k * k;
    mk.doubleDataMut()[1] = 1.0 - k * k;
    auto capk = numkit::builtin::ellipke(mk, mr);

    // capk1 = ellipke([k1², 1-k1²])
    Value mk1 = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    mk1.doubleDataMut()[0] = k1 * k1;
    mk1.doubleDataMut()[1] = 1.0 - k1 * k1;
    auto capk1 = numkit::builtin::ellipke(mk1, mr);

    // capk.K is 1×2: [K(k²), K(1-k²)]; same for capk.E. Need:
    //   N = ceil(K(k²) * E1(1-k1²) / (K(k²)... wait MATLAB uses capk(2) for
    //   complementary). The MATLAB call returns K (first output), so capk(1)
    //   = K(k²), capk(2) = K(1-k²). Same convention here.
    const keel::f64 K0  = capk.K.elemAsDouble(0);
    const keel::f64 K1c = capk.K.elemAsDouble(1);
    const keel::f64 K1k1  = capk1.K.elemAsDouble(0);
    const keel::f64 K1k1c = capk1.K.elemAsDouble(1);
    return std::ceil(K0 * K1k1c / (K1c * K1k1));
}

} // anon

std::tuple<keel::i32, Value>
ellipord(const Value &Wp_v, const Value &Ws_v, keel::f64 Rp, keel::f64 Rs, bool analog, keel::Allocator *mr)
{
    if (Rp <= 0.0 || Rs <= 0.0)
        throw Error("ellipord: Rp, Rs must be positive",
                    0, 0, "ellipord", "", "numkit:ellipord:BadRpRs");
    if (Wp_v.numel() != Ws_v.numel())
        throw Error("ellipord: Wp and Ws must have same length",
                    0, 0, "ellipord", "", "numkit:ellipord:DimMismatch");

    const keel::usize numW = Wp_v.numel();
    if (numW != 1 && numW != 2)
        throw Error("ellipord: Wp must be scalar or 2-vector",
                    0, 0, "ellipord", "", "numkit:ellipord:BadWp");

    keel::Array<keel::f64> wp(numW, mr), ws(numW, mr);
    for (keel::usize i = 0; i < numW; ++i) {
        wp[i] = Wp_v.elemAsDouble(i);
        ws[i] = Ws_v.elemAsDouble(i);
    }

    // ftype = 2*(numW-1) + (1 if wp[0] < ws[0] else 2)
    keel::i32 ftype = 2 * (keel::as<keel::i32>(numW) - 1);
    ftype += (wp[0] < ws[0]) ? 1 : 2;

    // Prewarp digital → analog if needed.
    keel::Array<keel::f64> WP(numW, mr), WS(numW, mr);
    if (!analog) {
        for (keel::usize i = 0; i < numW; ++i) {
            WP[i] = std::tan(M_PI * wp[i] / 2.0);
            WS[i] = std::tan(M_PI * ws[i] / 2.0);
        }
    } else {
        WP = wp.clone(); WS = ws.clone();
    }

    keel::Array<keel::f64> WA(mr);
    keel::i32 N = 0;
    switch (ftype) {
        case 1: { // lowpass: WA = WS / WP
            WA.push_back(WS[0] / WP[0]);
            N = keel::as<keel::i32>(findElliporderImpl(WA, Rp, Rs, mr));
            break;
        }
        case 2: { // highpass: WA = WP / WS
            WA.push_back(WP[0] / WS[0]);
            N = keel::as<keel::i32>(findElliporderImpl(WA, Rp, Rs, mr));
            break;
        }
        case 3: { // bandstop: reciprocal of the bandpass→lowpass map.
            // Ω_LP = (Bw·Ω) / (Ω² - Ω0²), Bw = WP2-WP1, Ω0² = WP1·WP2; written
            // here as the reciprocal of the case-4 expression so the shared
            // sign/abs convention in findElliporderImpl (WAmin = min|WA|) picks
            // the worst-case stopband edge. (Mirrors nk buttord/cheb*ord.)
            for (keel::usize i = 0; i < numW; ++i) {
                const keel::f64 w = WS[i];
                WA.push_back((w * (WP[0] - WP[1])) / (w * w - WP[0] * WP[1]));
            }
            N = keel::as<keel::i32>(findElliporderImpl(WA, Rp, Rs, mr));
            break;
        }
        case 4: { // bandpass: WA = (WS² - WP1·WP2) / (WS·(WP1-WP2))
            for (keel::usize i = 0; i < numW; ++i) {
                const keel::f64 w = WS[i];
                WA.push_back((w * w - WP[0] * WP[1]) / (w * (WP[0] - WP[1])));
            }
            N = keel::as<keel::i32>(findElliporderImpl(WA, Rp, Rs, mr));
            break;
        }
        default:
            throw Error("ellipord: invalid filter spec",
                        0, 0, "ellipord", "", "numkit:ellipord:BadSpec");
    }

    // wn = wp (digital) or WP (analog).
    Value Wn = Value::matrix(1, numW, ValueType::DOUBLE, mr);
    keel::f64 *wd = Wn.doubleDataMut();
    if (!analog) {
        for (keel::usize i = 0; i < numW; ++i) wd[i] = wp[i];
    } else {
        for (keel::usize i = 0; i < numW; ++i) wd[i] = WP[i];
    }

    return {N, Wn};
}


// ── Parks-McClellan FIR order estimator (Phase 4.7) ───────────────────
//
// Matches MATLAB R2025b firpmord.m + remlpord. Coefficient matrix is
// from Rabiner & Gold "Theory and Applications of DSP" pp. 156-7.

namespace {

keel::f64 remlpord(keel::f64 f1, keel::f64 f2, keel::f64 d1, keel::f64 d2)
{
    static constexpr keel::f64 AA[3][3] = {
        {-4.278e-01, -4.761e-01, 0.0},
        {-5.941e-01,  7.114e-02, 0.0},
        {-2.660e-03,  5.309e-03, 0.0}
    };
    const keel::f64 ld1 = std::log10(d1);
    const keel::f64 ld2 = std::log10(d2);
    const keel::f64 v[3] = {1.0, ld1, ld1 * ld1};
    const keel::f64 w[3] = {1.0, ld2, ld2 * ld2};
    keel::f64 D = 0.0;
    for (keel::i32 i = 0; i < 3; ++i)
        for (keel::i32 j = 0; j < 3; ++j)
            D += v[i] * AA[i][j] * w[j];
    const keel::f64 fK = 11.01217 + 0.51244 * (ld1 - ld2);
    const keel::f64 df = keel::abs(f2 - f1);
    return D / df - fK * df + 1.0;
}

} // anon

std::tuple<keel::i32, Value, Value, Value>
firpmord(const Value &F, const Value &A, const Value &dev, keel::f64 fs, keel::Allocator *mr)
{
    const keel::usize mf     = F.numel();
    const keel::usize nbands = A.numel();
    const keel::usize ndevs  = dev.numel();

    if (nbands != ndevs)
        throw Error("firpmord: A and DEV must have same length",
                    0, 0, "firpmord", "", "numkit:firpmord:MismatchedVectorLength");
    if (mf != 2 * (nbands - 1))
        throw Error("firpmord: numel(F) must equal 2*(numel(A)-1)",
                    0, 0, "firpmord", "", "numkit:firpmord:InvalidLength");
    if (fs <= 0.0)
        throw Error("firpmord: Fs must be positive",
                    0, 0, "firpmord", "", "numkit:firpmord:BadFs");

    keel::Array<keel::f64> fcuts(mf, mr), mags(nbands, mr), devs(nbands, mr);
    for (keel::usize i = 0; i < mf; ++i)     fcuts[i] = F.elemAsDouble(i) / fs;
    for (keel::usize i = 0; i < nbands; ++i) {
        mags[i] = A.elemAsDouble(i);
        devs[i] = dev.elemAsDouble(i);
    }
    {
        keel::f64 mx = fcuts[0];
        for (keel::usize i = 1; i < mf; ++i) if (fcuts[i] > mx) mx = fcuts[i];
        if (mx > 0.5)
            throw Error("firpmord: F edges must be <= Fs/2",
                        0, 0, "firpmord", "", "numkit:firpmord:InvalidRange");
    }

    // Relative deviation: devs[i] /= (stop + mag) (== 1 either way).
    for (keel::usize i = 0; i < nbands; ++i) {
        const keel::f64 zz = (mags[i] == 0.0) ? 1.0 : 0.0;
        const keel::f64 base = zz + mags[i];
        if (base != 0.0) devs[i] = devs[i] / base;
    }

    // Separate transition edges into f1, f2 pairs.
    keel::Array<keel::f64> f1v(mr), f2v(mr);
    for (keel::usize i = 0; i + 1 < mf; i += 2) {
        f1v.push_back(fcuts[i]);
        f2v.push_back(fcuts[i + 1]);
    }
    // Find narrowest transition.
    keel::usize nMin = 0;
    {
        keel::f64 minWidth = keel::abs(f2v[0] - f1v[0]);
        for (keel::usize i = 1; i < f1v.size(); ++i) {
            const keel::f64 w = keel::abs(f2v[i] - f1v[i]);
            if (w < minWidth) { minWidth = w; nMin = i; }
        }
    }

    keel::f64 L = 0.0;
    if (nbands == 2) {
        L = remlpord(f1v[nMin], f2v[nMin], devs[0], devs[1]);
    } else {
        for (keel::usize i = 1; i + 1 < nbands; ++i) {
            const keel::f64 L1 = remlpord(f1v[i - 1], f2v[i - 1], devs[i], devs[i - 1]);
            const keel::f64 L2 = remlpord(f1v[i],     f2v[i],     devs[i], devs[i + 1]);
            if (L1 > L) L = L1;
            if (L2 > L) L = L2;
        }
    }

    keel::i32 N = keel::as<keel::i32>(std::ceil(L)) - 1;

    // Build firpm-compatible spec vectors.
    // ff = [0; 2*fcuts; 1]  → length mf + 2
    Value ff = Value::matrix(1, mf + 2, ValueType::DOUBLE, mr);
    {
        keel::f64 *fd = ff.doubleDataMut();
        fd[0] = 0.0;
        for (keel::usize i = 0; i < mf; ++i) fd[i + 1] = 2.0 * fcuts[i];
        fd[mf + 1] = 1.0;
    }
    // am(1:2:2*nbands-1) = mags  → length 2*nbands-1, others 0
    // aa = [am'; 0] + [0; am']  (i.e., interleave)
    // Effectively aa is length 2*nbands, each band's amplitude appears
    // in both endpoints.
    const keel::usize aLen = 2 * nbands;
    Value aa = Value::matrix(1, aLen, ValueType::DOUBLE, mr);
    {
        keel::f64 *ad = aa.doubleDataMut();
        std::fill(ad, ad + aLen, 0.0);
        // am at indices 0, 2, 4, ..., 2*nbands-2 (in 0-based).
        // After the [am'; 0] + [0; am'] trick: position 2k and 2k+1 both = mags[k].
        for (keel::usize k = 0; k < nbands; ++k) {
            ad[2 * k]     = mags[k];
            ad[2 * k + 1] = mags[k];
        }
    }
    // wts = max(devs) ./ devs
    Value wts = Value::matrix(1, nbands, ValueType::DOUBLE, mr);
    {
        keel::f64 *wd = wts.doubleDataMut();
        keel::f64 maxDev = devs[0];
        for (keel::usize i = 1; i < nbands; ++i) if (devs[i] > maxDev) maxDev = devs[i];
        for (keel::usize i = 0; i < nbands; ++i)
            wd[i] = (devs[i] != 0.0) ? maxDev / devs[i] : 0.0;
    }

    // If gain at Nyquist != 0 and N odd, bump up.
    if (mags[nbands - 1] != 0.0 && (N % 2) != 0) ++N;

    return {N, ff, aa, wts};
}


// ── Kaiser-window FIR order estimator (Phase 4.5) ─────────────────────
//
// Matches MATLAB R2025b kaiserord.m exactly. References:
//   Kaiser, "Nonrecursive Digital Filter Design Using the I_o-sinh
//     Window Function", Proc. 1974 IEEE Symp. Circuits & Syst.
//   Rabiner & Gold, Theory and Applications of DSP, pp. 156-7.

namespace {

keel::f64 kaiserBetaFn(keel::f64 atten)
{
    if (atten > 50.0) return 0.1102 * (atten - 8.7);
    if (atten >= 21.0)
        return 0.5842 * std::pow(atten - 21.0, 0.4) + 0.07886 * (atten - 21.0);
    return 0.0;
}

struct KaislpResult { keel::f64 L; keel::f64 beta; };
KaislpResult kaislpord(keel::f64 f1, keel::f64 f2, keel::f64 d1, keel::f64 d2)
{
    const keel::f64 delta = std::min(d1, d2);
    const keel::f64 atten = -20.0 * std::log10(delta);
    const keel::f64 D = (atten - 7.95) / (2.0 * M_PI * 2.285);
    const keel::f64 L = D / keel::abs(f2 - f1) + 1.0;
    return {L, kaiserBetaFn(atten)};
}

} // anon

std::tuple<keel::i32, Value, keel::f64, keel::String>
kaiserord(const Value &F, const Value &A, const Value &dev, keel::f64 fs, keel::Allocator *mr)
{
    const keel::usize mf     = F.numel();
    const keel::usize nbands = A.numel();
    const keel::usize ndevs  = dev.numel();

    if (nbands != ndevs)
        throw Error("kaiserord: A and DEV must have same length",
                    0, 0, "kaiserord", "", "numkit:kaiserord:InvalidDimensionsADEV");
    if (mf != 2 * (nbands - 1))
        throw Error("kaiserord: numel(F) must equal 2*(numel(A)-1)",
                    0, 0, "kaiserord", "", "numkit:kaiserord:InvalidDimensionsLengthF");
    if (fs <= 0.0)
        throw Error("kaiserord: Fs must be positive",
                    0, 0, "kaiserord", "", "numkit:kaiserord:BadFs");

    keel::Array<keel::f64> fcuts(mf, mr), mags(nbands, mr), devs(nbands, mr);
    for (keel::usize i = 0; i < mf; ++i)     fcuts[i] = F.elemAsDouble(i) / fs;
    for (keel::usize i = 0; i < nbands; ++i) {
        mags[i] = A.elemAsDouble(i);
        devs[i] = dev.elemAsDouble(i);
    }
    {
        keel::f64 mx = fcuts[0];
        for (keel::usize i = 1; i < mf; ++i) if (fcuts[i] > mx) mx = fcuts[i];
        if (mx >= 0.5)
            throw Error("kaiserord: F edges must be < Fs/2",
                        0, 0, "kaiserord", "", "numkit:kaiserord:InvalidRange");
    }

    // Convert dev → relative deviation: dev /= (stop + mag)  (== 1 either way)
    for (keel::usize i = 0; i < nbands; ++i) {
        const keel::f64 stop = (mags[i] == 0.0) ? 1.0 : 0.0;
        const keel::f64 base = stop + mags[i];
        if (base != 0.0) devs[i] = devs[i] / base;
    }

    // Separate transition edges into f1 / f2 pairs.
    keel::Array<keel::f64> f1v(mr), f2v(mr);
    for (keel::usize i = 0; i + 1 < mf; i += 2) {
        f1v.push_back(fcuts[i]);
        f2v.push_back(fcuts[i + 1]);
    }

    // Find narrowest transition zone.
    keel::usize nMin = 0;
    {
        keel::f64 minWidth = keel::abs(f2v[0] - f1v[0]);
        for (keel::usize i = 1; i < f1v.size(); ++i) {
            const keel::f64 w = keel::abs(f2v[i] - f1v[i]);
            if (w < minWidth) { minWidth = w; nMin = i; }
        }
    }

    keel::f64 L = 0.0, bta = 0.0;
    if (nbands == 2) {
        auto r = kaislpord(f1v[nMin], f2v[nMin], devs[0], devs[1]);
        L = r.L;
        bta = r.beta;
    } else {
        for (keel::usize i = 1; i + 1 < nbands; ++i) {
            auto r1 = kaislpord(f1v[i - 1], f2v[i - 1], devs[i], devs[i - 1]);
            auto r2 = kaislpord(f1v[i],     f2v[i],     devs[i], devs[i + 1]);
            if (r1.L > L) { L = r1.L; bta = r1.beta; }
            if (r2.L > L) { L = r2.L; bta = r2.beta; }
        }
    }

    keel::i32 N = keel::as<keel::i32>(std::ceil(L)) - 1;

    // Wn = (f1 + f2) per pair (already factor-of-2 normalized to Nyquist).
    Value Wn = Value::matrix(1, f1v.size(), ValueType::DOUBLE, mr);
    {
        keel::f64 *wd = Wn.doubleDataMut();
        for (keel::usize i = 0; i < f1v.size(); ++i) wd[i] = f1v[i] + f2v[i];
    }

    keel::String ftype = keel::String::from("low", mr);
    if (nbands == 2 && mags[0] == 0.0)              ftype.assign("high");
    else if (nbands == 3 && mags[1] == 0.0)         ftype.assign("stop");
    else if (nbands >= 3 && mags[0] == 0.0)         ftype.assign("DC-0");
    else if (nbands >= 3 && mags[0] == 1.0)         ftype.assign("DC-1");

    if ((N % 2) != 0 && mags[nbands - 1] != 0.0) ++N;

    return {N, std::move(Wn), bta, keel::move(ftype)};
}

} // namespace numkit::signal
