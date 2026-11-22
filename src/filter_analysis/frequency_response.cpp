// toolboxes/signal/src/filter_analysis/frequency_response.cpp
//
// Frequency-domain analysis of an existing filter — freqz / phasez /
// grpdelay. Filter design (butter / fir1) lives in
// filter_design/filter_design.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/filter_analysis/frequency_response.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "../dsp_helpers.hpp"   // Complex typedef

#include <cmath>
#include <limits>
#include <tuple>


namespace numkit::signal {

namespace {

void freqz_argcheck(const Value &b, const Value &a)
{
    (void)b;
    (void)a;
}

} // namespace

std::tuple<Value, Value>
freqz(const Value &b, const Value &a, keel::usize npts, bool whole, keel::f64 fs, keel::Allocator *mr)
{
    freqz_argcheck(b, a);
    const bool bCplx = b.isComplex();
    const bool aCplx = a.isComplex();
    const keel::c64 *bc = bCplx ? b.complexData() : nullptr;
    const keel::c64 *ac = aCplx ? a.complexData() : nullptr;
    const keel::f64 *bd = bCplx ? nullptr : b.doubleData();
    const keel::f64 *ad = aCplx ? nullptr : a.doubleData();
    const keel::usize nb = b.numel(), na = a.numel();

    auto W = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    auto H = Value::complexMatrix(npts, 1, mr);

    // MATLAB freqz(b, a, n): n equispaced frequencies on [0, π) — the
    // upper endpoint π is excluded. Grid is w = (0:n-1) * π / n. With
    // 'whole', the grid spans the full unit circle [0, 2π):
    // w = (0:n-1) * 2π / n. The response H is always evaluated at these
    // normalised radian frequencies; when fs > 0 (the freqz(b,a,n,fs)
    // form) the RETURNED frequency vector is rescaled to Hz over [0, fs/2)
    // (or [0, fs) with 'whole') — i.e. f = w * fs / (2π).
    const keel::f64 span   = whole ? (2.0 * keel::kPi) : keel::kPi;
    const keel::f64 hzSpan = whole ? fs : (0.5 * fs);
    for (keel::usize k = 0; k < npts; ++k) {
        const keel::f64 w = span * k / npts;
        W.doubleDataMut()[k] = (fs > 0.0) ? (hzSpan * k / npts) : w;

        const keel::c64 ejw(std::cos(w), -std::sin(w));
        keel::c64 num(0, 0), den(0, 0);
        keel::c64 ejwk(1, 0);
        for (keel::usize i = 0; i < nb; ++i) {
            num += (bCplx ? bc[i] : keel::c64(bd[i], 0.0)) * ejwk;
            ejwk *= ejw;
        }
        ejwk = keel::c64(1, 0);
        for (keel::usize i = 0; i < na; ++i) {
            den += (aCplx ? ac[i] : keel::c64(ad[i], 0.0)) * ejwk;
            ejwk *= ejw;
        }
        H.complexDataMut()[k] = num / den;
    }

    return std::make_tuple(std::move(H), std::move(W));
}

std::tuple<Value, Value>
freqz(const Value &b, const Value &a, const Value &wvec, keel::Allocator *mr)
{
    // freqz(b, a, w): response at the GIVEN normalised radian frequencies
    // (the vector-w form; MATLAB evaluates exactly these points).
    const bool bCplx = b.isComplex();
    const bool aCplx = a.isComplex();
    const keel::c64 *bc = bCplx ? b.complexData() : nullptr;
    const keel::c64 *ac = aCplx ? a.complexData() : nullptr;
    const keel::f64 *bd = bCplx ? nullptr : b.doubleData();
    const keel::f64 *ad = aCplx ? nullptr : a.doubleData();
    const keel::usize nb = b.numel(), na = a.numel();
    const keel::usize n = wvec.numel();
    const keel::f64 *wd = wvec.doubleData();

    // MATLAB R2025b (probed): H and W preserve w's orientation — a row w
    // yields a row H, a column w a column H.
    const keel::usize rows = wvec.dims().rows(), cols = wvec.dims().cols();
    auto W = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    auto H = Value::complexMatrix(rows, cols, mr);
    for (keel::usize k = 0; k < n; ++k) {
        const keel::f64 w = wd[k];
        W.doubleDataMut()[k] = w;
        const keel::c64 ejw(std::cos(w), -std::sin(w));
        keel::c64 num(0, 0), den(0, 0);
        keel::c64 ejwk(1, 0);
        for (keel::usize i = 0; i < nb; ++i) {
            num += (bCplx ? bc[i] : keel::c64(bd[i], 0.0)) * ejwk;
            ejwk *= ejw;
        }
        ejwk = keel::c64(1, 0);
        for (keel::usize i = 0; i < na; ++i) {
            den += (aCplx ? ac[i] : keel::c64(ad[i], 0.0)) * ejwk;
            ejwk *= ejw;
        }
        H.complexDataMut()[k] = num / den;
    }
    return std::make_tuple(std::move(H), std::move(W));
}

namespace {

// Local unwrap (default tolerance π) — keeps phasez free of inter-file
// dependencies on filter_analysis/unwrap.cpp's Value-typed unwrap().
void unwrapInPlace(keel::f64 *p, keel::usize n)
{
    if (n < 2) return;
    constexpr keel::f64 kPi  = keel::kPi;
    constexpr keel::f64 k2Pi = 2.0 * keel::kPi;
    keel::f64 offset = 0.0;
    for (keel::usize i = 1; i < n; ++i) {
        const keel::f64 diff = p[i] + offset - p[i - 1];
        if (diff >  kPi) offset -= k2Pi;
        else if (diff < -kPi) offset += k2Pi;
        p[i] += offset;
    }
}

} // namespace

std::tuple<Value, Value>
phasez(const Value &b, const Value &a, keel::usize npts, keel::f64 fs, keel::Allocator *mr)
{
    auto [H, W] = freqz(b, a, npts, /*whole=*/false, fs, mr);
    auto phi = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    const keel::c64 *hd = H.complexData();
    keel::f64 *pd = phi.doubleDataMut();
    // MATLAB convention: when |H(w)| is exactly zero, phase is
    // undefined and reported as NaN. Use a tiny epsilon to catch the
    // numerical zeros that come out of the freqz polynomial sum.
    constexpr keel::f64 kZeroEps = 1e-300;
    for (keel::usize k = 0; k < npts; ++k) {
        const keel::f64 mag2 = hd[k].real() * hd[k].real()
                          + hd[k].imag() * hd[k].imag();
        if (mag2 < kZeroEps)
            pd[k] = std::numeric_limits<keel::f64>::quiet_NaN();
        else
            pd[k] = std::atan2(hd[k].imag(), hd[k].real());
    }
    unwrapInPlace(pd, npts);
    return std::make_tuple(std::move(phi), std::move(W));
}

std::tuple<Value, Value>
grpdelay(const Value &b, const Value &a, keel::usize npts, keel::f64 fs, keel::Allocator *mr)
{
    // EXACT group delay (MATLAB's method), NOT a finite-difference of the
    // phase. For H(z) = B(z)/A(z), form the combined polynomial
    //   c = conv(b, reverse(a)),
    // whose argument is arg(B) - w*(na-1) - arg(A). Hence
    //   gd(w) = -d/dw arg(H) = Re{ CR(e^jw) / C(e^jw) } - (na - 1),
    // where CR has coefficients n*c[n]. This is exact at every frequency
    // (the old phase finite-difference was wildly off at small npts).
    const keel::f64 *bd = b.doubleData();
    const keel::f64 *ad = a.doubleData();
    const keel::usize nb = b.numel(), na = a.numel();

    auto W  = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    auto gd = Value::matrix(npts, 1, ValueType::DOUBLE, mr);
    if (npts == 0 || nb == 0 || na == 0)
        return std::make_tuple(std::move(gd), std::move(W));

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize nc = nb + na - 1;
    auto c = keel::Array<keel::f64>(nc, scratch.allocator());
    for (keel::usize i = 0; i < nc; ++i) c[i] = 0.0;
    for (keel::usize i = 0; i < nb; ++i)
        for (keel::usize j = 0; j < na; ++j)
            c[i + j] += bd[i] * ad[na - 1 - j];   // conv(b, reverse(a))

    keel::f64 *wp = W.doubleDataMut();
    keel::f64 *g  = gd.doubleDataMut();
    const keel::f64 offset = keel::as<keel::f64>(na) - 1.0;
    // fs > 0 (grpdelay(b,a,n,fs)): return the frequency vector in Hz over
    // [0, fs/2). The group delay (in samples) is evaluated from the same
    // normalised w and is unchanged.
    const keel::f64 hzSpan = 0.5 * fs;
    for (keel::usize k = 0; k < npts; ++k) {
        const keel::f64 w = keel::kPi * keel::as<keel::f64>(k) / keel::as<keel::f64>(npts);
        wp[k] = (fs > 0.0) ? (hzSpan * keel::as<keel::f64>(k) / keel::as<keel::f64>(npts)) : w;
        const keel::c64 ejw(std::cos(w), -std::sin(w));
        keel::c64 C(0, 0), CR(0, 0), ejwk(1, 0);
        for (keel::usize n = 0; n < nc; ++n) {
            C  += c[n] * ejwk;
            CR += (keel::as<keel::f64>(n) * c[n]) * ejwk;
            ejwk *= ejw;
        }
        const keel::f64 cmag2 = C.real() * C.real() + C.imag() * C.imag();
        g[k] = (cmag2 < 1e-300) ? 0.0 : ((CR / C).real() - offset);
    }
    return std::make_tuple(std::move(gd), std::move(W));
}

} // namespace numkit::signal
