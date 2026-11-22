// toolboxes/signal/src/multirate/extras.cpp
//
// upfirdn / interp / intfilt / fftfilt.
// upfirdn and fftfilt delegate pure numerical kernels to keel::signal.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/multirate/extras.hpp>

#include <numkit/signal/filter_design/filter_design.hpp>   // firls
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <cstring>

namespace numkit::signal {

namespace {

keel::f64 readReal(const Value &v, keel::usize i) { return v.elemAsDouble(i); }

Value sameOrientReal(const Value &like, keel::usize n, keel::Allocator *mr)
{
    const bool isRow = (like.dims().rows() == 1);
    return isRow
            ? Value::matrix(1, n, ValueType::DOUBLE, mr)
            : Value::matrix(n, 1, ValueType::DOUBLE, mr);
}

void intfilt_argcheck(keel::usize r, keel::usize n, keel::f64 alpha)
{
    if (r < 1)
        throw Error("intfilt: r must be >= 1",
                    0, 0, "intfilt", "", "numkit:intfilt:badR");
    if (n < 1)
        throw Error("intfilt: n must be >= 1",
                    0, 0, "intfilt", "", "numkit:intfilt:badN");
    if (!(alpha > 0.0) || !(alpha <= 1.0))
        throw Error("intfilt: alpha must be in (0, 1]",
                    0, 0, "intfilt", "", "numkit:intfilt:badAlpha");
}

void upfirdn_argcheck(keel::usize p, keel::usize q)
{
    if (p < 1 || q < 1)
        throw Error("upfirdn: p and q must be >= 1",
                    0, 0, "upfirdn", "", "numkit:upfirdn:badPQ");
}

void interp_argcheck(const Value &x, keel::usize n)
{
    if (x.numel() < 2 * n + 1)
        throw Error(keel::format("interp: input 1 (x) length ({}) must be at least 2*n+1 ({})",
                                 x.numel(), 2 * n + 1),
                    0, 0, "interp", "", "numkit:interp:inputTooShort");
}

} // namespace

// ── intfilt (FIR design) ──────────────────────────────────────────────
Value intfilt(keel::usize r, keel::usize n, keel::f64 alpha, keel::Allocator *mr)
{
    intfilt_argcheck(r, n, alpha);

    if (2 * n * r < 1) return Value::matrix(1, 1, ValueType::DOUBLE, mr);
    const keel::i64 Ntaps = keel::as<keel::i64>(2 * n * r) - 1;
    const keel::i32 order = keel::as<keel::i32>(Ntaps - 1);
    const keel::f64 R = keel::as<keel::f64>(r);

    keel::Array<keel::f64> F(mr), M(mr);
    if (alpha == 1.0) {
        M = {R, R, 0.0, 0.0};
        F = {0.0, 1.0 / (2.0 * R), 1.0 / (2.0 * R), 0.5};
    } else {
        const keel::f64 hw = alpha / (2.0 * R);
        M = {R, R};
        F = {0.0, hw};
        for (keel::usize k = 1; keel::as<keel::f64>(k) / R <= 0.5 + 1e-9; ++k) {
            const keel::f64 f = keel::as<keel::f64>(k) / R;
            F.push_back(f - hw);
            F.push_back(f + hw);
            M.push_back(0.0);
            M.push_back(0.0);
        }
        if (F.back() > 0.5) F.back() = 0.5;
    }
    Value Fv = Value::matrix(1, F.size(), ValueType::DOUBLE, mr);
    Value Mv = Value::matrix(1, M.size(), ValueType::DOUBLE, mr);
    keel::f64 *fd = Fv.doubleDataMut();
    keel::f64 *md = Mv.doubleDataMut();
    for (keel::usize i = 0; i < F.size(); ++i) { fd[i] = F[i] * 2.0; md[i] = M[i]; }
    return firls(order, Fv, Mv, mr);
}

// ── upfirdn ───────────────────────────────────────────────────────────
Value upfirdn(const Value &x, const Value &h, keel::usize p, keel::usize q, keel::Allocator *mr)
{
    upfirdn_argcheck(p, q);
    const keel::usize Lx = x.numel();
    const keel::usize Lh = h.numel();
    if (Lx == 0 || Lh == 0) return sameOrientReal(x, 0, mr);

    const keel::usize outLen = keel::signal::upfirdn_length(Lx, Lh, p, q);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(Lx, scratch.allocator());
    keel::Array<keel::f64> hv(Lh, scratch.allocator());
    for (keel::usize i = 0; i < Lx; ++i) xv[i] = readReal(x, i);
    for (keel::usize i = 0; i < Lh; ++i) hv[i] = readReal(h, i);

    auto out = sameOrientReal(x, outLen, mr);
    keel::f64 *dst = out.doubleDataMut();

    // Polyphase reversed-subfilter decomposition: turns every output sample into
    // a single unit-stride contiguous dot product with zero inner modulo/div.
    keel::Array<keel::usize> subOff(p, scratch.allocator());
    keel::Array<keel::usize> subLen(p, scratch.allocator());
    keel::usize totalPacked = 0;
    for (keel::usize ph = 0; ph < p; ++ph) {
        subOff[ph] = totalPacked;
        const keel::usize len = (ph < Lh) ? ((Lh - 1 - ph) / p + 1) : 0;
        subLen[ph] = len;
        totalPacked += len;
    }
    keel::Array<keel::f64> hPoly(totalPacked, scratch.allocator());
    for (keel::usize ph = 0; ph < p; ++ph) {
        const keel::usize len = subLen[ph];
        keel::f64 *hp = hPoly.data() + subOff[ph];
        for (keel::usize u = 0; u < len; ++u)
            hp[u] = hv[ph + (len - 1 - u) * p];
    }

    const keel::f64 *xBase = xv.data();
    const keel::i64 iLx = keel::as<keel::i64>(Lx);
    for (keel::usize m = 0; m < outLen; ++m) {
        const keel::usize n = m * q;
        const keel::usize ph = n % p;
        const keel::usize len = subLen[ph];
        if (len == 0) {
            dst[m] = 0.0;
            continue;
        }
        const keel::i64 kHi = keel::as<keel::i64>(n / p);
        const keel::i64 kStart = kHi - keel::as<keel::i64>(len - 1);
        const keel::i64 uLo = (kStart < 0) ? -kStart : 0;
        const keel::i64 uHi = std::min(keel::as<keel::i64>(len), iLx - kStart);
        if (uLo >= uHi) {
            dst[m] = 0.0;
            continue;
        }
        const keel::f64 *xp = xBase + (kStart + uLo);
        const keel::f64 *hp = hPoly.data() + subOff[ph] + uLo;
        const keel::usize cnt = keel::as<keel::usize>(uHi - uLo);
        keel::f64 a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        keel::usize u = 0;
        for (; u + 4 <= cnt; u += 4) {
            a0 += xp[u]     * hp[u];
            a1 += xp[u + 1] * hp[u + 1];
            a2 += xp[u + 2] * hp[u + 2];
            a3 += xp[u + 3] * hp[u + 3];
        }
        keel::f64 acc = (a0 + a1) + (a2 + a3);
        for (; u < cnt; ++u)
            acc += xp[u] * hp[u];
        dst[m] = acc;
    }
    return out;
}

// ── interp ────────────────────────────────────────────────────────────
Value interp(const Value &x, keel::usize r, keel::usize n, keel::f64 alpha, keel::Allocator *mr)
{
    interp_argcheck(x, n);
    if (r == 1) {
        auto out = sameOrientReal(x, x.numel(), mr);
        std::memcpy(out.doubleDataMut(), x.doubleData(), x.numel() * sizeof(keel::f64));
        return out;
    }
    auto h = intfilt(r, n, alpha, mr);
    auto y = upfirdn(x, h, r, 1, mr);
    const keel::usize target = x.numel() * r;
    const keel::usize leadDelay = n * r;
    auto trimmed = sameOrientReal(x, target, mr);
    keel::f64 *dst = trimmed.doubleDataMut();
    const keel::f64 *src = y.doubleData();
    const keel::usize srcN = y.numel();
    for (keel::usize i = 0; i < target; ++i) {
        const keel::usize j = i + leadDelay;
        dst[i] = (j < srcN) ? src[j] : 0.0;
    }
    return trimmed;
}

// ── fftfilt ───────────────────────────────────────────────────────────
Value fftfilt(const Value &b, const Value &x, keel::usize nfft, keel::Allocator *mr)
{
    const keel::usize nb = b.numel();
    const keel::usize nx = x.numel();
    if (nb == 0 || nx == 0)
        return sameOrientReal(x, 0, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const bool isCx = b.isComplex() || x.isComplex();
    if (isCx) {
        keel::Array<keel::c64> bv(nb, scratch.allocator());
        keel::Array<keel::c64> xv(nx, scratch.allocator());
        for (keel::usize i = 0; i < nb; ++i)
            bv[i] = b.isComplex() ? b.complexElem(i) : keel::c64(readReal(b, i), 0.0);
        for (keel::usize i = 0; i < nx; ++i)
            xv[i] = x.isComplex() ? x.complexElem(i) : keel::c64(readReal(x, i), 0.0);

        const bool row = (x.dims().rows() == 1 && x.dims().cols() >= 1);
        auto out = Value::complexMatrix(row ? 1 : nx, row ? nx : 1, mr);
        (void)keel::signal::fftfilt_c64(bv.as_ref(), xv.as_ref(), nfft,
                                        keel::MutArrayRef<keel::c64>(out.complexDataMut(), nx),
                                        scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> bv(nb, scratch.allocator());
    keel::Array<keel::f64> xv(nx, scratch.allocator());
    for (keel::usize i = 0; i < nb; ++i) bv[i] = readReal(b, i);
    for (keel::usize i = 0; i < nx; ++i) xv[i] = readReal(x, i);

    auto out = sameOrientReal(x, nx, mr);
    (void)keel::signal::fftfilt(bv.as_ref(), xv.as_ref(), nfft,
                                keel::MutArrayRef<keel::f64>(out.doubleDataMut(), nx),
                                scratch.allocator());
    return out;
}

} // namespace numkit::signal
