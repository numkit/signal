// toolboxes/signal/src/smoothing/medfilt.cpp
//
// medfilt1 — sliding-window median. Split from toolboxes/signal/src/.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/signal/smoothing/medfilt.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "../signal_detail.hpp"     // shared argcheck cores

namespace numkit::signal {

namespace {

// Median-filter one length-n signal (src → dst) with an order-k window.
// The window for sample i spans [i-leftHalf .. i+rightHalf] with
// leftHalf = floor(k/2), rightHalf = k-1-leftHalf (so for even k the
// window leans LEFT). With `zeropad` out-of-range samples count as 0
// and the window is always k long; otherwise ('truncate') the window
// is clipped to the valid range.
void medfilt1Column(const keel::f64 *src, keel::f64 *dst, keel::usize n, keel::usize k,
                    keel::usize leftHalf, keel::usize rightHalf, bool zeropad,
                    keel::Allocator *mr)
{
    if (n == 0) return;

    // Incrementally maintain a sorted window of the IN-RANGE samples; each
    // step adds ≤1 on the right and drops ≤1 on the left (O(k) per step,
    // cache-friendly). Pad zeros are folded into the median by rank, never
    // materialised.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto win = keel::Array<keel::f64>(scratch.allocator());
    win.reserve(k);

    keel::usize hi = std::min(n, rightHalf + 1);
    keel::usize lo = 0;
    auto insertSorted = [&](keel::f64 val) {
        const keel::usize idx =
            keel::as<keel::usize>(std::upper_bound(win.begin(), win.end(), val) - win.begin());
        win.push_back(0.0);
        if (idx + 1 < win.size()) {
            keel::mem_move(win.data() + idx + 1, win.data() + idx,
                           (win.size() - 1 - idx) * sizeof(keel::f64));
        }
        win[idx] = val;
    };
    for (keel::usize i = 0; i < hi; ++i) {
        insertSorted(src[i]);
    }

    auto computeMedian = [&]() -> keel::f64 {
        const keel::usize m = win.size();                  // in-range count
        if (!zeropad) {
            if (m == 0) return 0.0;
            if (m & 1) return win[m / 2];
            return 0.5 * (win[m / 2 - 1] + win[m / 2]);
        }
        // Full window is k long: m in-range samples + (k-m) pad zeros.
        const keel::usize nz = k - m;
        const keel::usize nNeg =
            keel::as<keel::usize>(std::lower_bound(win.begin(), win.end(), 0.0) - win.begin());
        const keel::usize nZeroWin =
            keel::as<keel::usize>(std::upper_bound(win.begin(), win.end(), 0.0) - win.begin()) - nNeg;
        const keel::usize zerosTot = nz + nZeroWin;
        auto valAt = [&](keel::usize r) -> keel::f64 {
            if (r < nNeg)              return win[r];        // a negative
            if (r < nNeg + zerosTot)   return 0.0;           // a zero (pad or in-range)
            return win[r - nz];                              // a positive
        };
        if (k & 1) return valAt(k / 2);
        return 0.5 * (valAt(k / 2 - 1) + valAt(k / 2));
    };

    dst[0] = computeMedian();

    for (keel::usize i = 1; i < n; ++i) {
        const keel::usize new_lo = (i >= leftHalf) ? (i - leftHalf) : 0;
        const keel::usize new_hi = std::min(n, i + rightHalf + 1);
        for (keel::usize j = hi; j < new_hi; ++j) {
            insertSorted(src[j]);
        }
        hi = new_hi;
        for (keel::usize j = lo; j < new_lo; ++j) {
            const auto pos = std::lower_bound(win.begin(), win.end(), src[j]);
            if (pos != win.end() && *pos == src[j])
                win.erase(keel::as<keel::usize>(pos - win.begin()));
        }
        lo = new_lo;
        dst[i] = computeMedian();
    }
}

void fillgaps_argcheck(const Value &x, keel::usize maxlen, keel::usize order)
{
    detail::requireFloatInput(x, "fillgaps");
    if (x.isEmpty())
        throw Error("fillgaps: input X must be non-empty",
                    0, 0, "fillgaps", "", "numkit:fillgaps:emptyInput");
    if (x.dims().ndims() > 2)
        throw Error("fillgaps: input X must be a vector or 2-D matrix",
                    0, 0, "fillgaps", "", "numkit:fillgaps:expected2D");
    if (maxlen > 0 && order > 0 && order >= maxlen)
        throw Error(keel::format("fillgaps: ORDER must be strictly less than MAXLEN (got order={}, maxlen={})",
                                 order, maxlen),
                    0, 0, "fillgaps", "", "numkit:fillgaps:orderTooLarge");
}

inline bool isNanScalar(keel::f64 v) { return std::isnan(v); }
inline bool isNanScalar(keel::c64 v) { return std::isnan(v.real()) || std::isnan(v.imag()); }
inline keel::f64 normScalar(keel::f64 v) { return v * v; }
inline keel::f64 normScalar(keel::c64 v) { return keel::norm(v); }
inline keel::f64 conjScalar(keel::f64 v) { return v; }
inline keel::c64 conjScalar(keel::c64 v) { return keel::conj(v); }

template <typename Scalar>
keel::Array<Scalar> fillgapsFwdPass(const Scalar *col, keel::usize n,
                                    keel::usize maxlen, keel::usize order)
{
    keel::Array<Scalar> v(keel::ArrayRef<Scalar>(col, n));
    keel::usize i_first = n;
    for (keel::usize i = 0; i < n; ++i) {
        if (!isNanScalar(v[i])) {
            i_first = i;
            break;
        }
    }
    if (i_first == n) return v;

    keel::usize k = i_first + 1;
    while (k < n) {
        if (!isNanScalar(v[k])) {
            ++k;
            continue;
        }
        keel::usize g_end = k;
        while (g_end + 1 < n && isNanScalar(v[g_end + 1])) ++g_end;
        const keel::usize G = g_end - k + 1;

        const keel::usize s_start = (maxlen > 0 && (k - i_first) > maxlen)
            ? (k - maxlen) : i_first;
        const keel::usize L = k - s_start;

        Scalar mu{};
        for (keel::usize i = s_start; i < k; ++i) mu += v[i];
        mu /= keel::as<keel::f64>(L);

        keel::Array<Scalar> seg0(L);
        keel::f64 e0 = 0.0;
        for (keel::usize i = 0; i < L; ++i) {
            seg0[i] = v[s_start + i] - mu;
            e0 += normScalar(seg0[i]);
        }
        e0 /= keel::as<keel::f64>(L);

        if (L <= 1 || !(e0 > 0.0)) {
            for (keel::usize m = 0; m < G; ++m) v[k + m] = mu;
            k = g_end + 1;
            continue;
        }

        const keel::usize max_p = (order > 0) ? std::min(order, L - 1) : (L - 1);
        keel::Array<Scalar> ef = seg0.clone();
        keel::Array<Scalar> eb = seg0.clone();
        keel::Array<Scalar> a(max_p + 1, Scalar{});
        a[0] = Scalar(1.0);
        keel::f64 E = e0;

        keel::usize best_p = max_p;
        keel::f64 best_aic = std::numeric_limits<keel::f64>::infinity();
        keel::Array<Scalar> best_a;

        for (keel::usize m = 1; m <= max_p; ++m) {
            Scalar num{};
            keel::f64 den = 0.0;
            for (keel::usize idx = m; idx < L; ++idx) {
                num += ef[idx] * conjScalar(eb[idx - 1]);
                den += normScalar(ef[idx]) + normScalar(eb[idx - 1]);
            }
            const Scalar km = (den > 0.0) ? (Scalar(-2.0) * num / den) : Scalar{};

            for (keel::usize idx = L - 1; idx >= m; --idx) {
                const Scalar f = ef[idx];
                const Scalar b = eb[idx - 1];
                ef[idx] = f + km * b;
                eb[idx] = b + conjScalar(km) * f;
            }

            keel::Array<Scalar> a_next = a.clone();
            for (keel::usize j = 1; j < m; ++j)
                a_next[j] = a[j] + km * conjScalar(a[m - j]);
            a_next[m] = km;
            a = std::move(a_next);

            E *= std::max(0.0, 1.0 - normScalar(km));

            if (order == 0) {
                const keel::f64 aic_m =
                    keel::as<keel::f64>(L) * std::log(std::max(E, 1e-300)) +
                    2.0 * keel::as<keel::f64>(m);
                if (m == 1 || aic_m < best_aic) {
                    best_aic = aic_m;
                    best_p = m;
                    best_a.assign(a.slice(0, m + 1));
                } else if (m - best_p >= 31) {
                    break;
                }
            }
        }

        if (order > 0) {
            best_p = max_p;
            best_a = std::move(a);
        }

        keel::Array<Scalar> z(best_p + G);
        for (keel::usize i = 0; i < best_p; ++i)
            z[i] = seg0[L - best_p + i];
        for (keel::usize m = 0; m < G; ++m) {
            Scalar acc{};
            for (keel::usize j = 1; j <= best_p; ++j)
                acc -= best_a[j] * z[best_p + m - j];
            z[best_p + m] = acc;
            v[k + m] = acc + mu;
        }

        k = g_end + 1;
    }
    return v;
}

template <typename Scalar>
void fillgapsColumn(const Scalar *src, Scalar *dst, keel::usize n,
                    keel::usize maxlen, keel::usize order)
{
    if (n == 0) return;
    keel::usize i_first = n, i_last = n;
    for (keel::usize i = 0; i < n; ++i) {
        if (!isNanScalar(src[i])) {
            if (i_first == n) i_first = i;
            i_last = i;
        }
    }
    if (i_first == n) {
        std::copy(src, src + n, dst);
        return;
    }

    keel::Array<Scalar> yf = fillgapsFwdPass(src, n, maxlen, order);
    keel::Array<Scalar> rev(n);
    for (keel::usize i = 0; i < n; ++i) rev[i] = src[n - 1 - i];
    keel::Array<Scalar> yb_rev = fillgapsFwdPass(rev.data(), n, maxlen, order);
    keel::Array<Scalar> yb(n);
    for (keel::usize i = 0; i < n; ++i) yb[i] = yb_rev[n - 1 - i];

    std::copy(src, src + n, dst);
    for (keel::usize i = 0; i < i_first; ++i) dst[i] = yb[i];
    for (keel::usize i = i_last + 1; i < n; ++i) dst[i] = yf[i];

    keel::usize k = i_first + 1;
    while (k < i_last) {
        if (!isNanScalar(src[k])) {
            ++k;
            continue;
        }
        keel::usize g_end = k;
        while (g_end + 1 < i_last && isNanScalar(src[g_end + 1])) ++g_end;
        const keel::usize G = g_end - k + 1;
        for (keel::usize m = 0; m < G; ++m) {
            const keel::f64 w = keel::as<keel::f64>(m + 1) / keel::as<keel::f64>(G + 1);
            dst[k + m] = (1.0 - w) * yf[k + m] + w * yb[k + m];
        }
        k = g_end + 1;
    }
}

} // namespace

Value medfilt1(const Value &x, keel::usize k, bool zeropad, keel::Allocator *mr)
{
    detail::requireRealFloatInput(x, "medfilt1");
    const bool xSingle = (x.type() == ValueType::SINGLE);

    const keel::usize total = x.numel();
    auto r = createLike(x, ValueType::DOUBLE, mr);
    if (total == 0) return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
    // k == 0: the reference answers an all-NaN signal (median of an
    // empty window per sample) instead of rejecting.
    if (k == 0) {
        keel::f64 *dst = r.doubleDataMut();
        const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
        for (keel::usize i = 0; i < total; ++i) dst[i] = nan;
        return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
    }

    const keel::usize leftHalf  = k / 2;            // even-k window leans left
    const keel::usize rightHalf = k - 1 - leftHalf;

    Value xD = detail::widenSingleToDouble(x, mr);
    const keel::f64 *src = xD.doubleData();
    keel::f64 *dst = r.doubleDataMut();

    const keel::usize R = keel::as<keel::usize>(xD.dims().dim(0));
    // Vector / scalar → one signal over every element. True 2-D (or N-D)
    // matrix → median-filter each column (length R) independently.
    if (R <= 1 || xD.dims().isVector()) {
        medfilt1Column(src, dst, total, k, leftHalf, rightHalf, zeropad, mr);
    } else {
        const keel::usize ncols = total / R;
        for (keel::usize c = 0; c < ncols; ++c)
            medfilt1Column(src + c * R, dst + c * R, R, k, leftHalf, rightHalf, zeropad, mr);
    }
    return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
}

Value fillgaps(const Value &x, keel::usize maxlen, keel::usize order, keel::Allocator *mr)
{
    fillgaps_argcheck(x, maxlen, order);
    const bool xSingle = (x.type() == ValueType::SINGLE);
    const keel::usize total = x.numel();

    if (x.type() == ValueType::COMPLEX) {
        auto r = createLike(x, ValueType::COMPLEX, mr);
        const keel::c64 *src = x.complexData();
        keel::c64 *dst = r.complexDataMut();
        const keel::usize R = x.dims().rows();
        if (R <= 1 || x.dims().isVector()) {
            fillgapsColumn(src, dst, total, maxlen, order);
        } else {
            const keel::usize ncols = total / R;
            for (keel::usize c = 0; c < ncols; ++c)
                fillgapsColumn(src + c * R, dst + c * R, R, maxlen, order);
        }
        return r;
    }

    Value xD = detail::widenSingleToDouble(x, mr);
    auto r = createLike(x, ValueType::DOUBLE, mr);
    const keel::f64 *src = xD.doubleData();
    keel::f64 *dst = r.doubleDataMut();
    const keel::usize R = xD.dims().rows();
    if (R <= 1 || xD.dims().isVector()) {
        fillgapsColumn(src, dst, total, maxlen, order);
    } else {
        const keel::usize ncols = total / R;
        for (keel::usize c = 0; c < ncols; ++c)
            fillgapsColumn(src + c * R, dst + c * R, R, maxlen, order);
    }
    return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
}

} // namespace numkit::signal
