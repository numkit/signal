// toolboxes/signal/src/smoothing/medfilt.cpp
//
// medfilt1 — sliding-window median. Split from toolboxes/signal/src/.

#include <numkit/signal/smoothing/medfilt.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "../signal_detail.hpp"     // shared argcheck cores

namespace numkit::signal {

namespace {

// Median-filter one length-n signal (src → dst) with an order-k window.
// The window for sample i spans [i-leftHalf .. i+rightHalf] with
// leftHalf = floor(k/2), rightHalf = k-1-leftHalf (so for even k the
// window leans LEFT). With `zeropad` out-of-range samples count as 0
// and the window is always k long; otherwise ('truncate') the window
// is clipped to the valid range.
void medfilt1Column(const double *src, double *dst, size_t n, size_t k,
                    size_t leftHalf, size_t rightHalf, bool zeropad,
                    std::pmr::memory_resource *mr)
{
    if (n == 0) return;

    // Incrementally maintain a sorted window of the IN-RANGE samples; each
    // step adds ≤1 on the right and drops ≤1 on the left (O(k) per step,
    // cache-friendly). Pad zeros are folded into the median by rank, never
    // materialised.
    ScratchArena scratch(mr);
    auto win = ScratchVec<double>(&scratch);
    win.reserve(k);

    size_t hi = std::min(n, rightHalf + 1);
    size_t lo = 0;
    for (size_t i = 0; i < hi; ++i) {
        const auto pos = std::upper_bound(win.begin(), win.end(), src[i]);
        win.insert(pos, src[i]);
    }

    auto computeMedian = [&]() -> double {
        const size_t m = win.size();                  // in-range count
        if (!zeropad) {
            if (m == 0) return 0.0;
            if (m & 1) return win[m / 2];
            return 0.5 * (win[m / 2 - 1] + win[m / 2]);
        }
        // Full window is k long: m in-range samples + (k-m) pad zeros.
        const size_t nz = k - m;
        const size_t nNeg =
            static_cast<size_t>(std::lower_bound(win.begin(), win.end(), 0.0) - win.begin());
        const size_t nZeroWin =
            static_cast<size_t>(std::upper_bound(win.begin(), win.end(), 0.0) - win.begin()) - nNeg;
        const size_t zerosTot = nz + nZeroWin;
        auto valAt = [&](size_t r) -> double {
            if (r < nNeg)              return win[r];        // a negative
            if (r < nNeg + zerosTot)   return 0.0;           // a zero (pad or in-range)
            return win[r - nz];                              // a positive
        };
        if (k & 1) return valAt(k / 2);
        return 0.5 * (valAt(k / 2 - 1) + valAt(k / 2));
    };

    dst[0] = computeMedian();

    for (size_t i = 1; i < n; ++i) {
        const size_t new_lo = (i >= leftHalf) ? (i - leftHalf) : 0;
        const size_t new_hi = std::min(n, i + rightHalf + 1);
        for (size_t j = hi; j < new_hi; ++j) {
            const auto pos = std::upper_bound(win.begin(), win.end(), src[j]);
            win.insert(pos, src[j]);
        }
        hi = new_hi;
        for (size_t j = lo; j < new_lo; ++j) {
            const auto pos = std::lower_bound(win.begin(), win.end(), src[j]);
            if (pos != win.end() && *pos == src[j]) win.erase(pos);
        }
        lo = new_lo;
        dst[i] = computeMedian();
    }
}

void fillgaps_argcheck(const Value &x, size_t maxlen, size_t order)
{
    detail::requireFloatInput(x, "fillgaps");
    if (x.isEmpty())
        throw Error("fillgaps: input X must be non-empty",
                    0, 0, "fillgaps", "", "numkit:fillgaps:emptyInput");
    if (x.dims().ndims() > 2)
        throw Error("fillgaps: input X must be a vector or 2-D matrix",
                    0, 0, "fillgaps", "", "numkit:fillgaps:expected2D");
    if (maxlen > 0 && order > 0 && order >= maxlen)
        throw Error("fillgaps: ORDER must be strictly less than MAXLEN (got order=" +
                    std::to_string(order) + ", maxlen=" + std::to_string(maxlen) + ")",
                    0, 0, "fillgaps", "", "numkit:fillgaps:orderTooLarge");
}

inline bool isNanScalar(double v) { return std::isnan(v); }
inline bool isNanScalar(Complex v) { return std::isnan(v.real()) || std::isnan(v.imag()); }
inline double normScalar(double v) { return v * v; }
inline double normScalar(Complex v) { return std::norm(v); }
inline double conjScalar(double v) { return v; }
inline Complex conjScalar(Complex v) { return std::conj(v); }

template <typename Scalar>
std::vector<Scalar> fillgapsFwdPass(const Scalar *col, size_t n,
                                    size_t maxlen, size_t order)
{
    std::vector<Scalar> v(col, col + n);
    size_t i_first = n;
    for (size_t i = 0; i < n; ++i) {
        if (!isNanScalar(v[i])) {
            i_first = i;
            break;
        }
    }
    if (i_first == n) return v;

    size_t k = i_first + 1;
    while (k < n) {
        if (!isNanScalar(v[k])) {
            ++k;
            continue;
        }
        size_t g_end = k;
        while (g_end + 1 < n && isNanScalar(v[g_end + 1])) ++g_end;
        const size_t G = g_end - k + 1;

        const size_t s_start = (maxlen > 0 && (k - i_first) > maxlen)
            ? (k - maxlen) : i_first;
        const size_t L = k - s_start;

        Scalar mu{};
        for (size_t i = s_start; i < k; ++i) mu += v[i];
        mu /= static_cast<double>(L);

        std::vector<Scalar> seg0(L);
        double e0 = 0.0;
        for (size_t i = 0; i < L; ++i) {
            seg0[i] = v[s_start + i] - mu;
            e0 += normScalar(seg0[i]);
        }
        e0 /= static_cast<double>(L);

        if (L <= 1 || !(e0 > 0.0)) {
            for (size_t m = 0; m < G; ++m) v[k + m] = mu;
            k = g_end + 1;
            continue;
        }

        const size_t max_p = (order > 0) ? std::min(order, L - 1) : (L - 1);
        std::vector<Scalar> ef = seg0;
        std::vector<Scalar> eb = seg0;
        std::vector<Scalar> a(max_p + 1, Scalar{});
        a[0] = Scalar(1.0);
        double E = e0;

        size_t best_p = max_p;
        double best_aic = std::numeric_limits<double>::infinity();
        std::vector<Scalar> best_a;

        for (size_t m = 1; m <= max_p; ++m) {
            Scalar num{};
            double den = 0.0;
            for (size_t idx = m; idx < L; ++idx) {
                num += ef[idx] * conjScalar(eb[idx - 1]);
                den += normScalar(ef[idx]) + normScalar(eb[idx - 1]);
            }
            const Scalar km = (den > 0.0) ? (Scalar(-2.0) * num / den) : Scalar{};

            for (size_t idx = L - 1; idx >= m; --idx) {
                const Scalar f = ef[idx];
                const Scalar b = eb[idx - 1];
                ef[idx] = f + km * b;
                eb[idx] = b + conjScalar(km) * f;
            }

            std::vector<Scalar> a_next = a;
            for (size_t j = 1; j < m; ++j)
                a_next[j] = a[j] + km * conjScalar(a[m - j]);
            a_next[m] = km;
            a = std::move(a_next);

            E *= std::max(0.0, 1.0 - normScalar(km));

            if (order == 0) {
                const double aic_m =
                    static_cast<double>(L) * std::log(std::max(E, 1e-300)) +
                    2.0 * static_cast<double>(m);
                if (m == 1 || aic_m < best_aic) {
                    best_aic = aic_m;
                    best_p = m;
                    best_a.assign(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(m + 1));
                } else if (m - best_p >= 31) {
                    break;
                }
            }
        }

        if (order > 0) {
            best_p = max_p;
            best_a = std::move(a);
        }

        std::vector<Scalar> z(best_p + G);
        for (size_t i = 0; i < best_p; ++i)
            z[i] = seg0[L - best_p + i];
        for (size_t m = 0; m < G; ++m) {
            Scalar acc{};
            for (size_t j = 1; j <= best_p; ++j)
                acc -= best_a[j] * z[best_p + m - j];
            z[best_p + m] = acc;
            v[k + m] = acc + mu;
        }

        k = g_end + 1;
    }
    return v;
}

template <typename Scalar>
void fillgapsColumn(const Scalar *src, Scalar *dst, size_t n,
                    size_t maxlen, size_t order)
{
    if (n == 0) return;
    size_t i_first = n, i_last = n;
    for (size_t i = 0; i < n; ++i) {
        if (!isNanScalar(src[i])) {
            if (i_first == n) i_first = i;
            i_last = i;
        }
    }
    if (i_first == n) {
        std::copy(src, src + n, dst);
        return;
    }

    std::vector<Scalar> yf = fillgapsFwdPass(src, n, maxlen, order);
    std::vector<Scalar> rev(n);
    for (size_t i = 0; i < n; ++i) rev[i] = src[n - 1 - i];
    std::vector<Scalar> yb_rev = fillgapsFwdPass(rev.data(), n, maxlen, order);
    std::vector<Scalar> yb(n);
    for (size_t i = 0; i < n; ++i) yb[i] = yb_rev[n - 1 - i];

    std::copy(src, src + n, dst);
    for (size_t i = 0; i < i_first; ++i) dst[i] = yb[i];
    for (size_t i = i_last + 1; i < n; ++i) dst[i] = yf[i];

    size_t k = i_first + 1;
    while (k < i_last) {
        if (!isNanScalar(src[k])) {
            ++k;
            continue;
        }
        size_t g_end = k;
        while (g_end + 1 < i_last && isNanScalar(src[g_end + 1])) ++g_end;
        const size_t G = g_end - k + 1;
        for (size_t m = 0; m < G; ++m) {
            const double w = static_cast<double>(m + 1) / static_cast<double>(G + 1);
            dst[k + m] = (1.0 - w) * yf[k + m] + w * yb[k + m];
        }
        k = g_end + 1;
    }
}

} // namespace

Value medfilt1(const Value &x, size_t k, bool zeropad, std::pmr::memory_resource *mr)
{
    detail::requireRealFloatInput(x, "medfilt1");
    const bool xSingle = (x.type() == ValueType::SINGLE);

    const size_t total = x.numel();
    auto r = createLike(x, ValueType::DOUBLE, mr);
    if (total == 0) return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
    // k == 0: the reference answers an all-NaN signal (median of an
    // empty window per sample) instead of rejecting.
    if (k == 0) {
        double *dst = r.doubleDataMut();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (size_t i = 0; i < total; ++i) dst[i] = nan;
        return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
    }

    const size_t leftHalf  = k / 2;            // even-k window leans left
    const size_t rightHalf = k - 1 - leftHalf;

    Value xD = detail::widenSingleToDouble(x, mr);
    const double *src = xD.doubleData();
    double *dst = r.doubleDataMut();

    const size_t R = static_cast<size_t>(xD.dims().dim(0));
    // Vector / scalar → one signal over every element. True 2-D (or N-D)
    // matrix → median-filter each column (length R) independently.
    if (R <= 1 || xD.dims().isVector()) {
        medfilt1Column(src, dst, total, k, leftHalf, rightHalf, zeropad, mr);
    } else {
        const size_t ncols = total / R;
        for (size_t c = 0; c < ncols; ++c)
            medfilt1Column(src + c * R, dst + c * R, R, k, leftHalf, rightHalf, zeropad, mr);
    }
    return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
}

Value fillgaps(const Value &x, size_t maxlen, size_t order, std::pmr::memory_resource *mr)
{
    fillgaps_argcheck(x, maxlen, order);
    const bool xSingle = (x.type() == ValueType::SINGLE);
    const size_t total = x.numel();

    if (x.type() == ValueType::COMPLEX) {
        auto r = createLike(x, ValueType::COMPLEX, mr);
        const Complex *src = x.complexData();
        Complex *dst = r.complexDataMut();
        const size_t R = x.dims().rows();
        if (R <= 1 || x.dims().isVector()) {
            fillgapsColumn(src, dst, total, maxlen, order);
        } else {
            const size_t ncols = total / R;
            for (size_t c = 0; c < ncols; ++c)
                fillgapsColumn(src + c * R, dst + c * R, R, maxlen, order);
        }
        return r;
    }

    Value xD = detail::widenSingleToDouble(x, mr);
    auto r = createLike(x, ValueType::DOUBLE, mr);
    const double *src = xD.doubleData();
    double *dst = r.doubleDataMut();
    const size_t R = xD.dims().rows();
    if (R <= 1 || xD.dims().isVector()) {
        fillgapsColumn(src, dst, total, maxlen, order);
    } else {
        const size_t ncols = total / R;
        for (size_t c = 0; c < ncols; ++c)
            fillgapsColumn(src + c * R, dst + c * R, R, maxlen, order);
    }
    return detail::narrowDoubleToSingleIf(std::move(r), xSingle, mr);
}

} // namespace numkit::signal
