// toolboxes/signal/src/measurements/sig_utils.cpp
//
// Signal Processing Toolbox small utilities (cycle 5):
//
//   seqperiod(x [, tol])           Smallest period d ≤ N such that
//                                  x(i) ≈ x(((i-1) mod d) + 1) for all i.
//                                  Returns [p, nr] where nr = N/p.
//   zerocrossrate(x [, level])     Zero-crossing rate counting boundary
//                                  half-crossings. count = #crossings
//                                  + 0.5; rate = count / numel(x).
//                                  v1: vector input only, default
//                                  level=0, no Name=Value args.
//   cusum(x [, climit, mshift, tmean, tdev])
//                                  CUSUM control-chart change detector.
//                                  Returns [iupper, ilower] of first
//                                  out-of-control indices. v1 covers
//                                  [iupper, ilower] form (4-out form
//                                  with sums also supported).
//
// PMR HARD RULE: every fn takes std::pmr::memory_resource *mr.
//
// KNOWN GAPs:
//   * zerocrossrate: matrix/N-D input + Name=Value args
//     ('Threshold', 'TransitionEdge', 'WindowLength') deferred.
//   * cusum: 'all' string flag and the no-output plotting form
//     deferred.
//   * seqperiod: matrix/N-D operates column-wise — only vector
//     supported in v1.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/signal/measurements/sig_utils.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>

namespace numkit::signal {

namespace {

void cusum_argcheck(const Value &x, keel::f64 climit, keel::f64 mshift)
{
    (void)x;
    (void)climit;
    (void)mshift;
}

} // namespace

// ── seqperiod ─────────────────────────────────────────────────────────
keel::Pair<Value, Value>
seqperiod(const Value &x, keel::f64 tol, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (N == 0) {
        return {Value::scalar(0.0, mr), Value::scalar(0.0, mr)};
    }
    // Find smallest d such that |x(i) - x(((i-1) mod d) + 1)| ≤ tol for all i.
    keel::usize period = N;
    for (keel::usize d = 1; d <= N; ++d) {
        bool ok = true;
        for (keel::usize i = d; i < N; ++i) {
            const keel::f64 dv = keel::abs(x.elemAsDouble(i) - x.elemAsDouble(i % d));
            if (dv > tol) { ok = false; break; }
        }
        if (ok) { period = d; break; }
    }
    return {Value::scalar(keel::as<keel::f64>(period), mr),
            Value::scalar(keel::as<keel::f64>(N) / keel::as<keel::f64>(period), mr)};
}

// ── zerocrossrate ─────────────────────────────────────────────────────
// Count sign changes relative to `level` (default 0); add 0.5 boundary
// half-credit (matches MATLAB R2025b default ZeroPositive=false).
keel::Pair<Value, Value>
zerocrossrate(const Value &x, keel::f64 level, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (N <= 1) {
        // Edge: no transitions possible; rate = 0.5 / N (boundary credit).
        const keel::f64 cnt = (N == 0) ? 0.0 : 0.5;
        return {Value::scalar(N == 0 ? 0.0 : cnt / keel::as<keel::f64>(N), mr),
                Value::scalar(cnt, mr)};
    }
    keel::i64 crossings = 0;
    keel::f64 prev = x.elemAsDouble(0) - level;
    keel::i32 prev_sign = (prev > 0.0) ? 1 : (prev < 0.0 ? -1 : 0);
    for (keel::usize i = 1; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i) - level;
        const keel::i32 s = (v > 0.0) ? 1 : (v < 0.0 ? -1 : 0);
        // Count any flip including via zero (treat zero as continuation,
        // matching MATLAB ZeroPositive=false: zero stays at previous sign).
        if (s != 0 && prev_sign != 0 && s != prev_sign) ++crossings;
        if (s != 0) prev_sign = s;
    }
    const keel::f64 count = keel::as<keel::f64>(crossings) + 0.5;
    const keel::f64 rate  = count / keel::as<keel::f64>(N);
    return {Value::scalar(rate, mr), Value::scalar(count, mr)};
}

// ── cusum ─────────────────────────────────────────────────────────────
// Standard CUSUM detector (see measurements/sig_utils.hpp for the public
// API + CusumResult). Returns first indices where each one-sided cumulative
// sum exceeds the climit threshold (in standard-deviation units).

CusumResult cusum(const Value &x, keel::f64 climit, keel::f64 mshift,
                  const Value &tmean, const Value &tdev,
                  keel::Allocator *mr)
{
    cusum_argcheck(x, climit, mshift);
    const bool have_tmean   = !tmean.isEmpty();
    const bool have_tdev    = !tdev.isEmpty();
    const keel::f64 tmean_user = have_tmean ? tmean.toScalar() : 0.0;
    const keel::f64 tdev_user  = have_tdev  ? tdev.toScalar()  : 0.0;
    const keel::usize N = x.numel();
    CusumResult R;
    R.uppersum = Value::matrix(N, N == 0 ? 0 : 1, ValueType::DOUBLE, mr);
    R.lowersum = Value::matrix(N, N == 0 ? 0 : 1, ValueType::DOUBLE, mr);
    R.iupper = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    R.ilower = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (N == 0) return R;

    // Defaults: tmean = mean(x(1:25)), tdev = std(x(1:25)).
    const keel::usize baseN = std::min<keel::usize>(25, N);
    keel::f64 tmeanEff = tmean_user;
    keel::f64 tdevEff  = tdev_user;
    if (!have_tmean) {
        keel::f64 sum = 0.0;
        for (keel::usize i = 0; i < baseN; ++i) sum += x.elemAsDouble(i);
        tmeanEff = sum / keel::as<keel::f64>(baseN);
    }
    if (!have_tdev) {
        keel::f64 sum = 0.0, sumsq = 0.0;
        for (keel::usize i = 0; i < baseN; ++i) {
            const keel::f64 v = x.elemAsDouble(i);
            sum += v; sumsq += v * v;
        }
        const keel::f64 m = sum / keel::as<keel::f64>(baseN);
        const keel::f64 var = (sumsq - keel::as<keel::f64>(baseN) * m * m)
                          / keel::as<keel::f64>(baseN > 1 ? baseN - 1 : 1);
        tdevEff = keel::sqrt(std::max(0.0, var));
    }
    if (tdevEff <= 0.0) tdevEff = 1.0;  // guard

    keel::f64 *us = R.uppersum.doubleDataMut();
    keel::f64 *ls = R.lowersum.doubleDataMut();
    keel::f64 up = 0.0, lo = 0.0;
    keel::i64 iup_first = -1, ilo_first = -1;
    const keel::f64 half_shift = 0.5 * mshift;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 z = (x.elemAsDouble(i) - tmeanEff) / tdevEff;
        up = std::max(0.0, up + z - half_shift);
        lo = std::max(0.0, lo - z - half_shift);
        us[i] = up;
        ls[i] = lo;
        if (iup_first < 0 && up > climit) iup_first = keel::as<keel::i64>(i + 1);
        if (ilo_first < 0 && lo > climit) ilo_first = keel::as<keel::i64>(i + 1);
    }
    if (iup_first > 0) R.iupper = Value::scalar(keel::as<keel::f64>(iup_first), mr);
    if (ilo_first > 0) R.ilower = Value::scalar(keel::as<keel::f64>(ilo_first), mr);
    return R;
}

} // namespace numkit::signal
