// toolboxes/signal/src/measurements/vibration.cpp
//
// envspectrum / tachorpm / rainflow / tsa. Higher-order vibration
// functions (modal*, order tracking, rpm*maps) are not implemented
// here — they need a full state-space / RPM-resampling stack.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/measurements/vibration.hpp>

#include <numkit/signal/transforms/fft.hpp>
#include <numkit/signal/transforms/hilbert.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace {

keel::Array<double> readVec(const Value &x)
{
    const keel::usize n = x.numel();
    keel::Array<double> v(n);
    for (keel::usize i = 0; i < n; ++i) v[i] = x.elemAsDouble(i);
    return v;
}

Value vecCol(keel::Allocator *mr, keel::ArrayRef<double> v)
{
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty())
        std::memcpy(out.doubleDataMut(), v.data(), v.size() * sizeof(double));
    return out;
}

void tachorpm_argcheck(const Value &x, double fs, int ppr)
{
    (void)x;
    (void)fs;
    (void)ppr;
}

} // anonymous

// ── envspectrum ────────────────────────────────────────────────────

std::tuple<Value, Value>
envspectrum(const Value &x, double fs, keel::Allocator *mr)
{
    const keel::usize N = x.numel();
    if (N == 0)
        return std::make_tuple(Value::matrix(0, 1, ValueType::DOUBLE, mr),
                               Value::matrix(0, 1, ValueType::DOUBLE, mr));

    // 1. Analytic-signal envelope.
    Value env = envelope(x, mr);
    // 2. AC-couple by subtracting the mean.
    auto e = readVec(env);
    double m = 0.0;
    for (double v : e) m += v;
    m /= e.size();
    for (auto &v : e) v -= m;

    // 3. FFT of the AC envelope, magnitude on the one-sided half.
    Value envAc = vecCol(mr, e);
    Value Z = fft(envAc, /*n=*/-1, /*dim=*/0, mr);
    const keel::usize nFft = Z.numel();
    const keel::usize nOne = nFft / 2 + 1;

    auto Es = Value::matrix(nOne, 1, ValueType::DOUBLE, mr);
    auto F  = Value::matrix(nOne, 1, ValueType::DOUBLE, mr);
    double *esd = Es.doubleDataMut();
    double *fd  = F.doubleDataMut();
    const keel::c64 *zd = Z.complexData();

    const double fNyq = (fs > 0.0) ? fs * 0.5 : M_PI;
    const double scale = 2.0 / keel::as<double>(N);  // single-sided amp.
    for (keel::usize i = 0; i < nOne; ++i) {
        const double a = keel::abs(zd[i]) * scale;
        esd[i] = (i == 0 || i == nOne - 1) ? 0.5 * a : a;
        fd[i]  = fNyq * keel::as<double>(i) / keel::as<double>(nOne - 1);
    }
    return std::make_tuple(std::move(Es), std::move(F));
}

// ── tachorpm ───────────────────────────────────────────────────────

std::tuple<Value, Value>
tachorpm(const Value &x, double fs,
         const Value &threshold, int ppr, keel::Allocator *mr)
{
    tachorpm_argcheck(x, fs, ppr);
    auto v = readVec(x);
    const keel::usize N = v.size();
    double thr;
    if (!threshold.isEmpty()) {
        thr = threshold.toScalar();
    } else {
        double lo = v.empty() ? 0.0 : v[0], hi = lo;
        for (double y : v) { if (y < lo) lo = y; if (y > hi) hi = y; }
        thr = 0.5 * (lo + hi);
    }
    // Detect rising-edge crossings of `thr`. Linear-interpolate the
    // sub-sample crossing instant for each.
    keel::Array<double> times(mr);
    for (keel::usize i = 0; i + 1 < N; ++i) {
        if (v[i] < thr && v[i + 1] >= thr) {
            const double a = v[i], b = v[i + 1];
            const double frac = (b == a) ? 0.0 : (thr - a) / (b - a);
            times.push_back((i + frac) / fs);
        }
    }
    // RPM from inter-pulse periods. With `ppr` pulses per revolution,
    // RPM = 60 / (period * ppr). Output one RPM per pulse (carry the
    // last estimate forward to the final pulse).
    keel::Array<double> rpm(times.size(), 0.0, mr);
    for (keel::usize i = 0; i + 1 < times.size(); ++i) {
        const double dt = times[i + 1] - times[i];
        if (dt > 0.0) rpm[i] = 60.0 / (dt * ppr);
    }
    if (rpm.size() >= 2) rpm.back() = rpm[rpm.size() - 2];

    return std::make_tuple(vecCol(mr, rpm), vecCol(mr, times));
}

// ── rainflow ───────────────────────────────────────────────────────
// ASTM E1049-85 cycle counting. Implementation follows the standard
// "four-point" rule applied to a sequence of peaks-and-valleys.

namespace {

// Turning points + their original-signal indices (1-based, MATLAB
// convention).
struct Turn { double v; double idx; };

keel::Array<Turn> turningPointsWithIndex(keel::ArrayRef<double> v)
{
    keel::Array<Turn> out;
    if (v.empty()) return out;
    out.push_back({v.front(), 1.0});
    for (keel::usize i = 1; i + 1 < v.size(); ++i) {
        const double a = v[i - 1], b = v[i], c = v[i + 1];
        if ((b > a && b > c) || (b < a && b < c))
            out.push_back({b, keel::as<double>(i + 1)});
    }
    out.push_back({v.back(), keel::as<double>(v.size())});
    return out;
}

} // anonymous

Value rainflow(const Value &x, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto tp = turningPointsWithIndex(v);

    // ASTM E1049-85 four-point algorithm. Each cycle yields a row
    //   [count, range, mean, start_idx, end_idx]
    // matching MATLAB R2025b's rainflow() return shape.
    struct Cycle { double count, range, mean, idx0, idx1; };
    keel::Array<Cycle> cycles(mr);
    keel::Array<Turn> S(mr);
    for (const auto &y : tp) {
        S.push_back(y);
        // Apply the rule until no more cycles can be extracted.
        while (S.size() >= 3) {
            const keel::usize n = S.size();
            const double X = keel::abs(S[n - 1].v - S[n - 2].v);
            const double Y = keel::abs(S[n - 2].v - S[n - 3].v);
            if (X < Y) break;
            if (n == 3) {
                // Half cycle: pop the bottom of the stack.
                const double rng = keel::abs(S[0].v - S[1].v);
                const double mn  = 0.5 * (S[0].v + S[1].v);
                cycles.push_back({0.5, rng, mn, S[0].idx, S[1].idx});
                S.remove_at(0);
                break;
            }
            // Full cycle on the inner pair.
            const double rng = Y;
            const double mn  = 0.5 * (S[n - 2].v + S[n - 3].v);
            cycles.push_back({1.0, rng, mn, S[n - 3].idx, S[n - 2].idx});
            S[n - 3] = S[n - 1];
            S.resize(n - 2);
        }
    }
    // Drain residual: every consecutive pair on the residual stack is
    // a half-cycle.
    for (keel::usize i = 0; i + 1 < S.size(); ++i) {
        cycles.push_back({0.5,
                          keel::abs(S[i].v - S[i + 1].v),
                          0.5 * (S[i].v + S[i + 1].v),
                          S[i].idx,
                          S[i + 1].idx});
    }
    auto out = Value::matrix(cycles.size(), 5, ValueType::DOUBLE, mr);
    if (cycles.empty()) return out;
    double *d = out.doubleDataMut();
    const keel::usize rows = cycles.size();
    for (keel::usize i = 0; i < rows; ++i) {
        d[i + 0 * rows] = cycles[i].count;
        d[i + 1 * rows] = cycles[i].range;
        d[i + 2 * rows] = cycles[i].mean;
        d[i + 3 * rows] = cycles[i].idx0;
        d[i + 4 * rows] = cycles[i].idx1;
    }
    return out;
}

// ── tsa ────────────────────────────────────────────────────────────

std::tuple<Value, Value>
tsa(const Value &x, double fs,
    const Value &rpm, double fs_rpm, int n_per_rev,
    keel::Allocator *mr)
{
    auto signal_v = readVec(x);
    auto rpm_v    = readVec(rpm);
    if (signal_v.empty() || rpm_v.empty() || n_per_rev <= 0) {
        return std::make_tuple(
            Value::matrix(0, 1, ValueType::DOUBLE, mr),
            Value::matrix(0, 1, ValueType::DOUBLE, mr));
    }

    // 1. Compute the cumulative angle θ(t_signal) at each signal sample
    //    by integrating ω(t) = 2π·rpm/60 against time.
    //    rpm is sampled at fs_rpm, signal at fs; we interpolate rpm
    //    onto the signal-sample grid first.
    const keel::usize N = signal_v.size();
    keel::Array<double> theta(N, 0.0, mr);
    {
        double acc = 0.0;
        for (keel::usize i = 0; i < N; ++i) {
            const double t = keel::as<double>(i) / fs;
            // Linear-interpolate rpm at time t.
            const double idxR = t * fs_rpm;
            const keel::usize i0 = keel::as<keel::usize>(std::floor(idxR));
            const keel::usize i1 = std::min(i0 + 1, rpm_v.size() - 1);
            const double frac = std::min(std::max(idxR - i0, 0.0), 1.0);
            const double rpmI =
                (i0 < rpm_v.size())
                ? rpm_v[i0] + frac * (rpm_v[i1] - rpm_v[i0])
                : rpm_v.back();
            const double omega = 2.0 * M_PI * rpmI / 60.0;
            theta[i] = acc;
            acc += omega / fs;
        }
    }

    // 2. Re-sample onto a uniform angle grid by accumulating into
    //    n_per_rev bins per revolution.
    const double twoPi = 2.0 * M_PI;
    keel::Array<double> sumBin(n_per_rev, 0.0, mr);
    keel::Array<int> cntBin(n_per_rev, 0, mr);
    for (keel::usize i = 0; i < N; ++i) {
        double ang = std::fmod(theta[i], twoPi);
        if (ang < 0) ang += twoPi;
        int b = keel::as<int>(ang / twoPi * n_per_rev);
        if (b >= n_per_rev) b = n_per_rev - 1;
        sumBin[b] += signal_v[i];
        cntBin[b]++;
    }
    keel::Array<double> avg(n_per_rev, mr);
    for (int b = 0; b < n_per_rev; ++b)
        avg[b] = (cntBin[b] > 0) ? sumBin[b] / cntBin[b] : 0.0;

    keel::Array<double> th(n_per_rev, mr);
    for (int b = 0; b < n_per_rev; ++b)
        th[b] = twoPi * b / n_per_rev;

    return std::make_tuple(vecCol(mr, avg), vecCol(mr, th));
}

} // namespace numkit::signal
