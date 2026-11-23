// toolboxes/signal/src/transforms/hilbert.cpp
//
// Analytic signal (hilbert) and envelopes delegating FFT/FIR/RMS kernels to keel::signal.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/signal/transforms/hilbert.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/value/value.hpp>
#include "../dsp_helpers.hpp"

#include <algorithm>

namespace numkit::signal {

namespace {

keel::Array<keel::f64> read_real(const Value &x, keel::Allocator *mr) {
    const keel::usize N = x.numel();
    keel::Array<keel::f64> v(N, 0.0, mr);
    for (keel::usize i = 0; i < N; ++i) v[i] = x.elemAsDouble(i);
    return v;
}

keel::Array<keel::usize> findpeaks_mindist(const keel::Array<keel::f64> &x, keel::usize n,
                                           keel::Allocator *mr) {
    const keel::usize N = x.size();
    keel::Array<keel::usize> peaks(mr);
    if (N < 3) return peaks;
    keel::Array<keel::usize> cand(mr);
    for (keel::usize i = 1; i + 1 < N; ++i)
        if (x[i] > x[i - 1] && x[i] > x[i + 1]) cand.push_back(i);
    if (cand.empty()) return peaks;
    if (n <= 1) {
        peaks.assign(cand);
        return peaks;
    }
    keel::Array<keel::usize> order = cand.clone(mr);
    std::sort(order.begin(), order.end(),
              [&](keel::usize a, keel::usize b) { return x[a] > x[b]; });
    keel::Array<keel::u8> accepted(N, 0, mr);
    for (keel::usize idx : order) {
        bool ok = true;
        const keel::i64 lo = keel::as<keel::i64>(idx) - keel::as<keel::i64>(n);
        const keel::i64 hi = keel::as<keel::i64>(idx) + keel::as<keel::i64>(n);
        for (keel::i64 j = std::max(keel::i64{0}, lo);
             j <= std::min(keel::as<keel::i64>(N) - 1, hi); ++j)
            if (accepted[keel::as<keel::usize>(j)]) { ok = false; break; }
        if (ok) accepted[idx] = 1;
    }
    for (keel::usize i = 0; i < N; ++i) if (accepted[i]) peaks.push_back(i);
    return peaks;
}

void env_peak(const Value &x, keel::usize n, keel::Array<keel::f64> &up,
              keel::Array<keel::f64> &lo, keel::Allocator *mr) {
    const keel::usize N = x.numel();
    auto v = read_real(x, mr);
    keel::Array<keel::f64> neg(N, 0.0, mr);
    for (keel::usize i = 0; i < N; ++i) neg[i] = -v[i];

    auto build_envelope = [&](const keel::Array<keel::f64> &sig, keel::Array<keel::f64> &out) {
        keel::Array<keel::usize> peaks(mr);
        if (N > n + 1) peaks = findpeaks_mindist(sig, n, mr);
        keel::Array<keel::usize> locs(mr);
        if (peaks.size() < 2) {
            keel::Array<keel::usize> draft(mr);
            draft.push_back(0);
            for (keel::usize p : peaks) draft.push_back(p);
            draft.push_back(N - 1);
            for (keel::usize l : draft) {
                if (locs.empty() || locs.back() != l) locs.push_back(l);
            }
        } else {
            locs.assign(peaks);
        }
        out.resize(N);
        if (locs.size() == 3) {
            const keel::f64 x1 = keel::as<keel::f64>(locs[0] + 1);
            const keel::f64 x2 = keel::as<keel::f64>(locs[1] + 1);
            const keel::f64 x3 = keel::as<keel::f64>(locs[2] + 1);
            const keel::f64 y1 = sig[locs[0]];
            const keel::f64 y2 = sig[locs[1]];
            const keel::f64 y3 = sig[locs[2]];
            const keel::f64 det = (x1 - x2) * (x1 - x3) * (x2 - x3);
            const keel::f64 a = (y1 * (x2 - x3) - y2 * (x1 - x3) + y3 * (x1 - x2)) / det;
            const keel::f64 b = (y1 * (x3 * x3 - x2 * x2) - y2 * (x3 * x3 - x1 * x1) +
                                 y3 * (x2 * x2 - x1 * x1)) / det;
            const keel::f64 c = (y1 * (x2 * x2 * x3 - x2 * x3 * x3) -
                                 y2 * (x1 * x1 * x3 - x1 * x3 * x3) +
                                 y3 * (x1 * x1 * x2 - x1 * x2 * x2)) / det;
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 xi = keel::as<keel::f64>(i + 1);
                out[i] = a * xi * xi + b * xi + c;
            }
            return;
        }
        Value x_loc = Value::matrix(locs.size(), 1, ValueType::DOUBLE, mr);
        Value y_loc = Value::matrix(locs.size(), 1, ValueType::DOUBLE, mr);
        Value xq    = Value::matrix(N, 1, ValueType::DOUBLE, mr);
        keel::f64 *xd = x_loc.doubleDataMut();
        keel::f64 *yd = y_loc.doubleDataMut();
        keel::f64 *qd = xq.doubleDataMut();
        for (keel::usize i = 0; i < locs.size(); ++i) {
            xd[i] = keel::as<keel::f64>(locs[i] + 1);
            yd[i] = sig[locs[i]];
        }
        for (keel::usize i = 0; i < N; ++i) qd[i] = keel::as<keel::f64>(i + 1);
        Value yi = numkit::builtin::interp1(x_loc, y_loc, xq, "spline", mr);
        for (keel::usize i = 0; i < N; ++i) out[i] = yi.elemAsDouble(i);
    };
    build_envelope(v, up);
    build_envelope(neg, lo);
    for (auto &v_ : lo) v_ = -v_;
}

} // anonymous

Value hilbert(const Value &x, keel::Allocator *mr) {
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto xr = read_real(x, scratch.allocator());
    keel::Array<keel::c64> buf(N, keel::c64(0.0, 0.0), scratch.allocator());
    (void)keel::signal::hilbert(xr.as_ref(), buf.as_mut_ref(), scratch.allocator());
    return packComplexResult(buf.data(), N, mr);
}

Value envelope(const Value &x, keel::Allocator *mr) {
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto v = read_real(x, scratch.allocator());
    auto r = createLike(x, ValueType::DOUBLE, mr);
    if (N == 0) return r;
    (void)keel::signal::envelope_analytic(v.as_ref(),
                                          keel::MutArrayRef<keel::f64>(r.doubleDataMut(), N),
                                          {}, scratch.allocator());
    return r;
}

keel::Pair<Value, Value>
envelope_full(const Value &x, keel::i32 mode, keel::usize n, keel::Allocator *mr) {
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    Value yupper = createLike(x, ValueType::DOUBLE, mr);
    Value ylower = createLike(x, ValueType::DOUBLE, mr);
    if (N == 0) return {keel::move(yupper), keel::move(ylower)};

    if (mode == 3) {
        keel::Array<keel::f64> up(scratch.allocator()), lo(scratch.allocator());
        env_peak(x, n, up, lo, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) yupper.doubleDataMut()[i] = up[i];
        for (keel::usize i = 0; i < N; ++i) ylower.doubleDataMut()[i] = lo[i];
        return {keel::move(yupper), keel::move(ylower)};
    }

    const auto v = read_real(x, scratch.allocator());
    keel::MutArrayRef<keel::f64> uRef(yupper.doubleDataMut(), N);
    keel::MutArrayRef<keel::f64> lRef(ylower.doubleDataMut(), N);
    if (mode == 0) {
        (void)keel::signal::envelope_analytic(v.as_ref(), uRef, lRef, scratch.allocator());
    } else if (mode == 1) {
        (void)keel::signal::envelope_fir(v.as_ref(), n, uRef, lRef, scratch.allocator());
    } else {
        (void)keel::signal::envelope_rms(v.as_ref(), n, uRef, lRef);
    }
    return {keel::move(yupper), keel::move(ylower)};
}

keel::Pair<Value, Value> envelope_pair(const Value &x, keel::Allocator *mr) {
    return envelope_full(x, 0, 0, mr);
}

} // namespace numkit::signal
