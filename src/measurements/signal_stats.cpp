// toolboxes/signal/src/measurements/signal_stats.cpp
//
// rms / rssq / peak2peak / peak2rms — signal-side reductions over a
// chosen dim. Built on the same applyAlongDim infrastructure that
// builtin stats (mean/var/std/...) use.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/signal/measurements/signal_stats.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/reductions.hpp>  // numkit::ops::applyAlongDim


namespace numkit::signal {

namespace {

using ::numkit::ops::applyAlongDim;
using ::numkit::ops::firstNonSingletonDim;
using ::numkit::ops::resolveDim;   // 0 -> first non-singleton, else validate
using ::numkit::ops::validateDim;

keel::i32 signal_stats_argcheck(const Value &x, keel::i32 dim, const char *fn)
{
    return resolveDim(x, dim, fn);
}

} // namespace

// ── rms ────────────────────────────────────────────────────────────────
Value rms(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = signal_stats_argcheck(x, dim, "rms");
    return applyAlongDim(x, d,
        [](keel::usize /*outIdx*/, const keel::f64 *slice, keel::usize n) -> keel::f64 {
            if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < n; ++i)
                s += slice[i] * slice[i];
            return keel::sqrt(s / keel::as<keel::f64>(n));
        }, mr);
}

// ── rssq ───────────────────────────────────────────────────────────────
Value rssq(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = signal_stats_argcheck(x, dim, "rssq");
    return applyAlongDim(x, d,
        [](keel::usize /*outIdx*/, const keel::f64 *slice, keel::usize n) -> keel::f64 {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < n; ++i)
                s += slice[i] * slice[i];
            return keel::sqrt(s);
        }, mr);
}

// ── peak2peak ──────────────────────────────────────────────────────────
Value peak2peak(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = signal_stats_argcheck(x, dim, "peak2peak");
    return applyAlongDim(x, d,
        [](keel::usize /*outIdx*/, const keel::f64 *slice, keel::usize n) -> keel::f64 {
            if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
            keel::f64 mn = slice[0], mx = slice[0];
            for (keel::usize i = 1; i < n; ++i) {
                const keel::f64 v = slice[i];
                if (keel::is_nan(v)) return keel::Limits<keel::f64>::quiet_nan();
                if (v < mn) mn = v;
                else if (v > mx) mx = v;
            }
            if (keel::is_nan(mn) || keel::is_nan(mx))
                return keel::Limits<keel::f64>::quiet_nan();
            return mx - mn;
        }, mr);
}

// ── peak2rms ───────────────────────────────────────────────────────────
Value peak2rms(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    const keel::i32 d = signal_stats_argcheck(x, dim, "peak2rms");
    return applyAlongDim(x, d,
        [](keel::usize /*outIdx*/, const keel::f64 *slice, keel::usize n) -> keel::f64 {
            if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
            keel::f64 s = 0.0, peak = 0.0;
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 a = keel::abs(slice[i]);
                s += a * a;
                if (a > peak) peak = a;
            }
            const keel::f64 r = keel::sqrt(s / keel::as<keel::f64>(n));
            if (r == 0.0) return keel::Limits<keel::f64>::infinity();
            return peak / r;
        }, mr);
}

} // namespace numkit::signal
