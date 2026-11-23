// toolboxes/signal/src/measurements/pulse_metrics.cpp
//
// Pulse and transition metrics. Reference levels via histogram-mode
// statelevels; transitions detected via crossings of the lower / upper
// state boundaries (10% / 90% by default).

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/math.h>
#include <numkit/signal/measurements/pulse_metrics.hpp>
#include "pulse_metrics_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::signal {

namespace {

constexpr keel::f64 kLowPct  = 0.10;   // lower state boundary fraction
constexpr keel::f64 kHighPct = 0.90;   // upper state boundary fraction
constexpr keel::f64 kMidPct  = 0.50;   // mid-reference level
constexpr keel::i32    kHistBins = 100;

keel::f64 scalarOrDefault(const Value &v, keel::f64 dflt)
{
    return v.isEmpty() ? dflt : v.toScalar();
}

// Read x as a contiguous double array. If x is not DOUBLE, copy via
// elemAsDouble. Returns the buffer (heap) and length.
keel::Array<keel::f64> readVec(const Value &x)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> v(n);
    if (n == 0) return v;
    if (x.type() == ValueType::DOUBLE) {
        keel::copy_n(v.data(), x.doubleData(), n);
    } else {
        for (keel::usize i = 0; i < n; ++i) v[i] = x.elemAsDouble(i);
    }
    return v;
}

Value colVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    if (v.empty()) return Value::matrix(0, 1, ValueType::DOUBLE, mr);
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

// Histogram-mode state level estimate. Returns [low, high]. For flat
// inputs (range 0) returns [val, val].
struct LowHigh { keel::f64 low; keel::f64 high; };

LowHigh stateLevelsCalc(const keel::f64 *x, keel::usize n)
{
    if (n == 0) return {0.0, 0.0};
    keel::f64 xmin = x[0], xmax = x[0];
    for (keel::usize i = 1; i < n; ++i) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
    }
    if (!(xmax > xmin)) return {xmin, xmax};
    const keel::f64 width = (xmax - xmin) / kHistBins;
    keel::Array<keel::i32> hist(kHistBins, 0);
    for (keel::usize i = 0; i < n; ++i) {
        keel::i32 b = keel::as<keel::i32>((x[i] - xmin) / width);
        if (b >= kHistBins) b = kHistBins - 1;
        if (b < 0) b = 0;
        hist[b]++;
    }
    const keel::f64 mid = (xmin + xmax) * 0.5;
    const keel::i32 splitBin = keel::as<keel::i32>((mid - xmin) / width);
    keel::i32 lowBin = 0, highBin = kHistBins - 1;
    keel::i32 lowMax = -1, highMax = -1;
    for (keel::i32 b = 0; b < splitBin; ++b)
        if (hist[b] > lowMax) { lowMax = hist[b]; lowBin = b; }
    for (keel::i32 b = splitBin; b < kHistBins; ++b)
        if (hist[b] > highMax) { highMax = hist[b]; highBin = b; }
    return {
        xmin + (lowBin  + 0.5) * width,
        xmin + (highBin + 0.5) * width,
    };
}

// Linearly interpolate the sample index where x crosses `level` between
// indices i and i+1 (assumes x[i] and x[i+1] straddle level). Returns
// fractional 0-based index.
keel::f64 crossingIdx(const keel::f64 *x, keel::usize i, keel::f64 level)
{
    const keel::f64 a = x[i], b = x[i + 1];
    if (a == b) return keel::as<keel::f64>(i);
    return i + (level - a) / (b - a);
}

// Convert a fractional sample index to time given a sample rate. With
// fs nullptr/empty, returns the 1-based fractional sample number
// (matches MATLAB's "no fs" idiom: index returned as t).
keel::f64 idxToTime(keel::f64 idx, keel::f64 fs)
{
    if (fs > 0.0) return idx / fs;
    return idx + 1.0;  // 1-based sample index, fractional
}

// Find every monotonic transition between the lower and upper state
// boundaries. A transition has a `sign` (+1 = rising, -1 = falling),
// `start_idx` (frac sample where x crossed the trailing boundary),
// `end_idx` (frac sample where x crossed the leading boundary). The
// monotonic-trip rule mirrors MATLAB: between the trailing-boundary
// cross and the leading-boundary cross, x must not re-cross the
// trailing boundary.
struct Transition {
    keel::i32 sign;        // +1 rising, -1 falling
    keel::f64 startIdx; // crossing of trailing boundary (10% for rising)
    keel::f64 endIdx;   // crossing of leading boundary (90% for rising)
};

keel::Array<Transition>
findTransitions(const keel::f64 *x, keel::usize n, keel::f64 low, keel::f64 high)
{
    keel::Array<Transition> out;
    if (n < 2 || !(high > low)) return out;
    const keel::f64 range = high - low;
    const keel::f64 trail = low + kLowPct * range;
    const keel::f64 lead  = low + kHighPct * range;

    enum class State { Below, Mid, Above };
    auto classify = [&](keel::f64 v) -> State {
        if (v <= trail) return State::Below;
        if (v >= lead)  return State::Above;
        return State::Mid;
    };

    State curState = classify(x[0]);
    keel::f64 pendingStart = 0.0;
    bool inRise = false, inFall = false;
    for (keel::usize i = 1; i < n; ++i) {
        State s = classify(x[i]);
        // Rising-transition tracking: arm when crossing from Below to
        // Mid (record the trail crossing); commit on Mid→Above (lead
        // crossing); cancel on any return to Below.
        if (s == State::Below) {
            inRise = false;
            // Falling: commit if we were tracking a fall
            if (inFall) {
                out.push_back({-1, pendingStart, crossingIdx(x, i - 1, trail)});
                inFall = false;
            }
        } else if (s == State::Above) {
            inFall = false;
            if (inRise) {
                out.push_back({+1, pendingStart, crossingIdx(x, i - 1, lead)});
                inRise = false;
            }
        }

        // Detect entry into a transition from Below (start of rise) or from
        // Above (start of fall).
        if (curState == State::Below && s != State::Below && !inRise) {
            // Crossed trailing boundary going up.
            inRise = true;
            pendingStart = crossingIdx(x, i - 1, trail);
            // Sharp edge: a single-sample jump Below->Above crosses BOTH the
            // trailing and leading boundaries within this same interval
            // [i-1, i]. Commit now — the flat region after has no leading
            // crossing, so the delayed-commit path would otherwise pin the
            // leading crossing to the wrong (flat) interval.
            if (s == State::Above) {
                out.push_back({+1, pendingStart, crossingIdx(x, i - 1, lead)});
                inRise = false;
            }
        }
        if (curState == State::Above && s != State::Above && !inFall) {
            // Crossed leading boundary going down.
            inFall = true;
            pendingStart = crossingIdx(x, i - 1, lead);
            if (s == State::Below) {  // sharp falling edge — same interval
                out.push_back({-1, pendingStart, crossingIdx(x, i - 1, trail)});
                inFall = false;
            }
        }
        curState = s;
    }
    return out;
}

// Midcross helper: every fractional sample where x crosses `level`.
// Each crossing is counted once (no duplicate if x stays on level).
keel::Array<keel::f64>
midcrossesAt(const keel::f64 *x, keel::usize n, keel::f64 level)
{
    keel::Array<keel::f64> out;
    for (keel::usize i = 0; i + 1 < n; ++i) {
        const keel::f64 a = x[i], b = x[i + 1];
        if ((a < level && b >= level) || (a > level && b <= level)
            || (a == level && b != level && i == 0)) {
            out.push_back(crossingIdx(x, i, level));
        }
    }
    return out;
}

} // anonymous

// ── statelevels ────────────────────────────────────────────────────

Value statelevels(const Value &x, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = lh.low;
    out.doubleDataMut()[1] = lh.high;
    return out;
}

// ── midcross ───────────────────────────────────────────────────────

Value midcross(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    const keel::f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    const auto idx = midcrossesAt(v.data(), v.size(), mid);
    const keel::f64 f = scalarOrDefault(fs, 0.0);
    keel::Array<keel::f64> times(idx.size(), mr);
    for (keel::usize i = 0; i < idx.size(); ++i) times[i] = idxToTime(idx[i], f);
    return colVec(times, mr);
}

// ── risetime / falltime / slewrate ─────────────────────────────────

Value risetime(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto trs = findTransitions(v.data(), v.size(), lh.low, lh.high);
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    keel::Array<keel::f64> out(mr);
    for (const auto &t : trs)
        if (t.sign > 0) out.push_back((t.endIdx - t.startIdx) / keel::max(f, 1.0));
    return colVec(out, mr);
}

Value falltime(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto trs = findTransitions(v.data(), v.size(), lh.low, lh.high);
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    keel::Array<keel::f64> out(mr);
    for (const auto &t : trs)
        if (t.sign < 0) out.push_back((t.endIdx - t.startIdx) / keel::max(f, 1.0));
    return colVec(out, mr);
}

Value slewrate(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto trs = findTransitions(v.data(), v.size(), lh.low, lh.high);
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    const keel::f64 range = (lh.high - lh.low) * (kHighPct - kLowPct);
    keel::Array<keel::f64> out(mr);
    for (const auto &t : trs) {
        const keel::f64 dur = (t.endIdx - t.startIdx) / keel::max(f, 1.0);
        if (dur > 0.0) out.push_back(t.sign * range / dur);
        else           out.push_back(0.0);
    }
    return colVec(out, mr);
}

// ── overshoot / undershoot ─────────────────────────────────────────
// MATLAB: percent OS = 100 * (peak_after_transition - high_state) /
// (high_state - low_state). The peak window is from the end of the
// transition until the next opposite transition (or end of signal).

Value overshoot(const Value &x, const Value &fs, keel::Allocator *mr)
{
    (void)fs; // overshoot uses transition list only, no time arg needed
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto trs = findTransitions(v.data(), v.size(), lh.low, lh.high);
    const keel::f64 range = lh.high - lh.low;
    keel::Array<keel::f64> out(mr);
    if (range <= 0.0) return colVec(out, mr);
    for (keel::usize k = 0; k < trs.size(); ++k) {
        const auto &t = trs[k];
        if (t.sign <= 0) continue;
        // Window: from end of rise to next transition (or end of signal).
        const keel::usize wbeg = keel::as<keel::usize>(keel::ceil(t.endIdx));
        keel::usize wend = v.size();
        for (keel::usize j = k + 1; j < trs.size(); ++j) {
            wend = keel::as<keel::usize>(keel::floor(trs[j].startIdx));
            break;
        }
        if (wbeg >= wend) continue;
        keel::f64 peak = v[wbeg];
        for (keel::usize i = wbeg + 1; i < wend; ++i)
            if (v[i] > peak) peak = v[i];
        out.push_back(100.0 * (peak - lh.high) / range);
    }
    return colVec(out, mr);
}

Value undershoot(const Value &x, const Value &fs, keel::Allocator *mr)
{
    (void)fs;
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto trs = findTransitions(v.data(), v.size(), lh.low, lh.high);
    const keel::f64 range = lh.high - lh.low;
    keel::Array<keel::f64> out(mr);
    if (range <= 0.0) return colVec(out, mr);
    for (keel::usize k = 0; k < trs.size(); ++k) {
        const auto &t = trs[k];
        if (t.sign >= 0) continue;
        const keel::usize wbeg = keel::as<keel::usize>(keel::ceil(t.endIdx));
        keel::usize wend = v.size();
        for (keel::usize j = k + 1; j < trs.size(); ++j) {
            wend = keel::as<keel::usize>(keel::floor(trs[j].startIdx));
            break;
        }
        if (wbeg >= wend) continue;
        keel::f64 trough = v[wbeg];
        for (keel::usize i = wbeg + 1; i < wend; ++i)
            if (v[i] < trough) trough = v[i];
        out.push_back(100.0 * (lh.low - trough) / range);
    }
    return colVec(out, mr);
}

// ── settlingtime ──────────────────────────────────────────────────
// Time from the start of each transition until x stays within `tol`
// of the destination state for the rest of the post-transition window.

Value settlingtime(const Value &x, const Value &fs, keel::f64 tol, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto trs = findTransitions(v.data(), v.size(), lh.low, lh.high);
    const keel::f64 range = lh.high - lh.low;
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    keel::Array<keel::f64> out(mr);
    if (range <= 0.0) return colVec(out, mr);
    for (keel::usize k = 0; k < trs.size(); ++k) {
        const auto &t = trs[k];
        const keel::f64 dest = (t.sign > 0) ? lh.high : lh.low;
        const keel::f64 band = tol * range;
        keel::usize wend = v.size();
        for (keel::usize j = k + 1; j < trs.size(); ++j) {
            wend = keel::as<keel::usize>(keel::floor(trs[j].startIdx));
            break;
        }
        // Walk backwards from wend-1 to find the LAST sample outside band.
        keel::usize lastOut = keel::as<keel::usize>(keel::floor(t.startIdx));
        for (keel::usize i = keel::as<keel::usize>(keel::floor(t.startIdx));
             i + 1 < wend && i < v.size(); ++i) {
            if (keel::abs(v[i] - dest) > band) lastOut = i;
        }
        const keel::f64 settled = keel::as<keel::f64>(lastOut) - t.startIdx;
        out.push_back(keel::max(0.0, settled) / keel::max(f, 1.0));
    }
    return colVec(out, mr);
}

// ── pulsewidth / pulseperiod / pulsesep / dutycycle ────────────────
// All built from mid-state crossings.

namespace {

// Build a list of (idx, dir) pairs: dir = +1 for rising mid-cross, -1
// for falling mid-cross. Used by the pulse-shape metrics below.
keel::Array<keel::Pair<keel::f64, keel::i32>>
midCrossesDirected(const keel::f64 *x, keel::usize n, keel::f64 mid)
{
    keel::Array<keel::Pair<keel::f64, keel::i32>> out;
    for (keel::usize i = 0; i + 1 < n; ++i) {
        const keel::f64 a = x[i], b = x[i + 1];
        if (a < mid && b >= mid)      out.emplace_back(crossingIdx(x, i, mid), +1);
        else if (a > mid && b <= mid) out.emplace_back(crossingIdx(x, i, mid), -1);
    }
    return out;
}

} // anonymous

Value pulsewidth(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    const keel::f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    auto crs = midCrossesDirected(v.data(), v.size(), mid);
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    keel::Array<keel::f64> out(mr);
    // A pulse runs from a +1 crossing to the next -1 crossing.
    for (keel::usize i = 0; i + 1 < crs.size(); ++i) {
        if (crs[i].second == +1 && crs[i + 1].second == -1)
            out.push_back((crs[i + 1].first - crs[i].first) / keel::max(f, 1.0));
    }
    return colVec(out, mr);
}

Value pulseperiod(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    const keel::f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    auto crs = midCrossesDirected(v.data(), v.size(), mid);
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    keel::Array<keel::f64> out(mr);
    // Period: from one rising mid-cross to the next rising mid-cross.
    keel::Array<keel::f64> rising(mr);
    for (const auto &c : crs) if (c.second == +1) rising.push_back(c.first);
    for (keel::usize i = 0; i + 1 < rising.size(); ++i)
        out.push_back((rising[i + 1] - rising[i]) / keel::max(f, 1.0));
    return colVec(out, mr);
}

Value pulsesep(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    const keel::f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    auto crs = midCrossesDirected(v.data(), v.size(), mid);
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    keel::Array<keel::f64> out(mr);
    // Separation: trailing edge of one pulse to leading edge of the next.
    for (keel::usize i = 0; i + 1 < crs.size(); ++i) {
        if (crs[i].second == -1 && crs[i + 1].second == +1)
            out.push_back((crs[i + 1].first - crs[i].first) / keel::max(f, 1.0));
    }
    return colVec(out, mr);
}

Value dutycycle(const Value &x, const Value &fs, keel::Allocator *mr)
{
    auto v = readVec(x);
    auto lh = stateLevelsCalc(v.data(), v.size());
    const keel::f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    auto crs = midCrossesDirected(v.data(), v.size(), mid);
    const keel::f64 f = scalarOrDefault(fs, 1.0);
    (void)f;  // ratio is unitless; fs cancels.
    keel::Array<keel::f64> out(mr);
    // Duty cycle: pulsewidth / pulseperiod for each pulse.
    // Walk +1, -1, +1 triples.
    for (keel::usize i = 0; i + 2 < crs.size(); ++i) {
        if (crs[i].second == +1 && crs[i + 1].second == -1
            && crs[i + 2].second == +1) {
            const keel::f64 width  = crs[i + 1].first - crs[i].first;
            const keel::f64 period = crs[i + 2].first - crs[i].first;
            if (period > 0) out.push_back(width / period);
        }
    }
    return colVec(out, mr);
}

namespace {

void pulseRiseFall_argcheck(keel::ArrayRef<Value> args, const char *fnname)
{
    if (args.empty())
        throw Error(keel::format("{}: requires at least 1 argument", fnname),
                     0, 0, fnname, "", keel::format("numkit:{}:nargin", fnname));
}

} // anonymous

// risetime / falltime worker (declared in pulse_metrics_detail.hpp). Lives on
// the compute side — it only needs the file-local state-level helpers above —
// so the risetime_reg/falltime_reg adapters forward into it. Emits up to 5
// outputs: [R, LT, UT, LL, UL] — duration, lower(10%) crossing time, upper(90%)
// crossing time, and the lower/upper reference LEVELS. For a fall the lower
// crossing happens last, the upper first (UT < LT).
void pulseRiseFall(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                   bool rising, const char *fnname,
                   keel::Allocator *mr)
{
    pulseRiseFall_argcheck(args, fnname);
    const Value &fsv = (args.size() >= 2) ? args[1] : Value::Empty;
    auto v = readVec(args[0]);
    auto lh = stateLevelsCalc(v.data(), v.size());
    auto trs = findTransitions(v.data(), v.size(), lh.low, lh.high);
    const keel::f64 f = scalarOrDefault(fsv, 1.0);
    const keel::f64 range = lh.high - lh.low;
    const keel::f64 LL = lh.low + kLowPct  * range;   // lower (10%) reference level
    const keel::f64 UL = lh.low + kHighPct * range;   // upper (90%) reference level

    keel::Array<keel::f64> dur(mr), ltv(mr), utv(mr);
    for (const auto &t : trs) {
        if (rising && t.sign <= 0) continue;
        if (!rising && t.sign >= 0) continue;
        // Transition struct stores startIdx = first crossing in time,
        // endIdx = second. For a rise that is (lower, upper); for a fall it
        // is (upper, lower). Map to MATLAB's lower/upper crossing times.
        const keel::f64 lowerCrossIdx = rising ? t.startIdx : t.endIdx;
        const keel::f64 upperCrossIdx = rising ? t.endIdx   : t.startIdx;
        dur.push_back((t.endIdx - t.startIdx) / keel::max(f, 1.0));
        ltv.push_back(idxToTime(lowerCrossIdx, f));
        utv.push_back(idxToTime(upperCrossIdx, f));
    }
    outs[0] = colVec(dur, mr);
    if (nargout > 1) outs[1] = colVec(ltv, mr);
    if (nargout > 2) outs[2] = colVec(utv, mr);
    if (nargout > 3) outs[3] = Value::scalar(LL, mr);
    if (nargout > 4) outs[4] = Value::scalar(UL, mr);
}

} // namespace numkit::signal
