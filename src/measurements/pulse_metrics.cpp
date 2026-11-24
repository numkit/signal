// toolboxes/signal/src/measurements/pulse_metrics.cpp
//
// Pulse and transition metrics (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/measurements/pulse_metrics.hpp>
#include "pulse_metrics_detail.hpp"

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

keel::f64 scalarOrDefault(const Value &v, keel::f64 dflt)
{
    return v.isEmpty() ? dflt : v.toScalar();
}

keel::Array<keel::f64> readVec(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> v(n, mr);
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

void pulseRiseFall_argcheck(keel::ArrayRef<Value> args, const char *fnname)
{
    if (args.empty())
        throw Error(keel::format("{}: requires at least 1 argument", fnname),
                     0, 0, fnname, "", keel::format("numkit:{}:nargin", fnname));
}

} // anonymous

Value statelevels(const Value &x, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::signal::StateLevels lh;
    (void)keel::signal::statelevels(v.as_ref(), lh);
    auto out = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = lh.low;
    out.doubleDataMut()[1] = lh.high;
    return out;
}

Value midcross(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> times(scratch.allocator());
    (void)keel::signal::midcross(v.as_ref(), scalarOrDefault(fs, 0.0), times, scratch.allocator());
    return colVec(times.as_ref(), mr);
}

Value risetime(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::signal::RiseFallResult res;
    (void)keel::signal::risetime_full(v.as_ref(), scalarOrDefault(fs, 1.0), true, res,
                                      scratch.allocator());
    return colVec(res.dur.as_ref(), mr);
}

Value falltime(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::signal::RiseFallResult res;
    (void)keel::signal::risetime_full(v.as_ref(), scalarOrDefault(fs, 1.0), false, res,
                                      scratch.allocator());
    return colVec(res.dur.as_ref(), mr);
}

Value slewrate(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::slewrate(v.as_ref(), scalarOrDefault(fs, 1.0), out, scratch.allocator());
    return colVec(out.as_ref(), mr);
}

Value overshoot(const Value &x, const Value &fs, keel::Allocator *mr)
{
    (void)fs;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::overshoot(v.as_ref(), out, scratch.allocator());
    return colVec(out.as_ref(), mr);
}

Value undershoot(const Value &x, const Value &fs, keel::Allocator *mr)
{
    (void)fs;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::undershoot(v.as_ref(), out, scratch.allocator());
    return colVec(out.as_ref(), mr);
}

Value settlingtime(const Value &x, const Value &fs, keel::f64 tol, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::settlingtime(v.as_ref(), scalarOrDefault(fs, 1.0), tol, out,
                                     scratch.allocator());
    return colVec(out.as_ref(), mr);
}

Value pulsewidth(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::pulsewidth(v.as_ref(), scalarOrDefault(fs, 1.0), out, scratch.allocator());
    return colVec(out.as_ref(), mr);
}

Value pulseperiod(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::pulseperiod(v.as_ref(), scalarOrDefault(fs, 1.0), out, scratch.allocator());
    return colVec(out.as_ref(), mr);
}

Value pulsesep(const Value &x, const Value &fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::pulsesep(v.as_ref(), scalarOrDefault(fs, 1.0), out, scratch.allocator());
    return colVec(out.as_ref(), mr);
}

Value dutycycle(const Value &x, const Value &fs, keel::Allocator *mr)
{
    (void)fs;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(x, scratch.allocator());
    keel::Array<keel::f64> out(scratch.allocator());
    (void)keel::signal::dutycycle(v.as_ref(), out, scratch.allocator());
    return colVec(out.as_ref(), mr);
}

void pulseRiseFall(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                   bool rising, const char *fnname, keel::Allocator *mr)
{
    pulseRiseFall_argcheck(args, fnname);
    const Value &fsv = (args.size() >= 2) ? args[1] : Value::Empty;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = readVec(args[0], scratch.allocator());
    keel::signal::RiseFallResult res;
    (void)keel::signal::risetime_full(v.as_ref(), scalarOrDefault(fsv, 1.0), rising, res,
                                      scratch.allocator());
    outs[0] = colVec(res.dur.as_ref(), mr);
    if (nargout > 1) outs[1] = colVec(res.lt.as_ref(), mr);
    if (nargout > 2) outs[2] = colVec(res.ut.as_ref(), mr);
    if (nargout > 3) outs[3] = Value::scalar(res.ll, mr);
    if (nargout > 4) outs[4] = Value::scalar(res.ul, mr);
}

} // namespace numkit::signal
