// toolboxes/signal/src/measurements/vibration.cpp
//
// envspectrum / tachorpm / rainflow / tsa (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/measurements/vibration.hpp>

#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

keel::Array<keel::f64> readVec(const Value &x, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    keel::Array<keel::f64> v(n, mr);
    for (keel::usize i = 0; i < n; ++i) v[i] = x.elemAsDouble(i);
    return v;
}

Value vecCol(keel::Allocator *mr, keel::ArrayRef<keel::f64> v)
{
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

} // anonymous

keel::Tuple<Value, Value>
envspectrum(const Value &x, keel::f64 fs, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    keel::Array<keel::f64> es(scratch.allocator()), f(scratch.allocator());
    (void)keel::signal::envspectrum(xv.as_ref(), fs, es, f, scratch.allocator());
    return keel::make_tuple(vecCol(mr, es.as_ref()), vecCol(mr, f.as_ref()));
}

keel::Tuple<Value, Value>
tachorpm(const Value &x, keel::f64 fs, const Value &threshold, keel::i32 ppr, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    const bool hasThr = !threshold.isEmpty();
    const keel::f64 thr = hasThr ? threshold.toScalar() : 0.0;
    keel::Array<keel::f64> rpm(scratch.allocator()), times(scratch.allocator());
    (void)keel::signal::tachorpm(xv.as_ref(), fs, hasThr, thr, ppr, rpm, times, scratch.allocator());
    return keel::make_tuple(vecCol(mr, rpm.as_ref()), vecCol(mr, times.as_ref()));
}

Value rainflow(const Value &x, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    keel::Array<keel::signal::RainflowCycle> cycles(scratch.allocator());
    (void)keel::signal::rainflow(xv.as_ref(), cycles, scratch.allocator());

    const keel::usize rows = cycles.size();
    auto out = Value::matrix(rows, 5, ValueType::DOUBLE, mr);
    if (rows == 0) return out;
    keel::f64 *d = out.doubleDataMut();
    for (keel::usize i = 0; i < rows; ++i) {
        d[i + 0 * rows] = cycles[i].count;
        d[i + 1 * rows] = cycles[i].range;
        d[i + 2 * rows] = cycles[i].mean;
        d[i + 3 * rows] = cycles[i].start_idx;
        d[i + 4 * rows] = cycles[i].end_idx;
    }
    return out;
}

keel::Tuple<Value, Value>
tsa(const Value &x, keel::f64 fs, const Value &rpm, keel::f64 fs_rpm, keel::i32 n_per_rev,
    keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xv = readVec(x, scratch.allocator());
    auto rv = readVec(rpm, scratch.allocator());
    keel::Array<keel::f64> avg(scratch.allocator()), th(scratch.allocator());
    (void)keel::signal::tsa(xv.as_ref(), fs, rv.as_ref(), fs_rpm, n_per_rev, avg, th,
                            scratch.allocator());
    return keel::make_tuple(vecCol(mr, avg.as_ref()), vecCol(mr, th.as_ref()));
}

} // namespace numkit::signal
