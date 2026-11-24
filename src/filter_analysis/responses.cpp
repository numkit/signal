// toolboxes/signal/src/filter_analysis/responses.cpp
//
// impz / impzlength / stepz / phasedelay / zerophase (delegates to keel::signal).

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/filter_analysis/responses.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

keel::Array<keel::f64> readReal(const Value &v, keel::Allocator *mr)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, mr);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

void responses_argcheck(const Value &b, const Value &a)
{
    (void)b;
    (void)a;
}

} // namespace

keel::usize impzlength(const Value &b, const Value &a, keel::Allocator *mr)
{
    responses_argcheck(b, a);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readReal(b, scratch.allocator());
    auto av = readReal(a, scratch.allocator());
    keel::usize outLen = 1;
    (void)keel::signal::impzlength(bv.as_ref(), av.as_ref(), outLen, scratch.allocator());
    return outLen;
}

keel::Tuple<Value, Value>
impz(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    if (n == 0) n = impzlength(b, a, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readReal(b, scratch.allocator());
    auto av = readReal(a, scratch.allocator());
    auto h = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    auto t = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::impz(bv.as_ref(), av.as_ref(), n,
                             keel::MutArrayRef<keel::f64>(h.doubleDataMut(), n),
                             keel::MutArrayRef<keel::f64>(t.doubleDataMut(), n),
                             scratch.allocator());
    return keel::make_tuple(keel::move(h), keel::move(t));
}

keel::Tuple<Value, Value>
stepz(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    if (n == 0) n = impzlength(b, a, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readReal(b, scratch.allocator());
    auto av = readReal(a, scratch.allocator());
    auto s = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    auto t = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::stepz(bv.as_ref(), av.as_ref(), n,
                              keel::MutArrayRef<keel::f64>(s.doubleDataMut(), n),
                              keel::MutArrayRef<keel::f64>(t.doubleDataMut(), n),
                              scratch.allocator());
    return keel::make_tuple(keel::move(s), keel::move(t));
}

keel::Tuple<Value, Value>
phasedelay(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readReal(b, scratch.allocator());
    auto av = readReal(a, scratch.allocator());
    auto pd = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    auto w  = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::phasedelay(bv.as_ref(), av.as_ref(), n,
                                   keel::MutArrayRef<keel::f64>(pd.doubleDataMut(), n),
                                   keel::MutArrayRef<keel::f64>(w.doubleDataMut(), n));
    return keel::make_tuple(keel::move(pd), keel::move(w));
}

keel::Tuple<Value, Value>
zerophase(const Value &b, const Value &a, keel::usize n, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readReal(b, scratch.allocator());
    auto av = readReal(a, scratch.allocator());
    auto hr = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    auto w  = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::zerophase(bv.as_ref(), av.as_ref(), n,
                                  keel::MutArrayRef<keel::f64>(hr.doubleDataMut(), n),
                                  keel::MutArrayRef<keel::f64>(w.doubleDataMut(), n));
    return keel::make_tuple(keel::move(hr), keel::move(w));
}

} // namespace numkit::signal
