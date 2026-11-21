// toolboxes/signal/src/measurements/dbconv.cpp
//
// db / db2mag / mag2db / pow2db / db2pow.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/utf8.h>
#include <numkit/signal/measurements/dbconv.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>  // createLike (toolboxes/builtin/src/)

#include <algorithm>
#include <cmath>

namespace numkit::signal {

namespace {

// Walk every element of x, taking magnitude when complex; pass to f.
template <typename F>
void forEachAsMag(const Value &x, keel::f64 *dst, F &&f)
{
    const keel::usize n = x.numel();
    if (x.isComplex()) {
        const keel::c64 *src = x.complexData();
        for (keel::usize i = 0; i < n; ++i)
            dst[i] = f(keel::abs(src[i]));
    } else {
        const keel::f64 *src = x.doubleData();
        for (keel::usize i = 0; i < n; ++i)
            dst[i] = f(src[i]);
    }
}

keel::String toLower(keel::StringRef s)
{
    keel::String r = keel::String::from(s);
    for (auto &c : r) c = keel::to_ascii_lower(c);
    return r;
}

keel::f64 db_argcheck(keel::StringRef signalType)
{
    const keel::String mode = toLower(signalType);
    if (mode == "voltage" || mode.empty())
        return 20.0;
    if (mode == "power")
        return 10.0;
    throw Error("db: signalType must be 'voltage' or 'power'",
                 0, 0, "db", "", "numkit:db:badType");
}

} // namespace

// ── db ─────────────────────────────────────────────────────────────────
Value db(const Value &x, keel::StringRef signalType, keel::Allocator *mr)
{
    const keel::f64 scale = db_argcheck(signalType);
    auto out = createLike(x, ValueType::DOUBLE, mr);
    forEachAsMag(x, out.doubleDataMut(),
                 [scale](keel::f64 v) { return scale * std::log10(v); });
    return out;
}

// ── db2mag ─────────────────────────────────────────────────────────────
Value db2mag(const Value &d, keel::Allocator *mr)
{
    auto out = createLike(d, ValueType::DOUBLE, mr);
    forEachAsMag(d, out.doubleDataMut(),
                 [](keel::f64 v) { return std::pow(10.0, v / 20.0); });
    return out;
}

// ── mag2db ─────────────────────────────────────────────────────────────
Value mag2db(const Value &x, keel::Allocator *mr)
{
    auto out = createLike(x, ValueType::DOUBLE, mr);
    forEachAsMag(x, out.doubleDataMut(),
                 [](keel::f64 v) { return 20.0 * std::log10(v); });
    return out;
}

// ── db2pow ─────────────────────────────────────────────────────────────
Value db2pow(const Value &d, keel::Allocator *mr)
{
    auto out = createLike(d, ValueType::DOUBLE, mr);
    forEachAsMag(d, out.doubleDataMut(),
                 [](keel::f64 v) { return std::pow(10.0, v / 10.0); });
    return out;
}

// ── pow2db ─────────────────────────────────────────────────────────────
Value pow2db(const Value &p, keel::Allocator *mr)
{
    auto out = createLike(p, ValueType::DOUBLE, mr);
    forEachAsMag(p, out.doubleDataMut(),
                 [](keel::f64 v) { return 10.0 * std::log10(v); });
    return out;
}

} // namespace numkit::signal
