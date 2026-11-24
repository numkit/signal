// toolboxes/signal/src/filter_implementation/conversions.cpp
//
// zp2sos / tf2sos — convert filter representations to SOS form
// (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/filter_implementation/conversions.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

keel::Array<keel::c64> readComplexVec(const Value &v, const char *fn, keel::Allocator *mr)
{
    if (v.isEmpty()) return keel::Array<keel::c64>(mr);
    if (v.type() != ValueType::COMPLEX && v.type() != ValueType::DOUBLE && v.type() != ValueType::SINGLE)
        throw Error(keel::format("{}: zeros/poles must be double, single, or complex vectors", fn),
                    0, 0, fn, "", keel::format("numkit:{}:inputType", fn));
    keel::Array<keel::c64> out(mr);
    out.reserve(v.numel());
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *p = v.complexData();
        for (keel::usize i = 0; i < v.numel(); ++i) out.push_back(p[i]);
    } else {
        for (keel::usize i = 0; i < v.numel(); ++i) out.push_back(keel::c64(v.elemAsDouble(i), 0.0));
    }
    return out;
}

keel::Array<keel::c64> coeffsAsVector(const Value &v, keel::Allocator *mr)
{
    if ((v.type() != ValueType::DOUBLE && v.type() != ValueType::SINGLE && v.type() != ValueType::COMPLEX)
        || !v.dims().isVector())
        throw Error("tf2sos: b/a must be DOUBLE, SINGLE, or COMPLEX row/column vectors",
                    0, 0, "tf2sos", "", "numkit:tf2sos:type");
    const keel::usize n = v.numel();
    keel::Array<keel::c64> out(n, mr);
    if (v.type() == ValueType::COMPLEX) {
        const keel::c64 *p = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = p[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

void zp2sos_argcheck(const Value &zerosV, const Value &polesV)
{
    if (polesV.isEmpty())
        throw Error("zp2sos: at least one pole is required",
                     0, 0, "zp2sos", "", "numkit:zp2sos:noPoles");
    if (!zerosV.isEmpty() && zerosV.type() != ValueType::COMPLEX
        && zerosV.type() != ValueType::DOUBLE && zerosV.type() != ValueType::SINGLE)
        throw Error("zp2sos: zeros/poles must be double, single, or complex vectors",
                    0, 0, "zp2sos", "", "numkit:zp2sos:inputType");
    if (polesV.type() != ValueType::COMPLEX && polesV.type() != ValueType::DOUBLE
        && polesV.type() != ValueType::SINGLE)
        throw Error("zp2sos: zeros/poles must be double, single, or complex vectors",
                    0, 0, "zp2sos", "", "numkit:zp2sos:inputType");
}

Value sosFromResult(const keel::signal::SosResult &res, keel::Allocator *mr)
{
    auto sos = Value::matrix(res.num_sections, 6, ValueType::DOUBLE, mr);
    if (!res.sos_col_major.empty())
        keel::copy_n(sos.doubleDataMut(), res.sos_col_major.data(), res.sos_col_major.size());
    return sos;
}

} // namespace

keel::Tuple<Value, keel::f64>
zp2sosWithGain(const Value &zerosV, const Value &polesV, keel::f64 gain, bool surplusAtOrigin, keel::Allocator *mr)
{
    zp2sos_argcheck(zerosV, polesV);
    const bool anySingle = (zerosV.type() == ValueType::SINGLE) || (polesV.type() == ValueType::SINGLE);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto zeros = readComplexVec(zerosV, "zp2sos", scratch.allocator());
    auto poles = readComplexVec(polesV, "zp2sos", scratch.allocator());

    if (zeros.size() > poles.size())
        throw Error("zp2sos: more zeros than poles is not supported",
                     0, 0, "zp2sos", "", "numkit:zp2sos:moreZeros");

    keel::signal::SosResult res;
    const keel::signal::SignalStatus st = keel::signal::zp2sos(
        zeros.as_ref(), poles.as_ref(), gain, surplusAtOrigin, res, scratch.allocator());
    if (st != keel::signal::SignalStatus::kOk)
        throw Error("zp2sos: complex root has no conjugate in input",
                     0, 0, "zp2sos", "", "numkit:zp2sos:noConj");

    Value sos = sosFromResult(res, mr);
    return keel::make_tuple(detail::narrowDoubleToSingleIf(keel::move(sos), anySingle, mr),
                            res.gain);
}

Value zp2sos(const Value &zerosV, const Value &polesV, keel::f64 gain, keel::Allocator *mr)
{
    auto [sos, g] = zp2sosWithGain(zerosV, polesV, gain, /*surplusAtOrigin=*/true, mr);
    const keel::usize L = sos.dims().rows();
    if (sos.type() == ValueType::SINGLE) {
        keel::f32 *p = sos.singleDataMut();
        const keel::f32 gf = keel::as<keel::f32>(g);
        p[0 * L + 0] *= gf;
        p[1 * L + 0] *= gf;
        p[2 * L + 0] *= gf;
    } else {
        keel::f64 *p = sos.doubleDataMut();
        p[0 * L + 0] *= g;
        p[1 * L + 0] *= g;
        p[2 * L + 0] *= g;
    }
    return sos;
}

keel::Tuple<Value, keel::f64>
tf2sosWithGain(const Value &b, const Value &a, keel::Allocator *mr)
{
    const bool anySingle = (b.type() == ValueType::SINGLE) || (a.type() == ValueType::SINGLE);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = coeffsAsVector(b, scratch.allocator());
    auto av = coeffsAsVector(a, scratch.allocator());
    if (av.empty() || av[0] == keel::c64(0.0, 0.0))
        throw Error("tf2sos: a(1) must be nonzero",
                     0, 0, "tf2sos", "", "numkit:tf2sos:zeroLead");
    if (bv.empty())
        throw Error("tf2sos: b is empty",
                     0, 0, "tf2sos", "", "numkit:tf2sos:emptyB");

    keel::signal::SosResult res;
    const keel::signal::SignalStatus st =
        keel::signal::tf2sos(bv.as_ref(), av.as_ref(), res, scratch.allocator());
    if (st != keel::signal::SignalStatus::kOk)
        throw Error("zp2sos: complex root has no conjugate in input",
                     0, 0, "zp2sos", "", "numkit:zp2sos:noConj");

    Value sos = sosFromResult(res, mr);
    return keel::make_tuple(detail::narrowDoubleToSingleIf(keel::move(sos), anySingle, mr),
                            res.gain);
}

Value tf2sos(const Value &b, const Value &a, keel::Allocator *mr)
{
    auto [sos, g] = tf2sosWithGain(b, a, mr);
    const keel::usize L = sos.dims().rows();
    if (sos.type() == ValueType::SINGLE) {
        keel::f32 *p = sos.singleDataMut();
        const keel::f32 gf = keel::as<keel::f32>(g);
        p[0 * L + 0] *= gf;
        p[1 * L + 0] *= gf;
        p[2 * L + 0] *= gf;
    } else {
        keel::f64 *p = sos.doubleDataMut();
        p[0 * L + 0] *= g;
        p[1 * L + 0] *= g;
        p[2 * L + 0] *= g;
    }
    return sos;
}

} // namespace numkit::signal
