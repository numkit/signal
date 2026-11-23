// toolboxes/signal/src/filter_implementation/conversions.cpp
//
// zp2sos / tf2sos — convert filter representations to SOS form. The
// cascade applicator sosfilt lives in digital_filtering/sosfilt.cpp.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/filter_implementation/conversions.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <tuple>

#include "../dsp_helpers.hpp"
#include "../signal_detail.hpp"
#include <numkit/ops/poly_helpers.hpp>

#include <algorithm>
#include <utility>

namespace numkit::signal {

namespace {

using numkit::ops::polyRootsDurandKerner;

constexpr keel::f64 kImagThresh = 1e-10;  // |Im(r)| below this → treat as real

inline bool isReal(keel::c64 r)
{
    return keel::abs(r.imag()) <= kImagThresh;
}

struct RootPair { keel::c64 a, b; bool isPair; };

bool popPair(keel::Array<keel::c64> &pool, RootPair &out)
{
    if (pool.empty()) return false;
    // Section order (MATLAB R2025b, probed on buttap N=4/6, cheb1/cheb2/
    // ellip N=4): pole pairs are emitted by ASCENDING |p|, ties by
    // ascending |Re(p)| (buttap's equal-radius pairs → the pair nearest
    // the imaginary axis first). The old max-|p| pick emitted the
    // opposite order (bugs/closed/signal/zp2sos-section-order).
    keel::usize pickIdx = 0;
    auto key = [&](keel::usize i) {
        return std::make_pair(keel::abs(pool[i]), keel::abs(pool[i].real()));
    };
    for (keel::usize i = 1; i < pool.size(); ++i) {
        if (key(i) < key(pickIdx)) pickIdx = i;
    }
    const keel::c64 pick = pool[pickIdx];
    if (isReal(pick)) {
        keel::usize mateIdx = pool.size();
        keel::f64 bestDist = keel::Limits<keel::f64>::infinity();
        for (keel::usize i = 0; i < pool.size(); ++i) {
            if (i == pickIdx) continue;
            if (!isReal(pool[i])) continue;
            const keel::f64 d = keel::abs(pool[i] - pick);
            if (d < bestDist) { bestDist = d; mateIdx = i; }
        }
        if (mateIdx < pool.size()) {
            out = { keel::c64(pick.real(), 0.0), keel::c64(pool[mateIdx].real(), 0.0), true };
            const keel::usize hi = std::max(pickIdx, mateIdx);
            const keel::usize lo = std::min(pickIdx, mateIdx);
            pool.erase(hi);
            pool.erase(lo);
        } else {
            out = { keel::c64(pick.real(), 0.0), keel::c64(0.0, 0.0), false };
            pool.erase(pickIdx);
        }
    } else {
        const keel::c64 conjP = keel::conj(pick);
        keel::usize mateIdx = pool.size();
        keel::f64 bestDist = keel::Limits<keel::f64>::infinity();
        for (keel::usize i = 0; i < pool.size(); ++i) {
            if (i == pickIdx) continue;
            const keel::f64 d = keel::abs(pool[i] - conjP);
            if (d < bestDist) { bestDist = d; mateIdx = i; }
        }
        if (mateIdx >= pool.size())
            throw Error("zp2sos: complex root has no conjugate in input",
                         0, 0, "zp2sos", "", "numkit:zp2sos:noConj");
        out = { pick, pool[mateIdx], true };
        const keel::usize hi = std::max(pickIdx, mateIdx);
        const keel::usize lo = std::min(pickIdx, mateIdx);
        pool.erase(hi);
        pool.erase(lo);
    }
    return true;
}

struct QuadCoeffs { keel::f64 b1, b2; };

QuadCoeffs pairToQuad(const RootPair &p)
{
    if (!p.isPair) {
        return { -p.a.real(), 0.0 };
    }
    if (!isReal(p.a) || !isReal(p.b)) {
        return { -2.0 * p.a.real(),
                 p.a.real() * p.a.real() + p.a.imag() * p.a.imag() };
    }
    return { -(p.a.real() + p.b.real()),
             p.a.real() * p.b.real() };
}

inline keel::c64 pairCentroid(const RootPair &p)
{
    if (!p.isPair) return p.a;
    return (p.a + p.b) * 0.5;
}

bool popClosestZeroPair(keel::Array<keel::c64> &zeros, keel::c64 target,
                        RootPair &out)
{
    if (zeros.empty()) return false;
    keel::usize pickIdx = 0;
    keel::f64 bestDist = keel::abs(zeros[0] - target);
    for (keel::usize i = 1; i < zeros.size(); ++i) {
        const keel::f64 d = keel::abs(zeros[i] - target);
        if (d < bestDist) { bestDist = d; pickIdx = i; }
    }
    const keel::c64 pick = zeros[pickIdx];
    if (isReal(pick)) {
        keel::usize mateIdx = zeros.size();
        keel::f64 bestMate = keel::Limits<keel::f64>::infinity();
        for (keel::usize i = 0; i < zeros.size(); ++i) {
            if (i == pickIdx) continue;
            if (!isReal(zeros[i])) continue;
            const keel::f64 d = keel::abs(zeros[i] - target);
            if (d < bestMate) { bestMate = d; mateIdx = i; }
        }
        if (mateIdx < zeros.size()) {
            out = { keel::c64(pick.real(), 0.0),
                    keel::c64(zeros[mateIdx].real(), 0.0), true };
            const keel::usize hi = std::max(pickIdx, mateIdx);
            const keel::usize lo = std::min(pickIdx, mateIdx);
            zeros.erase(hi);
            zeros.erase(lo);
        } else {
            out = { keel::c64(pick.real(), 0.0), keel::c64(0.0, 0.0), false };
            zeros.erase(pickIdx);
        }
    } else {
        const keel::c64 conjP = keel::conj(pick);
        keel::usize mateIdx = zeros.size();
        keel::f64 bestDist2 = keel::Limits<keel::f64>::infinity();
        for (keel::usize i = 0; i < zeros.size(); ++i) {
            if (i == pickIdx) continue;
            const keel::f64 d = keel::abs(zeros[i] - conjP);
            if (d < bestDist2) { bestDist2 = d; mateIdx = i; }
        }
        if (mateIdx >= zeros.size())
            throw Error("zp2sos: complex zero has no conjugate in input",
                         0, 0, "zp2sos", "", "numkit:zp2sos:noConj");
        out = { pick, zeros[mateIdx], true };
        const keel::usize hi = std::max(pickIdx, mateIdx);
        const keel::usize lo = std::min(pickIdx, mateIdx);
        zeros.erase(hi);
        zeros.erase(lo);
    }
    return true;
}

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

Value buildSosMatrix(const keel::f64 *b1s, const keel::f64 *b2s, const keel::f64 *a1s, const keel::f64 *a2s, const keel::u8 *noZeros, bool surplusAtOrigin, keel::usize L, keel::f64 leadingGain, keel::Allocator *mr)
{
    auto sos = Value::matrix(L, 6, ValueType::DOUBLE, mr);
    keel::f64 *p = sos.doubleDataMut();
    for (keel::usize r = 0; r < L; ++r) {
        const keel::f64 scale = (r == 0) ? leadingGain : 1.0;
        if (noZeros && noZeros[r] && surplusAtOrigin) {
            // zp2sos: a section carrying no finite zeros gets MATLAB's
            // surplus zeros at the ORIGIN, i.e. numerator = scale*z^-2 ->
            // [0 0 scale]. Reconstructing the transfer function then keeps
            // the z^-2 delay (degree-deficient numerator) MATLAB produces.
            p[0 * L + r] = 0.0;
            p[1 * L + r] = 0.0;
            p[2 * L + r] = scale;
        } else {
            // Normal section, or tf2sos (surplusAtOrigin=false): keep the
            // numerator left-aligned [scale  scale*b1  scale*b2] so the
            // section product reproduces the ORIGINAL b polynomial exactly
            // (b2s/b1s are 0 for a no-zero section -> [scale 0 0]).
            p[0 * L + r] = scale;             // b0
            p[1 * L + r] = scale * b1s[r];    // b1
            p[2 * L + r] = scale * b2s[r];    // b2
        }
        p[3 * L + r] = 1.0;               // a0
        p[4 * L + r] = a1s[r];            // a1
        p[5 * L + r] = a2s[r];            // a2
    }
    return sos;
}

inline keel::Array<keel::c64> coeffsAsVector(const Value &v, keel::Allocator *mr)
{
    if ((v.type() != ValueType::DOUBLE && v.type() != ValueType::SINGLE && v.type() != ValueType::COMPLEX) || !v.dims().isVector())
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

} // namespace

std::tuple<Value, keel::f64>
zp2sosWithGain(const Value &zerosV, const Value &polesV, keel::f64 gain, bool surplusAtOrigin, keel::Allocator *mr)
{
    zp2sos_argcheck(zerosV, polesV);

    const bool anySingle = (zerosV.type() == ValueType::SINGLE) || (polesV.type() == ValueType::SINGLE);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto zeros = readComplexVec(zerosV, "zp2sos", scratch.allocator());
    auto poles = readComplexVec(polesV, "zp2sos", scratch.allocator());

    const keel::usize L = (poles.size() + 1) / 2;

    auto b1s = keel::Array<keel::f64>(L, scratch.allocator());
    auto b2s = keel::Array<keel::f64>(L, scratch.allocator());
    auto a1s = keel::Array<keel::f64>(L, scratch.allocator());
    auto a2s = keel::Array<keel::f64>(L, scratch.allocator());
    auto noZeros = keel::Array<keel::u8>(L, scratch.allocator());
    for (keel::usize s = 0; s < L; ++s) {
        RootPair polePair;
        if (!popPair(poles, polePair))
            throw Error("zp2sos: internal — ran out of poles",
                         0, 0, "zp2sos", "", "numkit:zp2sos:internal");
        const auto polQ = pairToQuad(polePair);
        a1s[s] = polQ.b1;
        a2s[s] = polQ.b2;

        if (!zeros.empty()) {
            RootPair zeroPair;
            (void) popClosestZeroPair(zeros, pairCentroid(polePair), zeroPair);
            const auto zerQ = pairToQuad(zeroPair);
            b1s[s] = zerQ.b1;
            b2s[s] = zerQ.b2;
            noZeros[s] = 0;
        } else {
            b1s[s] = 0.0;
            b2s[s] = 0.0;
            noZeros[s] = 1;   // surplus zeros -> placed at origin ([0 0 g])
        }
    }
    if (!zeros.empty())
        throw Error("zp2sos: more zeros than poles is not supported",
                     0, 0, "zp2sos", "", "numkit:zp2sos:moreZeros");

    Value sos = buildSosMatrix(b1s.data(), b2s.data(), a1s.data(), a2s.data(), noZeros.data(), surplusAtOrigin, L, /*leadingGain=*/1.0, mr);
    return std::make_tuple(detail::narrowDoubleToSingleIf(std::move(sos), anySingle, mr),
                           gain);
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

std::tuple<Value, keel::f64>
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

    const keel::f64 gain = (bv[0] / av[0]).real();
    auto zeros = polyRootsDurandKerner(bv.data(), bv.size(), scratch.allocator());
    auto poles = polyRootsDurandKerner(av.data(), av.size(), scratch.allocator());

    auto toCplxVec = [&](const keel::Array<keel::c64> &v) -> Value {
        if (v.empty())
            return Value::matrix(0, 0, ValueType::COMPLEX, mr);
        auto r = Value::matrix(v.size(), 1, ValueType::COMPLEX, mr);
        keel::c64 *p = r.complexDataMut();
        for (keel::usize i = 0; i < v.size(); ++i) p[i] = v[i];
        return r;
    };
    // tf2sos must reproduce the ORIGINAL b polynomial: surplus zeros stay
    // left-aligned at infinity ([g 0 0]), not pushed to the origin.
    auto [sos, g] = zp2sosWithGain(toCplxVec(zeros), toCplxVec(poles), gain, /*surplusAtOrigin=*/false, mr);
    return std::make_tuple(detail::narrowDoubleToSingleIf(std::move(sos), anySingle, mr), g);
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
