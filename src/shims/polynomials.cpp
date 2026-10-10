// src/shims/polynomials.cpp
//
// Standalone implementations of roots, poly, tf2zp, zp2tf for signal toolbox.

#include <nk/builtin/polyfun.hpp>
#include <nk/value/value.hpp>
#include <nk/value/scratch.hpp>
#include <nk/value/error.hpp>
#include <nk/ops/helpers.hpp>
#include <nk/ops/poly_helpers.hpp>

#include <cmath>
#include <cstring>
#include <memory_resource>
#include <tuple>

namespace nk::builtin {

namespace {

ScratchVec<double> readPolyAsDouble(const Value &p, const char *fn, std::pmr::memory_resource *mr)
{
    if (p.type() == ValueType::COMPLEX)
        throw Error(std::string(fn) + ": complex coefficient input is not supported",
                     0, 0, fn, "", std::string("nk:") + fn + ":complex");
    if (!p.dims().isVector() && !p.isScalar() && !p.isEmpty())
        throw Error(std::string(fn) + ": argument must be a vector",
                     0, 0, fn, "", std::string("nk:") + fn + ":notVector");
    const size_t n = p.numel();
    ScratchVec<double> v(n, mr);
    for (size_t i = 0; i < n; ++i) v[i] = p.elemAsDouble(i);
    return v;
}

ScratchVec<Complex> readVecAsComplex(const Value &p, const char *fn, std::pmr::memory_resource *mr)
{
    if (!p.dims().isVector() && !p.isScalar() && !p.isEmpty())
        throw Error(std::string(fn) + ": argument must be a vector",
                     0, 0, fn, "", std::string("nk:") + fn + ":notVector");
    const size_t n = p.numel();
    ScratchVec<Complex> v(n, mr);
    if (p.isComplex()) {
        const Complex *cd = p.complexData();
        for (size_t i = 0; i < n; ++i) v[i] = cd[i];
    } else {
        for (size_t i = 0; i < n; ++i) v[i] = Complex(p.elemAsDouble(i), 0.0);
    }
    return v;
}

Value rowFromVec(const double *data, size_t n, std::pmr::memory_resource *mr)
{
    auto out = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n > 0 && data)
        std::memcpy(out.doubleDataMut(), data, n * sizeof(double));
    return out;
}

Value realColIfFlat(const Complex *roots, size_t n, std::pmr::memory_resource *mr)
{
    bool anyComplex = false;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(roots[i].imag()) > 1e-12 * (std::abs(roots[i].real()) + 1.0)) {
            anyComplex = true;
            break;
        }
    }
    if (!anyComplex) {
        auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        for (size_t i = 0; i < n; ++i) out.doubleDataMut()[i] = roots[i].real();
        return out;
    }
    auto out = Value::complexMatrix(n, 1, mr);
    for (size_t i = 0; i < n; ++i) out.complexDataMut()[i] = roots[i];
    return out;
}

} // namespace

Value roots(const Value &p, std::pmr::memory_resource *mr)
{
    if (p.type() == ValueType::COMPLEX)
        throw Error("roots: complex coefficient input is not supported",
                     0, 0, "roots", "", "nk:roots:complex");
    if (!p.dims().isVector() && !p.isScalar() && !p.isEmpty())
        throw Error("roots: argument must be a vector",
                     0, 0, "roots", "", "nk:roots:notVector");

    ScratchArena scratch(mr);
    const std::size_t n = p.numel();
    auto coeffs = ScratchVec<double>(n, &scratch);
    for (std::size_t i = 0; i < n; ++i)
        coeffs[i] = p.elemAsDouble(i);

    auto rs = nk::ops::polyRootsDurandKerner(&scratch, coeffs.data(), coeffs.size());
    return realColIfFlat(rs.data(), rs.size(), mr);
}

Value poly(const Value &r, std::pmr::memory_resource *mr)
{
    if (r.isEmpty()) return rowFromVec(nullptr, 0, mr);
    ScratchArena scratch(mr);
    const size_t n = r.numel();
    auto cv = ScratchVec<Complex>(n, &scratch);
    if (r.isComplex()) {
        const Complex *p = r.complexData();
        for (size_t i = 0; i < n; ++i) cv[i] = p[i];
    } else {
        const double *p = r.doubleData();
        for (size_t i = 0; i < n; ++i) cv[i] = Complex(p[i], 0.0);
    }
    auto coeffs = nk::ops::polyExpandFromRoots(&scratch, cv.data(), n);
    return rowFromVec(coeffs.data(), coeffs.size(), mr);
}

std::tuple<Value, Value, Value>
tf2zp(const Value &b, const Value &a, std::pmr::memory_resource *mr)
{
    ScratchArena scratch(mr);
    auto bv = readPolyAsDouble(b, "tf2zp", &scratch);
    auto av = readPolyAsDouble(a, "tf2zp", &scratch);
    if (av.empty() || av[0] == 0.0)
        throw Error("tf2zp: leading denominator coefficient must be non-zero",
                     0, 0, "tf2zp", "", "nk:tf2zp:badDen");
    size_t bnz = 0;
    while (bnz < bv.size() && bv[bnz] == 0.0) ++bnz;
    if (bv.empty() || bnz == bv.size()) {
        auto z = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        auto pRoots = nk::ops::polyRootsDurandKerner(&scratch, av.data(), av.size());
        auto p = realColIfFlat(pRoots.data(), pRoots.size(), mr);
        auto k = Value::scalar(0.0, mr);
        return std::make_tuple(std::move(z), std::move(p), std::move(k));
    }
    auto zRoots = nk::ops::polyRootsDurandKerner(&scratch, bv.data(), bv.size());
    auto pRoots = nk::ops::polyRootsDurandKerner(&scratch, av.data(), av.size());
    const double k = bv[bnz] / av[0];

    return std::make_tuple(realColIfFlat(zRoots.data(), zRoots.size(), mr),
                           realColIfFlat(pRoots.data(), pRoots.size(), mr),
                           Value::scalar(k, mr));
}

std::tuple<Value, Value>
zp2tf(const Value &z, const Value &p, double k, std::pmr::memory_resource *mr)
{
    ScratchArena scratch(mr);
    auto zv = readVecAsComplex(z, "zp2tf", &scratch);
    auto pv = readVecAsComplex(p, "zp2tf", &scratch);
    auto bRaw = nk::ops::polyExpandFromRoots(&scratch, zv.data(), zv.size());
    auto aRaw = nk::ops::polyExpandFromRoots(&scratch, pv.data(), pv.size());
    for (auto &v : bRaw) v *= k;
    if (bRaw.size() < aRaw.size()) {
        const size_t off = aRaw.size() - bRaw.size();
        ScratchVec<double> bPad(aRaw.size(), &scratch);
        for (size_t i = 0; i < aRaw.size(); ++i) bPad[i] = 0.0;
        for (size_t i = 0; i < bRaw.size(); ++i) bPad[off + i] = bRaw[i];
        return std::make_tuple(rowFromVec(bPad.data(), bPad.size(), mr),
                               rowFromVec(aRaw.data(), aRaw.size(), mr));
    }
    return std::make_tuple(rowFromVec(bRaw.data(), bRaw.size(), mr),
                           rowFromVec(aRaw.data(), aRaw.size(), mr));
}

} // namespace nk::builtin
