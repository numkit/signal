// toolboxes/signal/src/digital_filtering/poly_utils.cpp
// Signal Processing toolbox polynomial utilities — polyscale + polystab
// (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/digital_filtering/poly_utils.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../signal_detail.hpp"

namespace numkit::signal {

namespace {

inline keel::c64 elemAsComplex(const Value &v, keel::usize k)
{
    if (v.type() == ValueType::COMPLEX) return v.complexData()[k];
    return keel::c64(v.elemAsDouble(k), 0.0);
}

inline bool isComplex(const Value &v)
{
    return v.type() == ValueType::COMPLEX;
}

inline bool isVectorShape(const Value &v)
{
    const Dims &d = v.dims();
    if (d.ndim() > 2) return false;
    return v.isEmpty() || d.rows() == 1 || d.cols() == 1;
}

void polyscale_argcheck(const Value &p, const Value &scale)
{
    detail::requireFloatInput(p, "polyscale", "p", 1);
    detail::requireFloatInput(scale, "polyscale", "scale", 2);
    if (p.isEmpty()) return;
    const keel::usize n = isVectorShape(p) ? p.numel() : p.dims().cols();
    const keel::usize scaleN = scale.numel();
    if (scaleN != 1 && scaleN != n) {
        throw numkit::Error(
            "polyscale: scale must be a scalar or a vector matching the polynomial length",
            0, 0, "polyscale", "", "numkit:polyscale:BadScale");
    }
}

void polystab_argcheck(const Value &a)
{
    detail::requireFloatInput(a, "polystab");
    if (!isVectorShape(a)) {
        throw numkit::Error("Input must be a vector.",
                            0, 0, "polystab", "", "numkit:polystab:notVector");
    }
}

} // anonymous namespace

Value polyscale(const Value &p, const Value &scale, keel::Allocator *mr)
{
    polyscale_argcheck(p, scale);
    const bool pSingle = (p.type() == ValueType::SINGLE);
    if (p.isEmpty())
        return Value::matrix(0, 0, pSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    const Dims &pd = p.dims();
    const keel::usize m = isVectorShape(p) ? 1 : pd.rows();
    const keel::usize n = isVectorShape(p) ? p.numel() : pd.cols();
    const keel::usize total = m * n;
    const bool wantComplex = isComplex(p) || isComplex(scale);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (wantComplex) {
        keel::Array<keel::c64> pBuf(total, scratch.allocator());
        for (keel::usize i = 0; i < total; ++i) pBuf[i] = elemAsComplex(p, i);
        keel::Array<keel::c64> sBuf(scale.numel(), scratch.allocator());
        for (keel::usize i = 0; i < scale.numel(); ++i) sBuf[i] = elemAsComplex(scale, i);
        Value out = Value::matrix(m, n, ValueType::COMPLEX, mr);
        (void)keel::signal::polyscale_c64(
            pBuf.as_ref(), m, n, sBuf.as_ref(),
            keel::MutArrayRef<keel::c64>(out.complexDataMut(), total), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> pBuf(total, scratch.allocator());
    for (keel::usize i = 0; i < total; ++i) pBuf[i] = p.elemAsDouble(i);
    keel::Array<keel::f64> sBuf(scale.numel(), scratch.allocator());
    for (keel::usize i = 0; i < scale.numel(); ++i) sBuf[i] = scale.elemAsDouble(i);
    Value out = Value::matrix(m, n, ValueType::DOUBLE, mr);
    (void)keel::signal::polyscale(
        pBuf.as_ref(), m, n, sBuf.as_ref(),
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), total), scratch.allocator());
    return detail::narrowDoubleToSingleIf(keel::move(out), pSingle, mr);
}

Value polystab(const Value &a, keel::Allocator *mr)
{
    polystab_argcheck(a);
    const bool aSingle = (a.type() == ValueType::SINGLE);
    if (a.isEmpty())
        return Value::matrix(0, 0, aSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    const bool aComplex = isComplex(a);
    const keel::usize n = a.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (aComplex) {
        keel::Array<keel::c64> av(n, scratch.allocator());
        for (keel::usize k = 0; k < n; ++k) av[k] = elemAsComplex(a, k);
        keel::Array<keel::c64> res(scratch.allocator());
        (void)keel::signal::polystab_c64(av.as_ref(), res, scratch.allocator());
        Value out = Value::matrix(1, res.size(), ValueType::COMPLEX, mr);
        if (!res.empty()) keel::copy_n(out.complexDataMut(), res.data(), res.size());
        return out;
    }

    keel::Array<keel::f64> av(n, scratch.allocator());
    for (keel::usize k = 0; k < n; ++k) av[k] = a.elemAsDouble(k);
    keel::Array<keel::f64> res(scratch.allocator());
    (void)keel::signal::polystab(av.as_ref(), res, scratch.allocator());
    Value out = Value::matrix(1, res.size(), ValueType::DOUBLE, mr);
    if (!res.empty()) keel::copy_n(out.doubleDataMut(), res.data(), res.size());
    return detail::narrowDoubleToSingleIf(keel::move(out), aSingle, mr);
}

} // namespace numkit::signal
