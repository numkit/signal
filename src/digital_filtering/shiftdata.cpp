// toolboxes/signal/src/digital_filtering/shiftdata.cpp
//
// shiftdata + unshiftdata (Phase 4.4 of audio sweep). Bit-equal MATLAB
// R2025b shiftdata.m / unshiftdata.m.
//
// shiftdata(x):           shifted = shiftdim(x); perm=[]; nshifts=k
// shiftdata(x, dim):      perm = [dim, 1..dim-1, dim+1..ndims]
//                          shifted = permute(x, perm); nshifts=[]
// unshiftdata(x, perm=[], nshifts=k): y = shiftdim(x, -k)
// unshiftdata(x, perm, nshifts=[]):    y = ipermute(x, perm)

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/signal/digital_filtering/shiftdata.hpp>

#include <numkit/signal/internal/shims/elmat.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <tuple>

#include <cmath>

namespace numkit::signal {

namespace {

// Compute MATLAB ndims for a Value. MATLAB: ndims(x) = max(2, ndims_actual).
keel::usize mtlNdims(const Value &x)
{
    if (x.dims().is3D()) return 3;
    return 2;
}

} // anon

std::tuple<Value, Value, Value>
shiftdata(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim == 0) {
        // Auto path: shiftdim(x) drops leading singletons.
        auto res = numkit::builtin::shiftdimAuto(x, mr);
        // perm = []  (return as 1×0 double matrix)
        Value emptyPerm = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        Value nsh = Value::scalar(keel::as<keel::f64>(res.dropped), mr);
        return {std::move(res.v), emptyPerm, nsh};
    }
    if (dim < 1)
        throw Error("shiftdata: DIM must be >= 1",
                    0, 0, "shiftdata", "", "numkit:shiftdata:BadDim");

    // perm = [dim, 1..dim-1, dim+1..N]
    const keel::usize N = std::max<keel::usize>(keel::as<keel::usize>(dim), mtlNdims(x));
    keel::Array<keel::i32> perm(N, mr);
    perm[0] = dim;
    keel::usize k = 1;
    for (keel::i32 i = 1; i < dim; ++i) perm[k++] = i;
    for (keel::usize i = dim + 1; i <= N; ++i) perm[k++] = keel::as<keel::i32>(i);

    Value shifted = numkit::builtin::permute(x, keel::ArrayRef<keel::i32>(perm.data(), N), mr);

    // Build perm Value (1 × N row).
    Value permV = Value::matrix(1, N, ValueType::DOUBLE, mr);
    {
        keel::f64 *pd = permV.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) pd[i] = keel::as<keel::f64>(perm[i]);
    }
    Value emptyNsh = Value::matrix(0, 0, ValueType::DOUBLE, mr);
    return {std::move(shifted), std::move(permV), std::move(emptyNsh)};
}

// unshiftdata domain checks: perm must be a true permutation vector
// (the reference lets permute's own validation fire, but the observable
// rejection should be voiced for THIS entry); nshifts must be scalar —
// a complex value with nonneg real part is legal (the reference
// validates with mustBeNonnegative, which compares the real part).
static void unshiftdata_argcheck(const Value &perm, const Value &nshifts)
{
    if (!nshifts.isEmpty()
        && (nshifts.numel() != 1 || nshifts.isChar() || nshifts.isString()
            || nshifts.isCell() || nshifts.isStruct()))
        throw Error("unshiftdata: nshifts must be a real scalar",
                    0, 0, "unshiftdata", "", "numkit:unshiftdata:nshifts");
    if (perm.isEmpty())
        return;
    const keel::usize n = perm.numel();
    keel::Array<char> seen(n + 1, 0);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = perm.elemAsDouble(i);
        const keel::f64 vi = std::floor(v);
        if (v != vi || vi < 1.0 || vi > keel::as<keel::f64>(n)
            || seen[keel::as<keel::usize>(vi)])
            throw Error(keel::format("unshiftdata: perm must be a permutation of 1..{}", n),
                        0, 0, "unshiftdata", "", "numkit:unshiftdata:badPerm");
        seen[keel::as<keel::usize>(vi)] = 1;
    }
}

Value unshiftdata(const Value &x, const Value &perm, const Value &nshifts, keel::Allocator *mr)
{
    unshiftdata_argcheck(perm, nshifts);
    if (perm.isEmpty()) {
        // nshifts real part (complex scalars are legal — see argcheck).
        const keel::f64 ns = nshifts.isEmpty() ? 0.0
            : (nshifts.isComplex() ? nshifts.complexData()[0].real()
                                   : nshifts.toScalar());
        return numkit::builtin::shiftdim(x, -keel::as<keel::i32>(ns), mr);
    }
    // ipermute(x, perm)
    const keel::usize N = perm.numel();
    keel::Array<keel::i32> permVec(N, mr);
    for (keel::usize i = 0; i < N; ++i)
        permVec[i] = keel::as<keel::i32>(perm.elemAsDouble(i));
    return numkit::builtin::ipermute(x, keel::ArrayRef<keel::i32>(permVec.data(), N), mr);
}

} // namespace numkit::signal
