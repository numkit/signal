// toolboxes/signal/src/transforms/fwht.cpp
//
// Fast Walsh-Hadamard transform + inverse delegating to keel::signal::fwht / ifwht.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/transforms/fwht.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

inline bool isPowerOfTwo(keel::usize n) { return n != 0 && (n & (n - 1)) == 0; }

inline keel::usize nextPow2(keel::usize n) {
    if (n <= 1) return 1;
    keel::usize p = 1;
    while (p < n) p <<= 1;
    return p;
}

keel::usize resolveLen(keel::usize L, keel::usize n) {
    if (n == 0) return nextPow2(L);
    if (!isPowerOfTwo(n))
        throw Error("fwht: transform length must be a power of 2",
                     0, 0, "fwht", "", "numkit:fwht:badLength");
    return n;
}

keel::signal::FwhtOrdering parseOrdering(keel::StringRef ordering) {
    if (ordering == "sequency") return keel::signal::FwhtOrdering::kSequency;
    if (ordering == "hadamard") return keel::signal::FwhtOrdering::kHadamard;
    if (ordering == "dyadic")   return keel::signal::FwhtOrdering::kDyadic;
    throw Error("fwht: ordering must be 'sequency', 'hadamard', or 'dyadic'",
                0, 0, "fwht", "", "numkit:fwht:badOrdering");
}

Value runWalshHadamard(const Value &x, keel::usize n,
                       keel::StringRef ordering, bool forward,
                       keel::Allocator *mr) {
    const keel::signal::FwhtOrdering ord = parseOrdering(ordering);
    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("fwht: input must be a vector or matrix",
                    0, 0, "fwht", "", "numkit:fwht:notMatrix");
    if (x.numel() == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    const keel::usize rows = d.rows();
    const keel::usize cols = d.cols();
    const bool isRowVec = (rows == 1);
    const keel::usize L = isRowVec ? cols : rows;
    const keel::usize nCols = isRowVec ? 1u : cols;
    const keel::usize N = resolveLen(L, n);

    Value out = isRowVec
        ? Value::matrix(1, N, ValueType::DOUBLE, mr)
        : Value::matrix(N, nCols, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> srcCol(L, 0.0, scratch.allocator());

    for (keel::usize c = 0; c < nCols; ++c) {
        if (isRowVec) {
            for (keel::usize i = 0; i < L; ++i) srcCol[i] = x.elemAsDouble(i);
        } else {
            for (keel::usize i = 0; i < L; ++i) srcCol[i] = x.elemAsDouble(i + c * rows);
        }
        keel::f64 *dstCol = isRowVec ? od : od + c * N;
        if (forward) {
            (void)keel::signal::fwht(srcCol.as_ref(), keel::MutArrayRef<keel::f64>(dstCol, N),
                                     ord, scratch.allocator());
        } else {
            (void)keel::signal::ifwht(srcCol.as_ref(), keel::MutArrayRef<keel::f64>(dstCol, N),
                                      ord, scratch.allocator());
        }
    }
    return out;
}

} // namespace

Value fwht(const Value &x, keel::usize n, keel::StringRef ordering, keel::Allocator *mr) {
    return runWalshHadamard(x, n, ordering, /*forward=*/true, mr);
}

Value ifwht(const Value &y, keel::usize n, keel::StringRef ordering, keel::Allocator *mr) {
    return runWalshHadamard(y, n, ordering, /*forward=*/false, mr);
}

} // namespace numkit::signal
