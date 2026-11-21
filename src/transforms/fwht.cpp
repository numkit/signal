// toolboxes/signal/src/transforms/fwht.cpp
//
// Fast Walsh-Hadamard transform + inverse.
//
// Algorithm (natural / Hadamard order):
//   1. Pad x to length N = next power of 2 above length(x), or to the
//      explicit n if requested (n must be a power of 2).
//   2. Run the in-place radix-2 Hadamard butterfly:
//        h = 1
//        while h < N:
//          for i in [0, 2h, 4h, ...]:
//            for j in [i, i+h):
//              (x[j], x[j+h]) := (x[j] + x[j+h], x[j] - x[j+h])
//          h *= 2
//   3. For fwht: divide by N (so y(1) = mean(x), matching MATLAB).
//      For ifwht: no division (H · H = N · I → cancels).
//   4. For non-natural orderings, permute:
//        - dyadic:   bit-reversal of indices
//        - sequency: bit-reversal then gray-to-binary (i ↔ g(i)=i XOR i>>1)
//
// References:
//   - Beauchamp, "Applications of Walsh and Related Functions", 1984.
//   - Walsh, "A Closed Set of Normal Orthogonal Functions",
//     Amer. J. Math. 45 (1923) 5-24.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/signal/transforms/fwht.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace numkit::signal {

namespace {

inline bool isPowerOfTwo(keel::usize n) { return n != 0 && (n & (n - 1)) == 0; }

inline keel::usize nextPow2(keel::usize n)
{
    if (n <= 1) return 1;
    keel::usize p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline keel::i32 log2_size(keel::usize n)
{
    keel::i32 k = 0;
    while ((keel::usize(1) << k) < n) ++k;
    return k;
}

// Reverse the low `bits` bits of `i`.
inline keel::usize bitReverse(keel::usize i, keel::i32 bits)
{
    keel::usize r = 0;
    for (keel::i32 b = 0; b < bits; ++b) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

// Radix-2 Hadamard butterfly (natural-order) — in-place.
void hadamardButterfly(keel::f64 *col, keel::usize N)
{
    for (keel::usize h = 1; h < N; h <<= 1) {
        for (keel::usize i = 0; i < N; i += (h << 1)) {
            for (keel::usize j = 0; j < h; ++j) {
                const keel::f64 a = col[i + j];
                const keel::f64 b = col[i + j + h];
                col[i + j]     = a + b;
                col[i + j + h] = a - b;
            }
        }
    }
}

// Permute from natural Hadamard order to dyadic (Paley): out[i] = in[bitrev(i)].
void permuteToDyadic(const keel::f64 *in, keel::f64 *out, keel::usize N, keel::i32 log2N)
{
    for (keel::usize i = 0; i < N; ++i)
        out[i] = in[bitReverse(i, log2N)];
}

// Permute from natural Hadamard order to sequency (Walsh).
// Probed against MATLAB: sequency[i] = dyadic[g(i)] where g(i) = i XOR (i>>1).
// Combined transform from natural: sequency[i] = natural[bitrev(g(i))].
void permuteToSequency(const keel::f64 *in, keel::f64 *out, keel::usize N, keel::i32 log2N)
{
    for (keel::usize i = 0; i < N; ++i) {
        const keel::usize g = i ^ (i >> 1);
        out[i] = in[bitReverse(g, log2N)];
    }
}

// Inverse permutations: given the externally-ordered y, undo the
// permutation to get a natural-ordered buffer ready for the inverse
// butterfly. For dyadic the inverse permutation is bit-reverse again
// (bitrev is its own inverse). For sequency the inverse is:
//   natural[bitrev(g(i))] = sequency[i]  →  natural[k] = sequency[i]
//   with k = bitrev(g(i)). The inverse mapping inverts both steps.
void unpermuteFromSequency(const keel::f64 *in, keel::f64 *out, keel::usize N, keel::i32 log2N)
{
    // Forward: out_seq[i] = in_nat[bitrev(g(i))].
    // Inverse: in_nat[k]  = out_seq[i] where bitrev(g(i)) = k.
    // We compute the inverse by iterating i and placing into the natural-
    // order slot bitrev(g(i)).
    for (keel::usize i = 0; i < N; ++i) {
        const keel::usize g = i ^ (i >> 1);
        out[bitReverse(g, log2N)] = in[i];
    }
}

// Resolve user-supplied n: 0 → next power of 2 above length(x). Otherwise
// require power of 2.
keel::usize resolveLen(keel::usize L, keel::usize n)
{
    if (n == 0) return nextPow2(L);
    if (!isPowerOfTwo(n))
        throw Error("fwht: transform length must be a power of 2",
                     0, 0, "fwht", "", "numkit:fwht:badLength");
    return n;
}

// Apply fwht / ifwht to one column of length L (source `srcCol`), writing
// to `dstCol` (length N where N = resolveLen(L, n)). `forward` toggles
// the 1/N scaling.
void transformOneColumn(const keel::f64 *srcCol, keel::usize L,
                         keel::f64 *dstCol, keel::usize N, keel::i32 log2N,
                         keel::StringRef ordering, bool forward,
                         keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> buf(N, 0.0, scratch.allocator());

    if (forward) {
        // Forward: copy / truncate / zero-pad input.
        const keel::usize toCopy = std::min(L, N);
        for (keel::usize i = 0; i < toCopy; ++i) buf[i] = srcCol[i];
    } else {
        // Inverse: input is already in `ordering`; undo permutation
        // into natural order before applying the butterfly.
        keel::Array<keel::f64> raw(N, 0.0, scratch.allocator());
        const keel::usize toCopy = std::min(L, N);
        for (keel::usize i = 0; i < toCopy; ++i) raw[i] = srcCol[i];
        if (ordering == "hadamard") {
            std::copy(raw.begin(), raw.end(), buf.begin());
        } else if (ordering == "dyadic") {
            // bit-reverse is self-inverse.
            permuteToDyadic(raw.data(), buf.data(), N, log2N);
        } else { // sequency
            unpermuteFromSequency(raw.data(), buf.data(), N, log2N);
        }
    }

    hadamardButterfly(buf.data(), N);

    if (forward) {
        // Normalise + permute into requested ordering.
        const keel::f64 inv = 1.0 / keel::as<keel::f64>(N);
        for (keel::usize i = 0; i < N; ++i) buf[i] *= inv;
        if (ordering == "hadamard") {
            for (keel::usize i = 0; i < N; ++i) dstCol[i] = buf[i];
        } else if (ordering == "dyadic") {
            permuteToDyadic(buf.data(), dstCol, N, log2N);
        } else { // sequency
            permuteToSequency(buf.data(), dstCol, N, log2N);
        }
    } else {
        // ifwht: H · H = N·I, so the butterfly inverse needs NO 1/N scaling.
        for (keel::usize i = 0; i < N; ++i) dstCol[i] = buf[i];
    }
}

void fwht_argcheck(const Value &x, keel::StringRef ordering)
{
    if (ordering != "sequency" && ordering != "hadamard" && ordering != "dyadic")
        throw Error("fwht: ordering must be 'sequency', 'hadamard', or 'dyadic'",
                     0, 0, "fwht", "", "numkit:fwht:badOrdering");
    if (x.dims().ndim() > 2)
        throw Error("fwht: input must be a vector or matrix",
                     0, 0, "fwht", "", "numkit:fwht:notMatrix");
}

Value runWalshHadamard(const Value &x, keel::usize n,
                       keel::StringRef ordering, bool forward,
                       keel::Allocator *mr)
{
    fwht_argcheck(x, ordering);
    if (ordering != "sequency" && ordering != "hadamard" && ordering != "dyadic")
        throw Error("fwht: ordering must be 'sequency', 'hadamard', or 'dyadic'",
                     0, 0, "fwht", "", "numkit:fwht:badOrdering");

    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("fwht: input must be a vector or matrix",
                     0, 0, "fwht", "", "numkit:fwht:notMatrix");
    if (x.numel() == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    // Treat the input as columns. Row vectors become a single row.
    const keel::usize rows = d.rows();
    const keel::usize cols = d.cols();
    const bool isRowVec = (rows == 1);
    const keel::usize L = isRowVec ? cols : rows;
    const keel::usize nCols = isRowVec ? 1u   : cols;
    const keel::usize N = resolveLen(L, n);
    const keel::i32 log2N = log2_size(N);

    // Output orientation matches MATLAB: row vector in → row vector out
    // (1 × N), otherwise column-wise transform produces N × cols.
    Value out = isRowVec
        ? Value::matrix(1, N, ValueType::DOUBLE, mr)
        : Value::matrix(N, nCols, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> srcCol(L, 0.0, scratch.allocator());

    for (keel::usize c = 0; c < nCols; ++c) {
        // Gather source column.
        if (isRowVec) {
            for (keel::usize i = 0; i < L; ++i)
                srcCol[i] = x.elemAsDouble(i);
        } else {
            for (keel::usize i = 0; i < L; ++i)
                srcCol[i] = x.elemAsDouble(i + c * rows);
        }

        // Output column. For row-vector input the "column" lives flat.
        keel::f64 *dstCol = isRowVec ? od : od + c * N;
        transformOneColumn(srcCol.data(), L, dstCol, N, log2N,
                            ordering, forward, mr);
    }

    return out;
}

} // namespace

Value fwht(const Value &x, keel::usize n, keel::StringRef ordering,
           keel::Allocator *mr)
{
    return runWalshHadamard(x, n, ordering, /*forward=*/true, mr);
}

Value ifwht(const Value &y, keel::usize n, keel::StringRef ordering,
            keel::Allocator *mr)
{
    return runWalshHadamard(y, n, ordering, /*forward=*/false, mr);
}

} // namespace numkit::signal
