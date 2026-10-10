// src/shims/elmat.cpp
//
// Standalone implementations of permute, ipermute, shiftdim, shiftdimAuto for signal toolbox.

#include <numkit/signal/internal/shims/elmat.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/shape_ops.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value_type.hpp>
#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace numkit::builtin {

namespace {

void validatePerm(const int *perm, std::size_t N, const char *fn)
{
    if (N == 0)
        throw Error(std::string(fn) + ": perm vector must not be empty",
                     0, 0, fn, "", std::string("nk:") + fn + ":emptyPerm");
    if (N > Dims::kMaxRank)
        throw Error(std::string(fn) + ": perm length exceeds 32",
                     0, 0, fn, "", std::string("nk:") + fn + ":tooManyDims");
    int sorted[Dims::kMaxRank];
    for (std::size_t i = 0; i < N; ++i) sorted[i] = perm[i];
    std::sort(sorted, sorted + N);
    for (std::size_t i = 0; i < N; ++i) {
        if (sorted[i] != static_cast<int>(i + 1))
            throw Error(std::string(fn) + ": perm must be a permutation of 1..N",
                         0, 0, fn, "", std::string("nk:") + fn + ":badPerm");
    }
}

void transposePage(const double *src, double *dst, size_t R, size_t C)
{
    constexpr size_t kBlock = 32;
    for (size_t ii = 0; ii < R; ii += kBlock) {
        const size_t iMax = std::min(ii + kBlock, R);
        for (size_t jj = 0; jj < C; jj += kBlock) {
            const size_t jMax = std::min(jj + kBlock, C);
            for (size_t j = jj; j < jMax; ++j) {
                const double *srcCol = src + j * R;
                double *dstRow = dst + j;
                for (size_t i = ii; i < iMax; ++i) {
                    dstRow[i * C] = srcCol[i];
                }
            }
        }
    }
}

Value reshapeHelper(const Value &x, Span<const size_t> dims, std::pmr::memory_resource *mr)
{
    Dims d(dims.data(), static_cast<int>(dims.size()));
    Value r = ops::createForDims(d, x.type(), mr);
    if (x.isComplex()) {
        std::copy(x.complexData(), x.complexData() + x.numel(), r.complexDataMut());
    } else {
        std::memcpy(r.rawDataMut(), x.rawData(), x.numel() * elementSize(x.type()));
    }
    return r;
}

} // namespace

Value permute(const Value &x, Span<const int> perm, std::pmr::memory_resource *mr)
{
    const std::size_t n = perm.size();
    validatePerm(perm.data(), n, "permute");

    const auto &dd = x.dims();
    const int inNd = std::max<int>(dd.ndim(), static_cast<int>(n));
    if (inNd > Dims::kMaxRank)
        throw Error("permute: rank exceeds 32",
                     0, 0, "permute", "", "nk:permute:tooManyDims");

    int p[Dims::kMaxRank];
    size_t inDims[Dims::kMaxRank];
    size_t outDimsArr[Dims::kMaxRank];
    for (int i = 0; i < inNd; ++i)
        p[i] = (i < static_cast<int>(n)) ? perm[i] : (i + 1);
    for (int i = 0; i < inNd; ++i) inDims[i] = dd.dim(i);
    for (int k = 0; k < inNd; ++k) outDimsArr[k] = inDims[p[k] - 1];

    if (x.type() != ValueType::DOUBLE) {
        const ValueType t = x.type();
        Value r = Value::matrixND(outDimsArr, inNd, t, mr);
        if (x.numel() == 0) return r;
        size_t inStrides[Dims::kMaxRank];
        computeStridesColMajor(Dims(inDims, inNd), inStrides);
        const size_t es = elementSize(t);
        const char *src = static_cast<const char *>(x.rawData());
        char *dst = static_cast<char *>(r.rawDataMut());
        size_t outCoords[Dims::kMaxRank] = {0};
        Dims outDimsObj(outDimsArr, inNd);
        size_t dstOff = 0;
        do {
            size_t srcOff = 0;
            for (int k = 0; k < inNd; ++k)
                srcOff += outCoords[k] * inStrides[p[k] - 1];
            std::memcpy(dst + dstOff * es, src + srcOff * es, es);
            ++dstOff;
        } while (incrementCoords(outCoords, outDimsObj));
        return r;
    }

    auto r = Value::matrixND(outDimsArr, inNd, ValueType::DOUBLE, mr);
    if (x.numel() == 0) return r;

    const double *src = x.doubleData();
    double *dst       = r.doubleDataMut();

    if (inNd >= 2 && p[0] == 2 && p[1] == 1) {
        bool tailIsIdentity = true;
        for (int k = 2; k < inNd; ++k)
            if (p[k] != k + 1) { tailIsIdentity = false; break; }
        if (tailIsIdentity) {
            const size_t inR = inDims[0], inC = inDims[1];
            size_t P = 1;
            for (int k = 2; k < inNd; ++k) P *= inDims[k];
            for (size_t pp = 0; pp < P; ++pp)
                transposePage(src + pp * inR * inC,
                              dst + pp * inC * inR,
                              inR, inC);
            return r;
        }
    }

    if (inNd == 3) {
        const size_t s[3] = {1, inDims[0], inDims[0] * inDims[1]};
        const size_t sa = s[p[0] - 1];
        const size_t sb = s[p[1] - 1];
        const size_t sc = s[p[2] - 1];
        const size_t outR = outDimsArr[0], outC = outDimsArr[1], outP = outDimsArr[2];
        size_t dstIdx = 0;
        for (size_t k = 0; k < outP; ++k) {
            const size_t baseK = k * sc;
            for (size_t j = 0; j < outC; ++j) {
                const size_t baseJK = baseK + j * sb;
                for (size_t i = 0; i < outR; ++i)
                    dst[dstIdx++] = src[baseJK + i * sa];
            }
        }
        return r;
    }

    if (inNd == 2) {
        const size_t s[2] = {1, inDims[0]};
        const size_t sa = s[p[0] - 1];
        const size_t sb = s[p[1] - 1];
        const size_t outR = outDimsArr[0], outC = outDimsArr[1];
        size_t dstIdx = 0;
        for (size_t j = 0; j < outC; ++j) {
            const size_t baseJ = j * sb;
            for (size_t i = 0; i < outR; ++i)
                dst[dstIdx++] = src[baseJ + i * sa];
        }
        return r;
    }

    size_t inStrides[Dims::kMaxRank];
    computeStridesColMajor(Dims(inDims, inNd), inStrides);

    size_t outCoords[Dims::kMaxRank] = {0};
    Dims outDimsObj(outDimsArr, inNd);
    size_t dstOff = 0;
    do {
        size_t srcOff = 0;
        for (int k = 0; k < inNd; ++k) {
            const int axIn = p[k] - 1;
            srcOff += outCoords[k] * inStrides[axIn];
        }
        dst[dstOff] = src[srcOff];
        ++dstOff;
    } while (incrementCoords(outCoords, outDimsObj));
    return r;
}

Value ipermute(const Value &x, Span<const int> perm, std::pmr::memory_resource *mr)
{
    const std::size_t n = perm.size();
    validatePerm(perm.data(), n, "ipermute");
    int inv[Dims::kMaxRank];
    for (std::size_t i = 0; i < n; ++i)
        inv[perm[i] - 1] = static_cast<int>(i + 1);
    return permute(x, Span<const int>(inv, n), mr);
}

Value shiftdim(const Value &x, int n, std::pmr::memory_resource *mr)
{
    if (n == 0) return x;

    const auto &d = x.dims();
    const int N = d.ndim();

    if (n > 0) {
        const int eff = (N > 0) ? (n % N) : 0;
        if (eff == 0) return x;
        constexpr int kMaxNd = Dims::kMaxRank;
        int perm[kMaxNd];
        for (int i = 0; i < N; ++i)
            perm[i] = ((i + eff) % N) + 1;
        return permute(x, Span<const int>(perm, static_cast<std::size_t>(N)), mr);
    }

    const int k = -n;
    const int newN = N + k;
    constexpr int kMaxNd = Dims::kMaxRank;
    if (newN > kMaxNd)
        throw Error("shiftdim: rank exceeds 32",
                     0, 0, "shiftdim", "", "nk:shiftdim:tooManyDims");
    size_t newDims[kMaxNd];
    for (int i = 0; i < k; ++i) newDims[i] = 1;
    for (int i = 0; i < N; ++i) newDims[k + i] = d.dim(i);
    return reshapeHelper(x, Span<const size_t>(newDims, static_cast<std::size_t>(newN)), mr);
}

ShiftDimAuto shiftdimAuto(const Value &x, std::pmr::memory_resource *mr)
{
    const auto &d = x.dims();
    const int N = d.ndim();
    int k = 0;
    while (k < N && d.dim(k) == 1) ++k;
    if (k == 0) return { x, 0 };
    if (k == N) k = N - 1;
    if (k <= 0) return { x, 0 };
    Value y = shiftdim(x, k, mr);
    const auto &yd = y.dims();
    const int Ny = yd.ndim();
    int newN = Ny;
    while (newN > 2 && yd.dim(newN - 1) == 1) --newN;
    if (newN < Ny) {
        constexpr int kMaxNd = Dims::kMaxRank;
        size_t trimmed[kMaxNd];
        for (int i = 0; i < newN; ++i) trimmed[i] = yd.dim(i);
        y = reshapeHelper(y, Span<const size_t>(trimmed, static_cast<std::size_t>(newN)), mr);
    }
    return { std::move(y), static_cast<size_t>(k) };
}

} // namespace numkit::builtin
