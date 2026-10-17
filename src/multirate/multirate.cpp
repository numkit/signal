// toolboxes/signal/src/multirate/multirate.cpp

#include <numkit/signal/multirate/multirate.hpp>
#include <numkit/signal/multirate/extras.hpp>          // upfirdn
#include <numkit/signal/filter_design/filter_design.hpp> // firls
#include <numkit/signal/filter_design/iir_designs.hpp>   // cheby1
#include <numkit/signal/digital_filtering/filter.hpp>    // filtfilt
#include <numkit/signal/windows/windows.hpp>           // kaiser

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName / isIntegerType

#include "../signal_detail.hpp"     // shared argcheck cores
#include "../dsp_helpers.hpp"       // Complex typedef

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <memory_resource>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace {

// ── downsample / upsample argchecks ───────────────────────────────────
// Probed R2025b: x accepts every numeric class plus logical and complex
// (each preserved in the result), vectors AND matrices (downsampled
// column-wise); N-D is accepted by the reference but the kernel here is
// 2-D (bugs/opened/signal/nd-and-matrix-gaps.md); integer classes would
// need integer-class outputs the Value layer cannot allocate yet
// (bugs/opened/signal/integer-class-outputs.md).
void multirateGatherArgcheck(const Value &x, const char *fn)
{
    const ValueType t = x.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL
        || isIntegerType(t)) {
        return;
    }
    detail::throwInputType(x, fn, "x", 1, "numeric or logical");
}

void multiratePhaseArgcheck(size_t n, size_t phase, const char *fn)
{
    if (phase >= n)
        throw Error(std::string(fn) + ": phase must be an integer in "
                    "[0, n-1] (got " + std::to_string(phase) + " with n = "
                    + std::to_string(n) + ")",
                    0, 0, fn, "", std::string("numkit:") + fn + ":badPhase");
}

// Column count / orientation helpers shared by the typed gathers.
struct GatherShape {
    size_t rows;      // per-column length
    size_t cols;      // product of trailing dimensions
    bool   isRow;     // input was a 2-D row vector → row output
};
GatherShape gatherShape(const Value &x)
{
    if (x.dims().rows() == 1 && x.dims().ndim() == 2)
        return {x.numel(), 1, true};  // row vector (incl. 1×1): row output
    if (x.dims().isVector() || x.isScalar())
        return {x.numel(), 1, false}; // column vector / scalar
    const size_t r = x.dims().rows();
    const size_t c = (r > 0) ? (x.numel() / r) : x.dims().cols();
    return {r, c, false};
}

Value allocMultirateOut(const Value &x, const GatherShape &gs, size_t outRows,
                        std::pmr::memory_resource *mr)
{
    const ValueType t = x.type();
    if (gs.isRow)
        return Value::matrix(1, outRows * gs.cols, t, mr);
    const int nd = x.dims().ndim();
    if (nd <= 2)
        return Value::matrix(outRows, gs.cols, t, mr);
    std::vector<size_t> od(static_cast<size_t>(nd));
    od[0] = outRows;
    for (int d = 1; d < nd; ++d)
        od[static_cast<size_t>(d)] = x.dims().dim(d);
    return Value::matrixND(od.data(), nd, t, mr);
}

} // anonymous namespace

namespace {

// Windowed-sinc lowpass FIR, Hamming window, cutoff wc (radians).
// Normalized so DC gain is 1. Order is filtLen - 1; filtLen must be >= 2.
ScratchVec<double> designLowpassFir(size_t filtLen, double wc, std::pmr::memory_resource *mr)
{
    const size_t filtOrder = filtLen - 1;
    const double half = filtOrder / 2.0;

    ScratchVec<double> h(filtLen, mr);
    double hSum = 0.0;
    for (size_t i = 0; i < filtLen; ++i) {
        const double n = i - half;
        const double sinc = (std::abs(n) < 1e-12)
                                ? wc / M_PI
                                : std::sin(wc * n) / (M_PI * n);
        const double win = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / filtOrder);
        h[i] = sinc * win;
        hSum += h[i];
    }
    for (size_t i = 0; i < filtLen; ++i)
        h[i] /= hSum;
    return h;
}

// Direct Form II transposed FIR apply — matches filter.cpp's core for
// the a = [1] denominator case. Used by decimate and resample.
ScratchVec<double> applyFirDf2t(const double *h, size_t filtLen, const double *x, size_t nx, std::pmr::memory_resource *mr)
{
    ScratchVec<double> out(nx, mr);
    ScratchVec<double> z(filtLen, mr);
    for (size_t n = 0; n < nx; ++n) {
        out[n] = h[0] * x[n] + z[0];
        for (size_t i = 1; i < filtLen; ++i)
            z[i - 1] = h[i] * x[n] + (i < filtLen - 1 ? z[i] : 0.0);
    }
    return out;
}

} // anonymous namespace

// ── downsample ────────────────────────────────────────────────────────
Value downsample(const Value &x, size_t n, size_t phase, std::pmr::memory_resource *mr)
{
    multirateGatherArgcheck(x, "downsample");
    multiratePhaseArgcheck(n, phase, "downsample");
    const GatherShape gs = gatherShape(x);
    const size_t outRows = (phase < gs.rows)
                               ? (gs.rows - phase + n - 1) / n
                               : 0;
    Value r = allocMultirateOut(x, gs, outRows, mr);
    if (outRows == 0 || gs.cols == 0 || x.numel() == 0)
        return r;
    const size_t es = x.rawBytes() / x.numel();
    const char *xd = static_cast<const char *>(x.rawData());
    char *od = static_cast<char *>(r.rawDataMut());
    for (size_t c = 0; c < gs.cols; ++c)
        for (size_t i = 0; i < outRows; ++i)
            std::memcpy(od + (c * outRows + i) * es,
                        xd + (c * gs.rows + phase + i * n) * es,
                        es);
    return r;
}

// ── upsample ──────────────────────────────────────────────────────────
Value upsample(const Value &x, size_t n, size_t phase, std::pmr::memory_resource *mr)
{
    multirateGatherArgcheck(x, "upsample");
    multiratePhaseArgcheck(n, phase, "upsample");
    const GatherShape gs = gatherShape(x);
    const size_t outRows = gs.rows * n;
    Value r = allocMultirateOut(x, gs, outRows, mr);
    if (outRows == 0 || gs.cols == 0 || x.numel() == 0)
        return r;
    const size_t es = x.rawBytes() / x.numel();
    const char *xd = static_cast<const char *>(x.rawData());
    char *od = static_cast<char *>(r.rawDataMut());
    for (size_t c = 0; c < gs.cols; ++c)
        for (size_t i = 0; i < gs.rows; ++i)
            std::memcpy(od + (c * outRows + phase + i * n) * es,
                        xd + (c * gs.rows + i) * es,
                        es);
    return r;
}

// ── decimate ──────────────────────────────────────────────────────────
// Type contract (probed R2025b): x accepts double/single/complex — the
// pipeline here is double-buffer only, so single and complex are
// honest capability gaps; logical and integer classes are rejected by
// the reference outright.
Value decimate(const Value &x, size_t factor, std::pmr::memory_resource *mr)
{
    detail::requireDoubleNotComplexInput(x, "decimate");
    const size_t nx = x.numel();
    const bool isRow = x.dims().rows() == 1;
    if (factor <= 1 || nx == 0) {
        auto r = isRow ? Value::matrix(1, nx, ValueType::DOUBLE, mr)
                       : Value::matrix(nx, 1, ValueType::DOUBLE, mr);
        if (nx > 0)
            std::copy(x.doubleData(), x.doubleData() + nx, r.doubleDataMut());
        return r;
    }

    auto [b, a] = cheby1(8, 0.05, Value::scalar(0.8 / static_cast<double>(factor), mr),
                         FilterType::Lowpass, /*analog=*/false, mr);
    Value filtered = filtfilt(b, a, x, mr);
    const double *fd = filtered.doubleData();

    const size_t outLen = (nx + factor - 1) / factor;
    const size_t nbeg = (nx - 1) % factor;
    auto r = isRow ? Value::matrix(1, outLen, ValueType::DOUBLE, mr)
                   : Value::matrix(outLen, 1, ValueType::DOUBLE, mr);
    double *rd = r.doubleDataMut();
    for (size_t i = 0; i < outLen; ++i)
        rd[i] = fd[nbeg + i * factor];
    return r;
}

// ── resample ──────────────────────────────────────────────────────────
// MATLAB resample.m algorithm (default N = 10, beta = 5): design a
// Kaiser-windowed least-squares anti-alias FIR, apply it via the polyphase
// upfirdn, then compensate the filter group delay and trim to ceil(Lx·p/q)
// samples. Reuses the shipped firls / kaiser / upfirdn — all bit-exact with
// MATLAB, so the assembled output matches MATLAB R2025b exactly.
// Type contract: like decimate — double passes, single and complex are
// capability gaps (the reference accepts both), logical/int rejected.
Value resample(const Value &x, size_t p, size_t q, std::pmr::memory_resource *mr)
{
    detail::requireDoubleNotComplexInput(x, "resample");
    if (p == 0 || q == 0)
        throw Error("resample: P and Q must be positive integers",
                    0, 0, "resample", "", "numkit:resample:pq");

    // Reduce P/Q by their GCD first (as MATLAB does).
    const size_t g = std::gcd(p, q);
    p /= g;
    q /= g;

    const size_t nx = x.numel();
    const bool isRow = x.dims().rows() == 1;

    // Anti-alias filter: fc = 1/(2·max(p,q)); L = 2·N·max(p,q)+1.
    //   h = p · firls(L-1,[0 2fc 2fc 1],[1 1 0 0]) · kaiser(L,beta) / sum(·)
    // (normalised so sum(h) = p — the interpolation gain).
    const int    N    = 10;
    const double beta = 5.0;
    const size_t pqmax = std::max(p, q);
    const double fc = 1.0 / (2.0 * static_cast<double>(pqmax));
    const int    L  = 2 * N * static_cast<int>(pqmax) + 1;

    Value F = Value::matrix(1, 4, ValueType::DOUBLE, mr);
    { double *f = F.doubleDataMut(); f[0] = 0.0; f[1] = 2.0 * fc; f[2] = 2.0 * fc; f[3] = 1.0; }
    Value A = Value::matrix(1, 4, ValueType::DOUBLE, mr);
    { double *a = A.doubleDataMut(); a[0] = 1.0; a[1] = 1.0; a[2] = 0.0; a[3] = 0.0; }

    Value fb = firls(L - 1, F, A, mr);                   // length L
    Value kw = kaiser(static_cast<size_t>(L), beta, mr); // length L
    const double *fbd = fb.doubleData();
    const double *kwd = kw.doubleData();

    ScratchArena scratch(mr);
    auto fk = ScratchVec<double>(static_cast<size_t>(L), &scratch);
    double sumfk = 0.0;
    for (int i = 0; i < L; ++i) { fk[i] = fbd[i] * kwd[i]; sumfk += fk[i]; }

    // Front-pad the filter so the group delay lands on an output sample.
    const long long Lhalf  = (L - 1) / 2;
    const long long nz     = static_cast<long long>(q) - (Lhalf % static_cast<long long>(q));
    const long long Lhalf2 = Lhalf + nz;

    Value hh = Value::matrix(1, static_cast<size_t>(nz) + static_cast<size_t>(L),
                             ValueType::DOUBLE, mr);
    {
        double *hd = hh.doubleDataMut();
        for (long long i = 0; i < nz; ++i) hd[i] = 0.0;
        const double scale = static_cast<double>(p) / sumfk;
        for (int i = 0; i < L; ++i) hd[nz + i] = scale * fk[i];
    }

    Value yfull = upfirdn(x, hh, p, q, mr);
    const size_t  yfn = yfull.numel();
    const double *yfd = yfull.doubleData();

    const long long delay = Lhalf2 / static_cast<long long>(q);     // floor(ceil(Lhalf2)/q)
    const long long Ly    = static_cast<long long>((nx * p + q - 1) / q); // ceil(Lx·p/q)
    const size_t    outLen = (Ly > 0) ? static_cast<size_t>(Ly) : 0;

    Value r = isRow ? Value::matrix(1, outLen, ValueType::DOUBLE, mr)
                    : Value::matrix(outLen, 1, ValueType::DOUBLE, mr);
    double *rd = r.doubleDataMut();
    for (size_t i = 0; i < outLen; ++i) {
        const long long idx = delay + static_cast<long long>(i);
        rd[i] = (idx >= 0 && static_cast<size_t>(idx) < yfn) ? yfd[idx] : 0.0;
    }
    return r;
}

} // namespace numkit::signal
