// toolboxes/signal/src/transforms/goertzel.cpp
//
// Goertzel single-bin DFT. Split from toolboxes/signal/src/.

#include <nk/signal/transforms/goertzel.hpp>

#include <nk/value/value.hpp>
#include <nk/value/error.hpp>
#include <nk/value/value_type.hpp>   // mtypeName (argcheck got-clauses)

#include <nk/value/scratch.hpp>

#include "../dsp_helpers.hpp"   // Complex typedef
#include "../signal_detail.hpp" // shared argcheck cores

#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nk::signal {

// Goertzel computes a single DFT bin via a 2nd-order IIR. For each
// 1-based bin index k in `ind` (1 == DC, 2 == lowest non-DC, ..., N
// == highest), output[k] = sum_n x[n] * exp(-2πi (k-1) n / N).
//
// Two-coefficient form. Numerically stable for reasonable N; matches
// MATLAB's `goertzel(x, ind)` to FP roundoff. x may be real or complex
// (the recurrence runs on complex state directly); ind accepts every
// numeric class and may have any shape — the result matches ind's
// shape, exactly as the reference (probed R2025b: fractional and
// out-of-range ind values are NOT validated by the reference either).
Value goertzel(const Value &x, const Value &ind, std::pmr::memory_resource *mr)
{
    // Contract (probed R2025b): x is single/double/complex (single is a
    // capability gap here); logical/integer are rejected. ind is
    // unvalidated by the reference — accept every numeric class.
    detail::requireFloatInput(x, "goertzel");
    detail::requireNumericOrLogicalInput(ind, "goertzel", "ind", 2);

    const size_t N = x.numel();
    const size_t M = ind.numel();
    auto r = Value::complexMatrix(ind.dims().rows(), ind.dims().cols(), mr);

    if (N == 0 || M == 0) return r;

    const bool xComplex = x.isComplex();
    ScratchArena scratch(mr);
    ScratchVec<Complex> xc(xComplex ? N : 0, &scratch);
    if (xComplex) {
        const Complex *xd = x.complexData();
        for (size_t i = 0; i < N; ++i) xc[i] = xd[i];
    }
    ScratchVec<double> id(M, &scratch);
    for (size_t j = 0; j < M; ++j) id[j] = ind.elemAsDouble(j);

    Complex *od = r.complexDataMut();
    for (size_t m = 0; m < M; ++m) {
        const double k1based = id[m];
        const double k0based = k1based - 1.0;
        const double w = 2.0 * M_PI * k0based / static_cast<double>(N);
        const double cw = std::cos(w);
        const double sw = std::sin(w);
        const double coeff = 2.0 * cw;

        if (xComplex) {
            Complex s_prev(0.0, 0.0), s_prev2(0.0, 0.0);
            for (size_t n = 0; n < N; ++n) {
                const Complex s = xc[n] + coeff * s_prev - s_prev2;
                s_prev2 = s_prev;
                s_prev  = s;
            }
            const Complex y = s_prev - Complex(cw, -sw) * s_prev2;
            const double pcw = std::cos(w * static_cast<double>(N - 1));
            const double psw = std::sin(w * static_cast<double>(N - 1));
            od[m] = y * Complex(pcw, -psw);
        } else {
            const double *xd = x.doubleData();
            double s_prev = 0.0, s_prev2 = 0.0;
            for (size_t n = 0; n < N; ++n) {
                const double s = xd[n] + coeff * s_prev - s_prev2;
                s_prev2 = s_prev;
                s_prev  = s;
            }
            // Recursive output before phase correction:
            //   y = s_prev - exp(-jw) * s_prev2
            //     = (s_prev - cos(w)*s_prev2) + j*sin(w)*s_prev2
            // True DFT bin X[k] = y * exp(-jw*(N-1)).
            const double y_re = s_prev - cw * s_prev2;
            const double y_im = sw * s_prev2;
            const double pcw = std::cos(w * static_cast<double>(N - 1));
            const double psw = std::sin(w * static_cast<double>(N - 1));
            const double out_re = y_re * pcw + y_im * psw;
            const double out_im = y_im * pcw - y_re * psw;
            od[m] = Complex(out_re, out_im);
        }
    }
    return r;
}

} // namespace nk::signal
