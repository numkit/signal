// toolboxes/signal/src/spectral_analysis/pseudospectrum.cpp
//
// Subspace pseudospectrum estimators: pmusic (MUSIC) and peig (eigenvector).
// Both build the signal correlation matrix R = X'·X via corrmtx, eigendecompose
// it (classical Jacobi — toolboxes carry their own small symmetric eigensolver,
// as in stats/pca), split into signal (top p) / noise (bottom M-p) subspaces,
// then evaluate
//   P(w) = 1 / Σ_{noise} |e(w)'·v_k|²            (pmusic)
//   P(w) = 1 / Σ_{noise} |e(w)'·v_k|² / λ_k      (peig)
// over a one-sided frequency grid. Matches MATLAB on the PEAK FREQUENCIES (this
// is a frequency estimator); the absolute pseudospectrum is scale-arbitrary and
// eigendecomposition-sensitive (peaks are 1/near-zero), so it is NOT bit-matched.
// bugs/signal/pmusic-peig.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>   // corrmtx, pmusic, peig

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace {

// Classical Jacobi eigendecomposition of a D×D symmetric matrix A (row-major).
// On exit A's diagonal holds the eigenvalues; V's columns are the eigenvectors.
void pseudospectrum_argcheck(const Value &x, keel::i32 p, keel::i32 nfft, keel::f64 fs)
{
    (void)x;
    (void)p;
    (void)nfft;
    (void)fs;
}

void jacobiEig(keel::Array<keel::f64> &A, keel::Array<keel::f64> &V, keel::usize D)
{
    V.assign(D * D, 0.0);
    for (keel::usize i = 0; i < D; ++i) V[i * D + i] = 1.0;
    for (keel::i32 sweep = 0; sweep < 60; ++sweep) {
        keel::f64 off = 0.0;
        for (keel::usize p = 0; p < D; ++p)
            for (keel::usize q = p + 1; q < D; ++q) off += A[p * D + q] * A[p * D + q];
        if (off < 1e-18) break;
        for (keel::usize p = 0; p + 1 < D; ++p)
            for (keel::usize q = p + 1; q < D; ++q) {
                const keel::f64 apq = A[p * D + q];
                if (std::fabs(apq) < 1e-300) continue;
                const keel::f64 app = A[p * D + p], aqq = A[q * D + q];
                const keel::f64 theta = (aqq - app) / (2.0 * apq);
                const keel::f64 sgn   = (theta >= 0) ? 1.0 : -1.0;
                const keel::f64 t     = (theta == 0.0) ? 1.0
                                      : sgn / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const keel::f64 c = 1.0 / std::sqrt(t * t + 1.0);
                const keel::f64 s = t * c;
                A[p * D + p] = app - t * apq;
                A[q * D + q] = aqq + t * apq;
                A[p * D + q] = 0.0;
                A[q * D + p] = 0.0;
                for (keel::usize r = 0; r < D; ++r) {
                    if (r == p || r == q) continue;
                    const keel::f64 arp = A[r * D + p], arq = A[r * D + q];
                    A[r * D + p] = c * arp - s * arq;
                    A[p * D + r] = A[r * D + p];
                    A[r * D + q] = c * arq + s * arp;
                    A[q * D + r] = A[r * D + q];
                }
                for (keel::usize r = 0; r < D; ++r) {
                    const keel::f64 vrp = V[r * D + p], vrq = V[r * D + q];
                    V[r * D + p] = c * vrp - s * vrq;
                    V[r * D + q] = c * vrq + s * vrp;
                }
            }
    }
}

std::tuple<Value, Value>
pseudospectrum(const Value &x, keel::i32 p, keel::i32 nfft, keel::f64 fs, bool eigWeight,
               keel::Allocator *mr)
{
    pseudospectrum_argcheck(x, p, nfft, fs);
    if (p < 1) p = 1;
    if (nfft < 2) nfft = 256;
    const keel::i32 M = 2 * p;                       // correlation-matrix order (MATLAB default)

    Value         Xc   = corrmtx(x, M - 1, mr);   // (N+M-1) × M data matrix; X'X = R
    const keel::i32     rows = keel::as<keel::i32>(Xc.dims().rows());
    const keel::f64 *Xd   = Xc.doubleData();         // column-major

    keel::Array<keel::f64> R(keel::as<keel::usize>(M) * M, 0.0, mr);
    for (keel::i32 i = 0; i < M; ++i)
        for (keel::i32 j = 0; j < M; ++j) {
            keel::f64 s = 0.0;
            for (keel::i32 r = 0; r < rows; ++r) s += Xd[i * rows + r] * Xd[j * rows + r];
            R[i * M + j] = s;
        }

    keel::Array<keel::f64> V(mr);
    jacobiEig(R, V, keel::as<keel::usize>(M));
    keel::Array<keel::f64> lam(M, mr);
    for (keel::i32 i = 0; i < M; ++i) lam[i] = R[i * M + i];

    // Ascending eigenvalue order; noise subspace = the smallest (M-p) vectors.
    keel::Array<keel::i32> ord(M, mr);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](keel::i32 a, keel::i32 b) { return lam[a] < lam[b]; });
    const keel::i32 nNoise = M - p;

    const keel::i32 nf = nfft / 2 + 1;
    Value     P  = Value::matrix(nf, 1, ValueType::DOUBLE, mr);
    Value     F  = Value::matrix(nf, 1, ValueType::DOUBLE, mr);
    keel::f64 *  Pd = P.doubleDataMut();
    keel::f64 *  Fd = F.doubleDataMut();
    for (keel::i32 k = 0; k < nf; ++k) {
        const keel::f64 w   = M_PI * k / (nf - 1);   // normalized angular freq [0, π]
        keel::f64       sum = 0.0;
        for (keel::i32 t = 0; t < nNoise; ++t) {
            const keel::i32 col = ord[t];
            keel::f64    re = 0.0, im = 0.0;
            for (keel::i32 m = 0; m < M; ++m) {
                const keel::f64 v = V[keel::as<keel::usize>(m) * M + col];
                re += v * std::cos(w * m);
                im += v * std::sin(w * m);
            }
            keel::f64 proj = re * re + im * im;
            if (eigWeight) proj /= (lam[col] > 1e-300 ? lam[col] : 1e-300);
            sum += proj;
        }
        Pd[k] = (sum > 0.0) ? 1.0 / sum : std::numeric_limits<keel::f64>::infinity();
        Fd[k] = (fs / 2.0) * k / (nf - 1);
    }
    return std::make_tuple(std::move(P), std::move(F));
}

} // namespace

std::tuple<Value, Value>
pmusic(const Value &x, keel::i32 p, keel::i32 nfft, keel::f64 fs, keel::Allocator *mr)
{
    return pseudospectrum(x, p, nfft, fs, /*eigWeight=*/false, mr);
}

std::tuple<Value, Value>
peig(const Value &x, keel::i32 p, keel::i32 nfft, keel::f64 fs, keel::Allocator *mr)
{
    return pseudospectrum(x, p, nfft, fs, /*eigWeight=*/true, mr);
}

} // namespace numkit::signal
