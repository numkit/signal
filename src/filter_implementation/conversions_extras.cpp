// toolboxes/signal/src/filter_implementation/conversions_extras.cpp
// sos2tf / sos2zp / tf2zpk + tf<->ss + sos<->ss + zpk<->ss adapters
// (delegates numeric kernels to keel::signal).

#include <keel/algo.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/filter_implementation/conversions.hpp>
#include <numkit/signal/filter_implementation/conversions_extras.hpp>

#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

Value rowFromVec(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    auto out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

Value complexColFromVec(keel::ArrayRef<keel::c64> v, keel::Allocator *mr)
{
    bool anyImag = false;
    for (const auto &c : v)
        if (keel::abs(c.imag()) > 1e-12) { anyImag = true; break; }
    if (anyImag) {
        auto out = Value::complexMatrix(v.size(), 1, mr);
        keel::c64 *dst = out.complexDataMut();
        for (keel::usize i = 0; i < v.size(); ++i) dst[i] = v[i];
        return out;
    }
    auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < v.size(); ++i) dst[i] = v[i].real();
    return out;
}

void requireSosShape(const Value &sos, const char *fn)
{
    if (sos.dims().cols() != 6 || sos.dims().rows() == 0)
        throw Error(keel::format("{}: sos must be an Lx6 matrix", fn),
                     0, 0, fn, "", keel::format("numkit:{}:badShape", fn));
}

keel::Array<keel::f64> readColMajorReal(const Value &m, keel::Allocator *mr)
{
    const keel::usize n = m.numel();
    keel::Array<keel::f64> out(n, mr);
    for (keel::usize i = 0; i < n; ++i) out[i] = m.elemAsDouble(i);
    return out;
}

void sos2tf_argcheck(const Value &sos) { requireSosShape(sos, "sos2tf"); }
void sos2zp_argcheck(const Value &sos) { requireSosShape(sos, "sos2zp"); }

inline keel::c64 sfs_elemAsComplex(const Value &v, keel::usize i)
{
    if (v.type() == ValueType::COMPLEX) return v.complexData()[i];
    return keel::c64(v.elemAsDouble(i), 0.0);
}

inline keel::c64 sfs_sign(keel::c64 x)
{
    if (x.imag() == 0.0) {
        const keel::f64 r = x.real();
        if (r > 0.0) return keel::c64(1.0, 0.0);
        if (r < 0.0) return keel::c64(-1.0, 0.0);
        return keel::c64(0.0, 0.0);
    }
    const keel::f64 mag = keel::abs(x);
    if (mag == 0.0) return keel::c64(0.0, 0.0);
    return x / mag;
}

} // namespace

keel::Tuple<Value, Value>
sos2tf(const Value &sos, keel::f64 g, keel::Allocator *mr)
{
    sos2tf_argcheck(sos);
    const keel::usize L = sos.dims().rows();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto sBuf = readColMajorReal(sos, scratch.allocator());
    keel::signal::TfResult tf;
    (void)keel::signal::sos2tf(sBuf.as_ref(), L, g, tf, scratch.allocator());
    return keel::make_tuple(rowFromVec(tf.b.as_ref(), mr), rowFromVec(tf.a.as_ref(), mr));
}

keel::Tuple<Value, Value, keel::f64>
sos2zp(const Value &sos, keel::f64 g, keel::Allocator *mr)
{
    sos2zp_argcheck(sos);
    const keel::usize L = sos.dims().rows();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto sBuf = readColMajorReal(sos, scratch.allocator());
    keel::signal::ZpkResult zpk;
    (void)keel::signal::sos2zp(sBuf.as_ref(), L, g, zpk, scratch.allocator());
    return keel::make_tuple(complexColFromVec(zpk.z.as_ref(), mr),
                            complexColFromVec(zpk.p.as_ref(), mr),
                            zpk.k);
}

keel::Tuple<Value, Value, keel::f64>
tf2zpk(const Value &b, const Value &a, keel::Allocator *mr)
{
    auto [z, p, k] = numkit::builtin::tf2zp(b, a, mr);
    return keel::make_tuple(keel::move(z), keel::move(p),
                            (k.numel() == 0) ? 0.0 : k.toScalar());
}

keel::Tuple<Value, Value, Value, Value>
tf2ss(const Value &b, const Value &a, keel::Allocator *mr)
{
    if (a.numel() == 0)
        throw Error("tf2ss: denominator a must be non-empty",
                     0, 0, "tf2ss", "", "numkit:tf2ss:badArg");
    if (keel::abs(a.elemAsDouble(0)) < 1e-15)
        throw Error("tf2ss: a(1) must be non-zero",
                     0, 0, "tf2ss", "", "numkit:tf2ss:badArg");
    if (a.numel() < 2)
        throw Error("tf2ss: order must be >= 1 (denominator length >= 2)",
                     0, 0, "tf2ss", "", "numkit:tf2ss:badArg");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readColMajorReal(b, scratch.allocator());
    auto av = readColMajorReal(a, scratch.allocator());
    keel::signal::StateSpaceResult ss;
    (void)keel::signal::tf2ss(bv.as_ref(), av.as_ref(), ss, scratch.allocator());

    const keel::usize N = ss.order;
    auto AVal = Value::matrix(N, N, ValueType::DOUBLE, mr);
    auto BVal = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    auto CVal = Value::matrix(1, N, ValueType::DOUBLE, mr);
    if (N > 0) {
        keel::copy_n(AVal.doubleDataMut(), ss.a_col_major.data(), N * N);
        keel::copy_n(BVal.doubleDataMut(), ss.b.data(), N);
        keel::copy_n(CVal.doubleDataMut(), ss.c.data(), N);
    }
    return keel::make_tuple(keel::move(AVal), keel::move(BVal),
                            keel::move(CVal), Value::scalar(ss.d, mr));
}

keel::Tuple<Value, Value>
ss2tf(const Value &A, const Value &B, const Value &C, keel::f64 D, keel::Allocator *mr)
{
    if (A.dims().rows() != A.dims().cols())
        throw Error("ss2tf: A must be square",
                     0, 0, "ss2tf", "", "numkit:ss2tf:badShape");
    const keel::usize N = A.dims().rows();
    if (B.dims().rows() != N || B.dims().cols() != 1)
        throw Error("ss2tf: B must be Nx1",
                     0, 0, "ss2tf", "", "numkit:ss2tf:badShape");
    if (C.dims().rows() != 1 || C.dims().cols() != N)
        throw Error("ss2tf: C must be 1xN",
                     0, 0, "ss2tf", "", "numkit:ss2tf:badShape");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto Av = readColMajorReal(A, scratch.allocator());
    auto Bv = readColMajorReal(B, scratch.allocator());
    auto Cv = readColMajorReal(C, scratch.allocator());
    keel::signal::TfResult tf;
    (void)keel::signal::ss2tf(Av.as_ref(), N, Bv.as_ref(), Cv.as_ref(), D, tf, scratch.allocator());
    return keel::make_tuple(rowFromVec(tf.b.as_ref(), mr), rowFromVec(tf.a.as_ref(), mr));
}

keel::Tuple<Value, Value, keel::f64>
ss2zp(const Value &A, const Value &B, const Value &C, keel::f64 D, keel::Allocator *mr)
{
    auto [b, a] = ss2tf(A, B, C, D, mr);
    return tf2zpk(b, a, mr);
}

keel::Tuple<Value, Value, Value, Value>
zp2ss(const Value &z, const Value &p, keel::f64 k, keel::Allocator *mr)
{
    auto [b, a] = numkit::builtin::zp2tf(z, p, k, mr);
    return tf2ss(b, a, mr);
}

keel::Tuple<Value, Value, Value, Value>
sos2ss(const Value &sos, keel::f64 g, keel::Allocator *mr)
{
    auto [b, a] = sos2tf(sos, g, mr);
    return tf2ss(b, a, mr);
}

Value ss2sos(const Value &A, const Value &B, const Value &C, keel::f64 D, keel::Allocator *mr)
{
    auto [b, a] = ss2tf(A, B, C, D, mr);
    return tf2sos(b, a, mr);
}

keel::Tuple<Value, Value, keel::f64>
ctf2zp(const Value &NUM, const Value &DEN, const Value &SV, keel::Allocator *mr)
{
    auto rowsOf = [](const Value &v) -> keel::usize {
        return (v.dims().rows() == 1 && v.dims().cols() > 1) ? 1 : v.dims().rows();
    };
    auto colsOf = [](const Value &v) -> keel::usize {
        return v.dims().cols();
    };
    const keel::usize Knum = rowsOf(NUM);
    const keel::usize Kden = rowsOf(DEN);
    const keel::usize K = keel::max(Knum, Kden);
    if (K == 0)
        throw Error("ctf2zp: NUM/DEN must be non-empty",
                    0, 0, "ctf2zp", "", "numkit:ctf2zp:Empty");

    keel::Array<keel::f64> sv(mr);
    if (SV.isEmpty()) {
        sv.assign(K + 1, 1.0);
    } else if (SV.numel() == 1) {
        sv.assign(K + 1, 1.0);
        sv[0] = SV.toScalar();
    } else if (SV.numel() == K + 1) {
        sv.resize(K + 1);
        for (keel::usize i = 0; i <= K; ++i) sv[i] = SV.elemAsDouble(i);
    } else {
        throw Error("ctf2zp: SV must be scalar or K+1 vector",
                    0, 0, "ctf2zp", "", "numkit:ctf2zp:invalidSVDims");
    }

    auto extractRow = [&](const Value &M, keel::usize i, keel::usize Krows) {
        const keel::usize cols = colsOf(M);
        Value row = Value::matrix(1, cols, ValueType::DOUBLE, mr);
        keel::f64 *rd = row.doubleDataMut();
        if (M.dims().rows() == 1 && M.dims().cols() > 1) {
            for (keel::usize k = 0; k < cols; ++k) rd[k] = M.elemAsDouble(k);
        } else if (Krows == 1 && M.numel() == 1) {
            rd[0] = M.toScalar();
        } else {
            const keel::usize rows = M.dims().rows();
            for (keel::usize k = 0; k < cols; ++k) rd[k] = M.elemAsDouble(i + k * rows);
        }
        return row;
    };

    keel::Array<keel::c64> zeros(mr), poles(mr);
    keel::f64 gainAccum = 1.0;
    for (keel::usize i = 0; i < K; ++i) {
        Value brow = extractRow(NUM, keel::min(i, Knum - 1), Knum);
        Value arow = extractRow(DEN, keel::min(i, Kden - 1), Kden);
        auto [zsec, psec, gsec] = tf2zpk(brow, arow, mr);
        const keel::usize nz = zsec.numel(), np = psec.numel();
        if (zsec.type() == ValueType::COMPLEX) {
            const keel::c64 *zd = zsec.complexData();
            for (keel::usize k = 0; k < nz; ++k) zeros.push_back(zd[k]);
        } else {
            for (keel::usize k = 0; k < nz; ++k) zeros.push_back(keel::c64(zsec.elemAsDouble(k), 0.0));
        }
        if (psec.type() == ValueType::COMPLEX) {
            const keel::c64 *pd = psec.complexData();
            for (keel::usize k = 0; k < np; ++k) poles.push_back(pd[k]);
        } else {
            for (keel::usize k = 0; k < np; ++k) poles.push_back(keel::c64(psec.elemAsDouble(k), 0.0));
        }
        gainAccum *= gsec;
    }
    for (keel::f64 v : sv) gainAccum *= v;

    Value Z = Value::matrix(zeros.size(), zeros.empty() ? 0 : 1, ValueType::COMPLEX, mr);
    Value P = Value::matrix(poles.size(), poles.empty() ? 0 : 1, ValueType::COMPLEX, mr);
    if (!zeros.empty()) keel::copy(zeros.begin(), zeros.end(), Z.complexDataMut());
    if (!poles.empty()) keel::copy(poles.begin(), poles.end(), P.complexDataMut());
    return {Z, P, gainAccum};
}

Value scaleFilterSections(const Value &CTFNum, const Value &SV, keel::Allocator *mr)
{
    const Dims &bd = CTFNum.dims();
    const keel::usize K = bd.rows();
    const keel::usize Q = bd.cols();
    const keel::usize nsv = SV.numel();
    const bool scalarG = (nsv == 1);
    const bool vectorG = (nsv == K + 1);
    if (!scalarG && !vectorG) {
        throw numkit::Error(
            "Invalid number of scale values. Specify either a scalar "
            "or a vector of length equal to the number of sections + 1.",
            0, 0, "scaleFilterSections", "",
            "numkit:scaleFilterSections:invalidNumberOfScaleValues");
    }

    const bool resultComplex = (CTFNum.type() == ValueType::COMPLEX)
                            || (SV.type() == ValueType::COMPLEX);
    Value out = Value::matrix(K, Q, resultComplex ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    const keel::c64 overall = scalarG ? sfs_elemAsComplex(SV, 0) : sfs_elemAsComplex(SV, K);
    const keel::f64 root = (K == 0) ? 0.0 : keel::pow(keel::abs(overall), 1.0 / keel::as<keel::f64>(K));
    const keel::c64 sgn = sfs_sign(overall);

    if (resultComplex) {
        keel::c64 *od = out.complexDataMut();
        for (keel::usize k = 0; k < K; ++k) {
            keel::c64 factor = vectorG ? root * sfs_elemAsComplex(SV, k) : keel::c64(root, 0.0);
            if (k + 1 == K) factor *= sgn;
            for (keel::usize j = 0; j < Q; ++j) {
                const keel::usize idx = k + j * K;
                od[idx] = factor * sfs_elemAsComplex(CTFNum, idx);
            }
        }
    } else {
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize k = 0; k < K; ++k) {
            keel::f64 factor = vectorG ? root * SV.elemAsDouble(k) : root;
            if (k + 1 == K) factor *= sgn.real();
            for (keel::usize j = 0; j < Q; ++j) {
                const keel::usize idx = k + j * K;
                od[idx] = factor * CTFNum.elemAsDouble(idx);
            }
        }
    }
    return out;
}

} // namespace numkit::signal
