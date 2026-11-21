// toolboxes/signal/src/digital_filtering/quantizers.cpp
//
// uencode / udecode — uniform quantization (Phase 4.2 of audio sweep).
// Bit-equal MATLAB R2025b uencode.m / udecode.m.
//
// uencode formula (FloatTo32Real):
//   Q = 2^N - 1
//   T = (Q+1) / (2*V)
//   x = (u + V) * T
//   if unsigned: clamp to [0, Q]; uint(N) = floor(x)
//   if signed:   shift x += SignMin (=-2^(N-1)); clamp [-2^(N-1), 2^(N-1)-1];
//                int(N) = floor(x_shifted)
//
// udecode formula (DecodeReal):
//   W = 2^(N-1) for signed, 0 for unsigned
//   T = V * 2^(1-N)
//   if saturate: clamp u to [lowerBnd, upperBnd]
//   if wrap:     u = mod((u - lowerBnd), 2^N) + lowerBnd  (signed)
//                u = mod(u, 2^N)                           (unsigned)
//   y = (u + W) * T - V
//
// PMR HARD RULE.

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/signal/digital_filtering/quantizers.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace numkit::signal {

namespace {

ValueType pickUnsignedType(int N)
{
    if (N <= 8)  return ValueType::UINT8;
    if (N <= 16) return ValueType::UINT16;
    return ValueType::UINT32;
}

ValueType pickSignedType(int N)
{
    if (N <= 8)  return ValueType::INT8;
    if (N <= 16) return ValueType::INT16;
    return ValueType::INT32;
}

// Allocate output Value of the requested integer type with same shape as u.
Value allocLikeIntType(const Value &u, ValueType vt, keel::Allocator *mr)
{
    if (u.dims().is3D())
        return Value::matrix3d(u.dims().rows(), u.dims().cols(),
                                u.dims().pages(), vt, mr);
    return Value::matrix(u.dims().rows(), u.dims().cols(), vt, mr);
}

// Floor-and-clip helper: returns floor(x), clamped to [lo, hi].
inline double floorClip(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return std::floor(x);
}

} // anon

Value uencode(const Value &u, int N, double V, bool signedOutput, keel::Allocator *mr)
{
    if (N < 2 || N > 32)
        throw Error("uencode: N must be in [2, 32]",
                    0, 0, "uencode", "", "numkit:uencode:BadN");
    if (V <= 0.0)
        throw Error("uencode: V must be positive",
                    0, 0, "uencode", "", "numkit:uencode:NeedPosRealScalar");

    const double Q = std::pow(2.0, N) - 1.0;
    const double T = (Q + 1.0) / (2.0 * V);
    const double signMax = std::pow(2.0, N - 1) - 1.0;
    const double signMin = -(1.0 + signMax);

    const ValueType vt = signedOutput ? pickSignedType(N) : pickUnsignedType(N);
    Value out = allocLikeIntType(u, vt, mr);
    const keel::usize M = u.numel();
    if (M == 0) return out;

    // Helper to write one element into the typed buffer.
    auto writeElem = [&](keel::usize i, double v) {
        switch (vt) {
            case ValueType::UINT8:  out.uint8DataMut()[i]  = keel::as<keel::u8>(v); break;
            case ValueType::UINT16: out.uint16DataMut()[i] = keel::as<keel::u16>(v); break;
            case ValueType::UINT32: out.uint32DataMut()[i] = keel::as<keel::u32>(v); break;
            case ValueType::INT8:   out.int8DataMut()[i]   = keel::as<keel::i8>(v); break;
            case ValueType::INT16:  out.int16DataMut()[i]  = keel::as<keel::i16>(v); break;
            case ValueType::INT32:  out.int32DataMut()[i]  = keel::as<keel::i32>(v); break;
            default: break;
        }
    };

    for (keel::usize i = 0; i < M; ++i) {
        const double ui = u.elemAsDouble(i);
        double scaled = (ui + V) * T;
        if (signedOutput) {
            double shifted = scaled + signMin;
            shifted = floorClip(shifted, signMin, signMax);
            writeElem(i, shifted);
        } else {
            const double clipped = floorClip(scaled, 0.0, Q);
            writeElem(i, clipped);
        }
    }
    return out;
}

// udecode domain checks (all content validation lives here so the
// interpreter and direct-C++ calls cross the same contract; the reg only
// marshals). Rejection classes probed vs R2025b.
static void udecode_argcheck(const Value &u, int N, double V)
{
    if (N < 2 || N > 32)
        throw Error("udecode: N must be in [2, 32]",
                    0, 0, "udecode", "", "numkit:udecode:BadN");
    if (V <= 0.0)
        throw Error("udecode: V must be positive",
                    0, 0, "udecode", "", "numkit:udecode:NeedPosRealScalar");

    // Detect signed vs unsigned from input type.
    const ValueType ut = u.type();
    const bool isSigned = (ut == ValueType::INT8 || ut == ValueType::INT16
                            || ut == ValueType::INT32);
    const bool isUnsigned = (ut == ValueType::UINT8 || ut == ValueType::UINT16
                              || ut == ValueType::UINT32);
    if (!isSigned && !isUnsigned)
        throw Error("udecode: input must be int8/16/32 or uint8/16/32",
                    0, 0, "udecode", "", "numkit:udecode:BadInputType");
}

Value udecode(const Value &u, int N, double V, bool wrapOnOverflow, keel::Allocator *mr)
{
    udecode_argcheck(u, N, V);
    const bool isSigned = (u.type() == ValueType::INT8 || u.type() == ValueType::INT16
                           || u.type() == ValueType::INT32);

    const double W = isSigned ? std::pow(2.0, N - 1) : 0.0;
    const double T = V * std::pow(2.0, 1.0 - N);
    const double upper = isSigned ? (std::pow(2.0, N - 1) - 1.0)
                                    : (std::pow(2.0, N) - 1.0);
    const double lower = isSigned ? -std::pow(2.0, N - 1) : 0.0;
    const double mod2N = std::pow(2.0, N);

    Value out = (u.dims().is3D())
                ? Value::matrix3d(u.dims().rows(), u.dims().cols(),
                                   u.dims().pages(), ValueType::DOUBLE, mr)
                : Value::matrix(u.dims().rows(), u.dims().cols(),
                                 ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    const keel::usize M = u.numel();
    for (keel::usize i = 0; i < M; ++i) {
        double v = u.elemAsDouble(i);  // promotes int → double
        if (wrapOnOverflow) {
            // mod((v - lower), 2^N) + lower (matches MATLAB sign convention)
            const double shifted = v - lower;
            double m = std::fmod(shifted, mod2N);
            if (m < 0.0) m += mod2N;
            v = m + lower;
        } else {
            // saturate
            if (v > upper) v = upper;
            if (v < lower) v = lower;
        }
        od[i] = (v + W) * T - V;
    }
    return out;
}

} // namespace numkit::signal
