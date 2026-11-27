// modules/signal/src/register/dsp_system_objects.cpp
//
// Stateful DSP System objects: dsp.IIRFilter, dsp.FIRInterpolator, dsp.ArrayPlot.

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>

#include <numkit/sys/plugin.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/object.hpp>
#include <numkit/value/value.hpp>

namespace numkit::signal::detail {

namespace {

keel::Allocator *allocOf(const CallContext &ctx) noexcept
{
    return ctx.engine ? ctx.engine->allocator() : keel::default_allocator();
}

keel::Array<keel::f64> toRealVec(const Value &v, keel::Allocator *mr)
{
    keel::Array<keel::f64> out(mr);
    out.reserve(v.numel());
    for (keel::usize i = 0; i < v.numel(); ++i)
        out.push_back(v.elemAsDouble(i));
    return out;
}

Value fromRealVecRow(const keel::Array<keel::f64> &vec, keel::Allocator *mr)
{
    Value v = Value::matrix(1, vec.size(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < vec.size(); ++i)
        v.doubleDataMut()[i] = vec[i];
    return v;
}

// ── dsp.IIRFilter ───────────────────────────────────────────────────────
struct IirFilterPayload final : public NativePayload {
    keel::String structure;
    keel::Array<keel::f64> num;
    keel::Array<keel::f64> den;
    // Direct form II transposed delay states per channel: [ch][tap]
    keel::Array<keel::Array<keel::c64>> states;
    bool isComplexState = false;

    explicit IirFilterPayload(keel::Allocator *mr = nullptr)
        : structure(keel::String::from("Direct form II transposed"))
        , num(mr)
        , den(mr)
        , states(mr)
    {
        num.push_back(0.5);
        num.push_back(0.5);
        den.push_back(1.0);
        den.push_back(0.5);
    }

    void resetStates()
    {
        states.clear();
        isComplexState = false;
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<IirFilterPayload>(a, a);
        p->structure = structure.clone();
        p->num = num.clone(a);
        p->den = den.clone(a);
        p->isComplexState = isComplexState;
        p->states.reserve(states.size());
        for (keel::usize i = 0; i < states.size(); ++i)
            p->states.push_back(states[i].clone(a));
        return p;
    }
};

void iirApplyNV(IirFilterPayload &st, keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    keel::usize i = 0;
    if (args.size() >= 1 && !args[0].isChar() && !args[0].isString()) {
        st.num = toRealVec(args[0], mr);
        i = 1;
        if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()) {
            st.den = toRealVec(args[1], mr);
            i = 2;
        }
    }
    for (; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString())
            continue;
        const keel::String key = args[i].toString();
        if (key.equals_ignore_ascii_case("Structure")) {
            st.structure = args[i + 1].toString();
        } else if (key.equals_ignore_ascii_case("Numerator")) {
            st.num = toRealVec(args[i + 1], mr);
            st.resetStates();
        } else if (key.equals_ignore_ascii_case("Denominator")) {
            st.den = toRealVec(args[i + 1], mr);
            st.resetStates();
        }
    }
}

void iirStepImpl(Value &self, keel::ArrayRef<Value> args,
                 keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "dsp.IIRFilter.step");
    auto *mr = allocOf(ctx);
    ObjectState *os = self.objectStateMut();
    if (!os || !os->native)
        throw Error("dsp.IIRFilter: invalid object state.", 0, 0, "step", "", "numkit:dsp:InvalidObject");
    auto &p = *static_cast<IirFilterPayload *>(os->native.get());

    const Value &x = args[0];
    if (x.isEmpty()) {
        outs[0] = Value::matrix(x.dims().rows(), x.dims().cols(), x.type(), mr);
        return;
    }
    const keel::usize nRows = x.dims().rows();
    const keel::usize nCols = x.dims().cols();
    const keel::usize nb = p.num.size();
    const keel::usize na = p.den.size();
    if (nb == 0 || na == 0 || p.den[0] == 0.0)
        throw Error("dsp.IIRFilter: Numerator and Denominator must be non-empty with Denominator(1) ~= 0.",
                    0, 0, "step", "", "numkit:dsp:InvalidCoefficients");

    const keel::f64 a0 = p.den[0];
    const keel::usize nOrder = (nb > na ? nb : na);
    const keel::usize nState = (nOrder > 0) ? (nOrder - 1) : 0;

    if (p.states.size() != nCols) {
        p.states.clear();
        p.states.reserve(nCols);
        for (keel::usize ch = 0; ch < nCols; ++ch) {
            keel::Array<keel::c64> s(nState, keel::c64{0.0, 0.0}, mr);
            p.states.push_back(keel::move(s));
        }
    } else {
        for (keel::usize ch = 0; ch < nCols; ++ch) {
            if (p.states[ch].size() != nState)
                p.states[ch].resize(nState, keel::c64{0.0, 0.0});
        }
    }

    const bool outComplex = x.isComplex() || p.isComplexState;
    if (outComplex)
        p.isComplexState = true;

    Value y = Value::matrix(nRows, nCols, outComplex ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    for (keel::usize ch = 0; ch < nCols; ++ch) {
        auto &z = p.states[ch];
        for (keel::usize r = 0; r < nRows; ++r) {
            const keel::usize idx = r + ch * nRows;
            const keel::c64 xin = x.isComplex()
                ? x.complexData()[idx]
                : keel::c64{x.elemAsDouble(idx), 0.0};
            const keel::f64 b0 = p.num[0] / a0;
            const keel::c64 z0 = (nState > 0) ? z[0] : keel::c64{0.0, 0.0};
            const keel::c64 yout{b0 * xin.re + z0.re, b0 * xin.im + z0.im};

            for (keel::usize k = 0; k < nState; ++k) {
                const keel::f64 bk = (k + 1 < nb) ? (p.num[k + 1] / a0) : 0.0;
                const keel::f64 ak = (k + 1 < na) ? (p.den[k + 1] / a0) : 0.0;
                const keel::c64 znext = (k + 1 < nState) ? z[k + 1] : keel::c64{0.0, 0.0};
                z[k] = keel::c64{
                    bk * xin.re - ak * yout.re + znext.re,
                    bk * xin.im - ak * yout.im + znext.im
                };
            }
            if (outComplex)
                y.complexDataMut()[idx] = yout;
            else
                y.doubleDataMut()[idx] = yout.re;
        }
    }
    outs[0] = keel::move(y);
}

BuiltinClass makeIirFilterClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("dsp.IIRFilter");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<IirFilterPayload>(mr, mr);
        iirApplyNV(*payload, args, mr);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("dsp.IIRFilter", keel::move(st), mr);
    };
    cls.subsref = &iirStepImpl;
    cls.methods.put(keel::String::from("step"), &iirStepImpl);
    cls.methods.put(keel::String::from("reset"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<IirFilterPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.methods.put(keel::String::from("release"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<IirFilterPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.methods.put(keel::String::from("clone"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &ctx) {
                        auto *mr = allocOf(ctx);
                        if (const ObjectState *os = self.objectStateConst())
                            outs[0] = Value::object("dsp.IIRFilter", os->deepCopy(mr), mr);
                    });
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const ObjectState *os = self.objectStateConst();
        if (!os || !os->native) return false;
        const auto &p = *static_cast<const IirFilterPayload *>(os->native.get());
        if (name.equals_ignore_ascii_case("Structure")) {
            out = Value::fromString(p.structure, mr);
            return true;
        }
        if (name.equals_ignore_ascii_case("Numerator")) {
            out = fromRealVecRow(p.num, mr);
            return true;
        }
        if (name.equals_ignore_ascii_case("Denominator")) {
            out = fromRealVecRow(p.den, mr);
            return true;
        }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        ObjectState *os = self.objectStateMut();
        if (!os || !os->native) return false;
        auto &p = *static_cast<IirFilterPayload *>(os->native.get());
        if (name.equals_ignore_ascii_case("Structure")) {
            p.structure = val.toString();
            return true;
        }
        if (name.equals_ignore_ascii_case("Numerator")) {
            p.num = toRealVec(val, mr);
            p.resetStates();
            return true;
        }
        if (name.equals_ignore_ascii_case("Denominator")) {
            p.den = toRealVec(val, mr);
            p.resetStates();
            return true;
        }
        return false;
    };
    return cls;
}

void dspIirFilter_ctor_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                           keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<IirFilterPayload>(mr, mr);
    iirApplyNV(*payload, args, mr);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("dsp.IIRFilter", keel::move(st), mr);
}

// ── dsp.FIRInterpolator ─────────────────────────────────────────────────
struct FirInterpPayload final : public NativePayload {
    keel::i32 factor = 2;
    keel::Array<keel::f64> num;
    keel::Array<keel::Array<keel::c64>> delay;
    bool isComplex = false;

    explicit FirInterpPayload(keel::Allocator *mr = nullptr)
        : num(mr), delay(mr)
    {}

    void resetStates()
    {
        delay.clear();
        isComplex = false;
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<FirInterpPayload>(a, a);
        p->factor = factor;
        p->num = num.clone(a);
        p->isComplex = isComplex;
        p->delay.reserve(delay.size());
        for (keel::usize i = 0; i < delay.size(); ++i)
            p->delay.push_back(delay[i].clone(a));
        return p;
    }
};

void firInterpInitDefault(FirInterpPayload &p, keel::Allocator *mr)
{
    if (!p.num.empty())
        return;
    // Standard windowed-sinc interpolation filter of length 24*L + 1.
    const keel::i32 L = keel::max(1, p.factor);
    const keel::i32 halfLen = 12 * L;
    const keel::usize N = keel::as<keel::usize>(2 * halfLen + 1);
    p.num.resize(N, 0.0);
    const keel::f64 pi = 3.14159265358979323846;
    const keel::f64 fc = 0.5 / keel::as<keel::f64>(L);
    for (keel::usize k = 0; k < N; ++k) {
        const keel::f64 n = keel::as<keel::f64>(k) - keel::as<keel::f64>(halfLen);
        const keel::f64 sinc = (n == 0.0) ? (2.0 * fc)
                                          : (keel::sin(2.0 * pi * fc * n) / (pi * n));
        const keel::f64 win = 0.54 - 0.46 * keel::cos(2.0 * pi * keel::as<keel::f64>(k) / keel::as<keel::f64>(N - 1));
        p.num[k] = keel::as<keel::f64>(L) * sinc * win;
    }
    (void)mr;
}

void firInterpStepImpl(Value &self, keel::ArrayRef<Value> args,
                       keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                       CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "dsp.FIRInterpolator.step");
    auto *mr = allocOf(ctx);
    ObjectState *os = self.objectStateMut();
    if (!os || !os->native)
        throw Error("dsp.FIRInterpolator: invalid object state.", 0, 0, "step", "", "numkit:dsp:InvalidObject");
    auto &p = *static_cast<FirInterpPayload *>(os->native.get());
    firInterpInitDefault(p, mr);

    const Value &x = args[0];
    const keel::usize L = keel::as<keel::usize>(keel::max(1, p.factor));
    const keel::usize nRows = x.dims().rows();
    const keel::usize nCols = x.dims().cols();
    const keel::usize outRows = nRows * L;
    const keel::usize nTaps = p.num.size();
    const bool outComplex = x.isComplex() || p.isComplex;
    if (outComplex) p.isComplex = true;

    if (p.delay.size() != nCols) {
        p.delay.clear();
        p.delay.reserve(nCols);
        for (keel::usize ch = 0; ch < nCols; ++ch)
            p.delay.push_back(keel::Array<keel::c64>(nTaps > 0 ? nTaps - 1 : 0, keel::c64{0.0, 0.0}, mr));
    }

    Value y = Value::matrix(outRows, nCols, outComplex ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    for (keel::usize ch = 0; ch < nCols; ++ch) {
        auto &buf = p.delay[ch];
        const keel::usize histLen = buf.size();
        for (keel::usize r = 0; r < nRows; ++r) {
            const keel::usize idx = r + ch * nRows;
            const keel::c64 xin = x.isComplex()
                ? x.complexData()[idx]
                : keel::c64{x.elemAsDouble(idx), 0.0};
            for (keel::usize sub = 0; sub < L; ++sub) {
                const keel::c64 s = (sub == 0) ? xin : keel::c64{0.0, 0.0};
                keel::c64 acc{p.num[0] * s.re, p.num[0] * s.im};
                for (keel::usize k = 0; k < histLen; ++k) {
                    acc.re += p.num[k + 1] * buf[k].re;
                    acc.im += p.num[k + 1] * buf[k].im;
                }
                for (keel::usize k = histLen; k > 1; --k)
                    buf[k - 1] = buf[k - 2];
                if (histLen > 0)
                    buf[0] = s;
                const keel::usize outIdx = (r * L + sub) + ch * outRows;
                if (outComplex) y.complexDataMut()[outIdx] = acc;
                else y.doubleDataMut()[outIdx] = acc.re;
            }
        }
    }
    outs[0] = keel::move(y);
}

void firInterpApplyArgs(FirInterpPayload &p, keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    keel::usize i = 0;
    if (args.size() >= 1 && args[0].isNumeric()) {
        p.factor = keel::max(1, keel::as<keel::i32>(keel::round_i64(args[0].toScalar())));
        i = 1;
        if (args.size() >= 2 && args[1].isNumeric()) {
            p.num = toRealVec(args[1], mr);
            i = 2;
        }
    }
    for (; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String key = args[i].toString();
        if (key.equals_ignore_ascii_case("InterpolationFactor")) {
            p.factor = keel::max(1, keel::as<keel::i32>(keel::round_i64(args[i + 1].toScalar())));
            p.resetStates();
        } else if (key.equals_ignore_ascii_case("Numerator")) {
            p.num = toRealVec(args[i + 1], mr);
            p.resetStates();
        }
    }
}

BuiltinClass makeFirInterpClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("dsp.FIRInterpolator");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<FirInterpPayload>(mr, mr);
        firInterpApplyArgs(*payload, args, mr);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("dsp.FIRInterpolator", keel::move(st), mr);
    };
    cls.subsref = &firInterpStepImpl;
    cls.methods.put(keel::String::from("step"), &firInterpStepImpl);
    cls.methods.put(keel::String::from("reset"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<FirInterpPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.methods.put(keel::String::from("release"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<FirInterpPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    return cls;
}

void dspFirInterp_ctor_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                           keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<FirInterpPayload>(mr, mr);
    firInterpApplyArgs(*payload, args, mr);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("dsp.FIRInterpolator", keel::move(st), mr);
}

// ── dsp.VariableFractionalDelay ─────────────────────────────────────────
struct VarFracDelayPayload final : public NativePayload {
    keel::String interpMethod;
    keel::i32 maxDelay = 65535;
    keel::f64 initCond = 0.0;
    // Per-channel circular history buffer: ring[ch] of size cap, head = next write slot
    keel::Array<keel::Array<keel::c64>> rings;
    keel::usize ringCap = 0;
    keel::usize head = 0;
    keel::usize filled = 0;
    bool isComplex = false;

    explicit VarFracDelayPayload(keel::Allocator *mr = nullptr)
        : interpMethod(keel::String::from("Linear"))
        , rings(mr)
    {}

    void resetStates()
    {
        rings.clear();
        ringCap = 0;
        head = 0;
        filled = 0;
        isComplex = false;
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<VarFracDelayPayload>(a, a);
        p->interpMethod = interpMethod.clone();
        p->maxDelay = maxDelay;
        p->initCond = initCond;
        p->ringCap = ringCap;
        p->head = head;
        p->filled = filled;
        p->isComplex = isComplex;
        p->rings.reserve(rings.size());
        for (keel::usize i = 0; i < rings.size(); ++i)
            p->rings.push_back(rings[i].clone(a));
        return p;
    }
};

void vfdApplyNV(VarFracDelayPayload &p, keel::ArrayRef<Value> args)
{
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("InterpolationMethod")) {
            p.interpMethod = args[i + 1].toString();
        } else if (k.equals_ignore_ascii_case("MaximumDelay")) {
            p.maxDelay = keel::max(1, keel::as<keel::i32>(keel::round_i64(args[i + 1].toScalar())));
            p.resetStates();
        } else if (k.equals_ignore_ascii_case("InitialConditions")) {
            p.initCond = args[i + 1].toScalar();
            p.resetStates();
        }
    }
}

void vfdStepImpl(Value &self, keel::ArrayRef<Value> args,
                 keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "dsp.VariableFractionalDelay.step");
    auto *mr = allocOf(ctx);
    ObjectState *os = self.objectStateMut();
    if (!os || !os->native)
        throw Error("dsp.VariableFractionalDelay: invalid object state.", 0, 0, "step", "", "numkit:dsp:InvalidObject");
    auto &p = *static_cast<VarFracDelayPayload *>(os->native.get());

    const Value &x = args[0];
    const Value &dVal = args[1];
    if (x.isEmpty()) {
        outs[0] = Value::matrix(x.dims().rows(), x.dims().cols(), x.type(), mr);
        return;
    }
    const keel::usize nRows = x.dims().rows();
    const keel::usize nCols = x.dims().cols();
    const bool outComplex = x.isComplex() || p.isComplex;
    if (outComplex) p.isComplex = true;

    // Determine max integer delay needed in this frame so ring stays compact.
    keel::f64 maxD = 0.0;
    const keel::usize dNum = dVal.numel();
    for (keel::usize i = 0; i < dNum; ++i) {
        const keel::f64 dv = keel::clamp(dVal.elemAsDouble(i), 0.0, keel::as<keel::f64>(p.maxDelay));
        if (dv > maxD) maxD = dv;
    }
    const keel::usize reqCap = keel::as<keel::usize>(keel::floor(maxD)) + 4;
    const keel::usize maxCap = keel::as<keel::usize>(p.maxDelay) + 4;
    const keel::usize targetCap = keel::min(maxCap, keel::max<keel::usize>(reqCap, 64));

    if (p.rings.size() != nCols) {
        p.rings.clear();
        p.rings.reserve(nCols);
        for (keel::usize ch = 0; ch < nCols; ++ch)
            p.rings.push_back(keel::Array<keel::c64>(targetCap, keel::c64{p.initCond, 0.0}, mr));
        p.ringCap = targetCap;
        p.head = 0;
        p.filled = 0;
    } else if (targetCap > p.ringCap) {
        // Grow ring while preserving most recent p.filled samples in chronological order.
        const keel::usize oldCap = p.ringCap;
        const keel::usize keep = keel::min(p.filled, oldCap);
        for (keel::usize ch = 0; ch < nCols; ++ch) {
            keel::Array<keel::c64> nr(targetCap, keel::c64{p.initCond, 0.0}, mr);
            for (keel::usize k = 0; k < keep; ++k) {
                const keel::usize oldIdx = (p.head + oldCap - keep + k) % oldCap;
                nr[k] = p.rings[ch][oldIdx];
            }
            p.rings[ch] = keel::move(nr);
        }
        p.ringCap = targetCap;
        p.head = keep % targetCap;
        p.filled = keep;
    }

    Value y = Value::matrix(nRows, nCols, outComplex ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    const keel::usize cap = p.ringCap;
    const keel::f64 dMaxF = keel::as<keel::f64>(p.maxDelay);

    for (keel::usize r = 0; r < nRows; ++r) {
        const keel::usize curPos = p.head;
        for (keel::usize ch = 0; ch < nCols; ++ch) {
            const keel::usize idx = r + ch * nRows;
            const keel::c64 xin = x.isComplex()
                ? x.complexData()[idx]
                : keel::c64{x.elemAsDouble(idx), 0.0};
            p.rings[ch][curPos] = xin;

            const keel::f64 dRaw = (dNum == 1)
                ? dVal.elemAsDouble(0)
                : (dNum == nRows ? dVal.elemAsDouble(r) : dVal.elemAsDouble(idx % dNum));
            const keel::f64 d = keel::clamp(dRaw, 0.0, dMaxF);
            const keel::usize d0 = keel::as<keel::usize>(keel::floor(d));
            const keel::f64 frac = d - keel::as<keel::f64>(d0);

            auto sampleAt = [&](keel::usize lag) -> keel::c64 {
                if (lag > p.filled || lag >= cap)
                    return keel::c64{p.initCond, 0.0};
                const keel::usize slot = (curPos + cap - lag) % cap;
                return p.rings[ch][slot];
            };

            const keel::c64 s0 = sampleAt(d0);
            const keel::c64 s1 = (frac > 0.0) ? sampleAt(d0 + 1) : s0;
            const keel::c64 yout{
                (1.0 - frac) * s0.re + frac * s1.re,
                (1.0 - frac) * s0.im + frac * s1.im
            };
            if (outComplex) y.complexDataMut()[idx] = yout;
            else y.doubleDataMut()[idx] = yout.re;
        }
        p.head = (p.head + 1) % cap;
        if (p.filled < cap) ++p.filled;
    }
    outs[0] = keel::move(y);
}

BuiltinClass makeVarFracDelayClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("dsp.VariableFractionalDelay");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<VarFracDelayPayload>(mr, mr);
        vfdApplyNV(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("dsp.VariableFractionalDelay", keel::move(st), mr);
    };
    cls.subsref = &vfdStepImpl;
    cls.methods.put(keel::String::from("step"), &vfdStepImpl);
    cls.methods.put(keel::String::from("reset"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<VarFracDelayPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.methods.put(keel::String::from("release"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<VarFracDelayPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.methods.put(keel::String::from("clone"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &ctx) {
                        auto *mr = allocOf(ctx);
                        if (const ObjectState *os = self.objectStateConst())
                            outs[0] = Value::object("dsp.VariableFractionalDelay", os->deepCopy(mr), mr);
                    });
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const ObjectState *os = self.objectStateConst();
        if (!os || !os->native) return false;
        const auto &p = *static_cast<const VarFracDelayPayload *>(os->native.get());
        if (name.equals_ignore_ascii_case("InterpolationMethod")) { out = Value::fromString(p.interpMethod, mr); return true; }
        if (name.equals_ignore_ascii_case("MaximumDelay")) { out = Value::scalar(keel::as<keel::f64>(p.maxDelay), mr); return true; }
        if (name.equals_ignore_ascii_case("InitialConditions")) { out = Value::scalar(p.initCond, mr); return true; }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &) -> bool {
        ObjectState *os = self.objectStateMut();
        if (!os || !os->native) return false;
        auto &p = *static_cast<VarFracDelayPayload *>(os->native.get());
        if (name.equals_ignore_ascii_case("InterpolationMethod")) { p.interpMethod = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("MaximumDelay")) { p.maxDelay = keel::max(1, keel::as<keel::i32>(keel::round_i64(val.toScalar()))); p.resetStates(); return true; }
        if (name.equals_ignore_ascii_case("InitialConditions")) { p.initCond = val.toScalar(); p.resetStates(); return true; }
        return false;
    };
    return cls;
}

void dspVarFracDelay_ctor_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<VarFracDelayPayload>(mr, mr);
    vfdApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("dsp.VariableFractionalDelay", keel::move(st), mr);
}

// ── dsp.ArrayPlot ───────────────────────────────────────────────────────
struct ArrayPlotPayload final : public NativePayload {
    keel::i32 numInputPorts = 1;
    keel::f64 xOffset = 0.0;
    keel::f64 sampleIncrement = 1.0;
    bool showGrid = false;
    keel::String title;
    keel::String xLabel;
    keel::String yLabel;
    keel::String name;
    keel::Array<keel::f64> yLimits;
    keel::Array<keel::f64> position;

    explicit ArrayPlotPayload(keel::Allocator *mr = nullptr)
        : title(keel::String::from(""))
        , xLabel(keel::String::from(""))
        , yLabel(keel::String::from(""))
        , name(keel::String::from("Array Plot"))
        , yLimits(mr)
        , position(mr)
    {
        yLimits.push_back(-10.0);
        yLimits.push_back(10.0);
        position.push_back(100.0);
        position.push_back(100.0);
        position.push_back(600.0);
        position.push_back(400.0);
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<ArrayPlotPayload>(a, a);
        p->numInputPorts = numInputPorts;
        p->xOffset = xOffset;
        p->sampleIncrement = sampleIncrement;
        p->showGrid = showGrid;
        p->title = title.clone();
        p->xLabel = xLabel.clone();
        p->yLabel = yLabel.clone();
        p->name = name.clone();
        p->yLimits = yLimits.clone(a);
        p->position = position.clone(a);
        return p;
    }
};

void arrayPlotApplyNV(ArrayPlotPayload &p, keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("NumInputPorts")) p.numInputPorts = keel::max(1, keel::as<keel::i32>(keel::round_i64(args[i + 1].toScalar())));
        else if (k.equals_ignore_ascii_case("XOffset")) p.xOffset = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("SampleIncrement")) p.sampleIncrement = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("ShowGrid")) p.showGrid = args[i + 1].toBool();
        else if (k.equals_ignore_ascii_case("Title")) p.title = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("XLabel")) p.xLabel = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("YLabel")) p.yLabel = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("Name")) p.name = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("YLimits")) p.yLimits = toRealVec(args[i + 1], mr);
        else if (k.equals_ignore_ascii_case("Position")) p.position = toRealVec(args[i + 1], mr);
    }
}

void arrayPlotStepImpl(Value & /*self*/, keel::ArrayRef<Value> args,
                       keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                       CallContext &)
{
    argcheck::checkNargin(args.size(), 1, 16, "dsp.ArrayPlot.step");
    if (!outs.empty()) outs[0] = Value();
}

BuiltinClass makeArrayPlotClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("dsp.ArrayPlot");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<ArrayPlotPayload>(mr, mr);
        arrayPlotApplyNV(*payload, args, mr);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("dsp.ArrayPlot", keel::move(st), mr);
    };
    cls.subsref = &arrayPlotStepImpl;
    cls.methods.put(keel::String::from("step"), &arrayPlotStepImpl);
    auto noop = [](Value &, keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value> outs, CallContext &) {
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), noop);
    cls.methods.put(keel::String::from("release"), noop);
    cls.methods.put(keel::String::from("show"), noop);
    cls.methods.put(keel::String::from("hide"), noop);
    cls.methods.put(keel::String::from("clone"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &ctx) {
                        auto *mr = allocOf(ctx);
                        if (const ObjectState *os = self.objectStateConst())
                            outs[0] = Value::object("dsp.ArrayPlot", os->deepCopy(mr), mr);
                    });
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const ObjectState *os = self.objectStateConst();
        if (!os || !os->native) return false;
        const auto &p = *static_cast<const ArrayPlotPayload *>(os->native.get());
        if (name.equals_ignore_ascii_case("NumInputPorts")) { out = Value::scalar(keel::as<keel::f64>(p.numInputPorts), mr); return true; }
        if (name.equals_ignore_ascii_case("XOffset")) { out = Value::scalar(p.xOffset, mr); return true; }
        if (name.equals_ignore_ascii_case("SampleIncrement")) { out = Value::scalar(p.sampleIncrement, mr); return true; }
        if (name.equals_ignore_ascii_case("ShowGrid")) { out = Value::logicalScalar(p.showGrid, mr); return true; }
        if (name.equals_ignore_ascii_case("Title")) { out = Value::fromString(p.title, mr); return true; }
        if (name.equals_ignore_ascii_case("XLabel")) { out = Value::fromString(p.xLabel, mr); return true; }
        if (name.equals_ignore_ascii_case("YLabel")) { out = Value::fromString(p.yLabel, mr); return true; }
        if (name.equals_ignore_ascii_case("Name")) { out = Value::fromString(p.name, mr); return true; }
        if (name.equals_ignore_ascii_case("YLimits")) { out = fromRealVecRow(p.yLimits, mr); return true; }
        if (name.equals_ignore_ascii_case("Position")) { out = fromRealVecRow(p.position, mr); return true; }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        ObjectState *os = self.objectStateMut();
        if (!os || !os->native) return false;
        auto &p = *static_cast<ArrayPlotPayload *>(os->native.get());
        if (name.equals_ignore_ascii_case("NumInputPorts")) { p.numInputPorts = keel::max(1, keel::as<keel::i32>(keel::round_i64(val.toScalar()))); return true; }
        if (name.equals_ignore_ascii_case("XOffset")) { p.xOffset = val.toScalar(); return true; }
        if (name.equals_ignore_ascii_case("SampleIncrement")) { p.sampleIncrement = val.toScalar(); return true; }
        if (name.equals_ignore_ascii_case("ShowGrid")) { p.showGrid = val.toBool(); return true; }
        if (name.equals_ignore_ascii_case("Title")) { p.title = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("XLabel")) { p.xLabel = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("YLabel")) { p.yLabel = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("Name")) { p.name = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("YLimits")) { p.yLimits = toRealVec(val, mr); return true; }
        if (name.equals_ignore_ascii_case("Position")) { p.position = toRealVec(val, mr); return true; }
        return true; // tolerate additional visualization properties cleanly
    };
    return cls;
}

void dspArrayPlot_ctor_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                           keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<ArrayPlotPayload>(mr, mr);
    arrayPlotApplyNV(*payload, args, mr);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("dsp.ArrayPlot", keel::move(st), mr);
}

} // anonymous namespace

void installDspSystemObjects(PluginContext &engine)
{
    engine.registerClass(makeIirFilterClass());
    engine.registerFunction("dsp", "IIRFilter", &dspIirFilter_ctor_reg);
    engine.registerClass(makeFirInterpClass());
    engine.registerFunction("dsp", "FIRInterpolator", &dspFirInterp_ctor_reg);
    engine.registerClass(makeVarFracDelayClass());
    engine.registerFunction("dsp", "VariableFractionalDelay", &dspVarFracDelay_ctor_reg);
    engine.registerClass(makeArrayPlotClass());
    engine.registerFunction("dsp", "ArrayPlot", &dspArrayPlot_ctor_reg);
}

} // namespace numkit::signal::detail
