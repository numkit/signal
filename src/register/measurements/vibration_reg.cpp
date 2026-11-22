// toolboxes/signal/src/measurements/vibration_reg.cpp
//
// CallContext register half of measurements/vibration.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/measurements/vibration.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/signal/transforms/hilbert.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

void envspectrum_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 8, "envspectrum");
    keel::f64 fs = 0.0;
    if (args.size() >= 2 && !args[1].isEmpty()) fs = args[1].toScalar();
    auto [Es, F] = envspectrum(args[0], fs, ctx.engine->allocator());
    outs[0] = std::move(Es);
    if (nargout > 1) outs[1] = std::move(F);
}

void tachorpm_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "tachorpm");  // reference takes name-value options past arg 4 (unimplemented)
    const keel::f64 fs = args[1].toScalar();
    const Value &thr = (args.size() >= 3) ? args[2] : Value::Empty;
    keel::i32 ppr = 1;
    if (args.size() >= 4 && !args[3].isEmpty())
        ppr = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[3], 4, "ppr", "tachorpm"), 4, "ppr", "tachorpm");
    auto [rpm, t] = tachorpm(args[0], fs, thr, ppr, ctx.engine->allocator());
    outs[0] = std::move(rpm);
    if (nargout > 1) outs[1] = std::move(t);
}

void rainflow_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "rainflow");
    outs[0] = rainflow(args[0], ctx.engine->allocator());
}

// MATLAB tsa convention: tsa(x, fs, tPulse) where tPulse is a vector
// of pulse arrival times (seconds) defining revolutions. Output is
// the average waveform over those revolutions.
//
// nk historic convention: tsa(x, fs, rpm, fs_rpm[, n_per_rev])
// where rpm is a continuously-sampled rotation-rate signal.
//
// Dispatcher: 3-arg call -> MATLAB form; 4+ arg call -> nk form.
namespace {

std::tuple<Value, Value>
tsa_matlab_form(keel::Allocator *mr, const Value &x, keel::f64 fs,
                const Value &tPulse)
{
    const keel::usize nx = x.numel();
    const keel::usize np = tPulse.numel();
    if (nx == 0 || np < 2) {
        return std::make_tuple(
            Value::matrix(0, 1, ValueType::DOUBLE, mr),
            Value::matrix(0, 1, ValueType::DOUBLE, mr));
    }
    const keel::f64 *xd = x.doubleData();
    keel::Array<keel::f64> tv(np, mr);
    for (keel::usize i = 0; i < np; ++i) tv[i] = tPulse.elemAsDouble(i);

    // Decide output length L = nominal samples per revolution. Use the
    // floor of the average inter-pulse interval times fs, matching
    // MATLAB's default behaviour.
    const keel::f64 T_total = tv[np - 1] - tv[0];
    const keel::usize nRev = np - 1;
    const keel::f64 T_avg = T_total / keel::as<keel::f64>(nRev);
    keel::usize L = keel::as<keel::usize>(keel::floor(T_avg * fs));
    if (L < 1) L = 1;

    // For each revolution, linear-interpolate x onto an L-point grid
    // that spans [tv[i], tv[i+1]) and accumulate.
    keel::Array<keel::f64> avg(L, 0.0, mr);
    keel::usize nValidRevs = 0;
    for (keel::usize r = 0; r < nRev; ++r) {
        const keel::f64 t0 = tv[r];
        const keel::f64 t1 = tv[r + 1];
        const keel::f64 dur = t1 - t0;
        if (dur <= 0.0) continue;
        bool revOK = true;
        keel::Array<keel::f64> sample(L, 0.0, mr);
        for (keel::usize k = 0; k < L; ++k) {
            const keel::f64 tk = t0 + dur * keel::as<keel::f64>(k)
                                    / keel::as<keel::f64>(L);
            const keel::f64 idxF = tk * fs;
            const keel::i64  i0   = keel::as<keel::i64>(keel::floor(idxF));
            const keel::f64 frac = idxF - keel::as<keel::f64>(i0);
            if (i0 < 0 || keel::as<keel::usize>(i0 + 1) >= nx) { revOK = false; break; }
            sample[k] = (1.0 - frac) * xd[i0] + frac * xd[i0 + 1];
        }
        if (!revOK) continue;
        for (keel::usize k = 0; k < L; ++k) avg[k] += sample[k];
        ++nValidRevs;
    }
    if (nValidRevs > 0) {
        for (keel::usize k = 0; k < L; ++k)
            avg[k] /= keel::as<keel::f64>(nValidRevs);
    }

    auto Ya = Value::matrix(L, 1, ValueType::DOUBLE, mr);
    auto Ta = Value::matrix(L, 1, ValueType::DOUBLE, mr);
    keel::f64 *ya = Ya.doubleDataMut();
    keel::f64 *ta = Ta.doubleDataMut();
    for (keel::usize k = 0; k < L; ++k) {
        ya[k] = avg[k];
        ta[k] = T_avg * keel::as<keel::f64>(k) / keel::as<keel::f64>(L);
    }
    return std::make_tuple(std::move(Ya), std::move(Ta));
}

} // namespace

void tsa_reg(keel::ArrayRef<Value> args, keel::usize nargout,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "tsa");  // args 4..5: legacy nk continuous-rpm form; reference: 3 numerical + name-value
    const keel::f64 fs = args[1].toScalar();
    if (args.size() == 3) {
        // MATLAB form: tsa(x, fs, tPulse)
        auto [avg, th] = tsa_matlab_form(ctx.engine->allocator(),
                                         args[0], fs, args[2]);
        outs[0] = std::move(avg);
        if (nargout > 1) outs[1] = std::move(th);
        return;
    }
    // 4+ args -> legacy nk form (continuous rpm signal).
    const keel::f64 fs_rpm = args[3].toScalar();
    keel::i32 npr = 1024;
    if (args.size() >= 5 && !args[4].isEmpty())
        npr = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[4], 5, "npr", "tsa"), 5, "npr", "tsa");
    auto [avg, th] = tsa(args[0], fs, args[2], fs_rpm, npr, ctx.engine->allocator());
    outs[0] = std::move(avg);
    if (nargout > 1) outs[1] = std::move(th);
}

} // namespace detail

} // namespace numkit::signal
