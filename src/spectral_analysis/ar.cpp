// toolboxes/signal/src/spectral_analysis/ar.cpp
//
// Parametric AR PSD estimators (pyulear, pburg) delegating to keel::signal.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <keel/tuple.h>

namespace numkit::signal {

namespace {

void ar_argcheck(const Value &x, keel::i32 p, const char *fn) {
    if (p < 1)
        throw Error(keel::format("{}: order p must be ≥ 1", fn),
                    0, 0, fn, "", "numkit:ar:order");
    if (keel::as<keel::usize>(p) >= x.numel())
        throw Error(keel::format("{}: order p must be < length(x)", fn),
                    0, 0, fn, "", "numkit:ar:order");
}

} // anonymous

keel::Tuple<Value, Value>
pyulear(const Value &x, keel::i32 p, keel::usize nfft, keel::Allocator *mr) {
    ar_argcheck(x, p, "pyulear");
    if (nfft == 0) nfft = 256;
    const keel::usize N = x.numel();
    const keel::usize nOut = nfft / 2 + 1;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);

    Value Pxx = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::pyulear(xv.as_ref(), keel::as<keel::usize>(p), nfft,
                                keel::MutArrayRef<keel::f64>(Pxx.doubleDataMut(), nOut),
                                keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nOut),
                                scratch.allocator());
    return keel::make_tuple(keel::move(Pxx), keel::move(F));
}

keel::Tuple<Value, Value>
pburg(const Value &x, keel::i32 p, keel::usize nfft, keel::Allocator *mr) {
    ar_argcheck(x, p, "pburg");
    if (nfft == 0) nfft = 256;
    const keel::usize N = x.numel();
    const keel::usize nOut = nfft / 2 + 1;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) xv[i] = x.elemAsDouble(i);

    Value Pxx = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    Value F   = Value::matrix(nOut, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::pburg(xv.as_ref(), keel::as<keel::usize>(p), nfft,
                              keel::MutArrayRef<keel::f64>(Pxx.doubleDataMut(), nOut),
                              keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nOut),
                              scratch.allocator());
    return keel::make_tuple(keel::move(Pxx), keel::move(F));
}

} // namespace numkit::signal
