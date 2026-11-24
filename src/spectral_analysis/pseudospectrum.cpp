// toolboxes/signal/src/spectral_analysis/pseudospectrum.cpp
//
// Subspace pseudospectrum estimators: pmusic and peig (delegates to keel::signal).

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <keel/tuple.h>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>

#include <numkit/value/value.hpp>

namespace numkit::signal {

namespace {

keel::Tuple<Value, Value>
pseudospectrum(const Value &x, keel::i32 p, keel::i32 nfft, keel::f64 fs, bool eigWeight,
               keel::Allocator *mr)
{
    if (p < 1) p = 1;
    if (nfft < 2) nfft = 256;
    const keel::usize nf = keel::as<keel::usize>(nfft / 2 + 1);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(x.numel(), scratch.allocator());
    for (keel::usize i = 0; i < x.numel(); ++i) xv[i] = x.elemAsDouble(i);

    Value P = Value::matrix(nf, 1, ValueType::DOUBLE, mr);
    Value F = Value::matrix(nf, 1, ValueType::DOUBLE, mr);
    if (eigWeight) {
        (void)keel::signal::peig(xv.as_ref(), p, nfft, fs,
                                 keel::MutArrayRef<keel::f64>(P.doubleDataMut(), nf),
                                 keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nf),
                                 scratch.allocator());
    } else {
        (void)keel::signal::pmusic(xv.as_ref(), p, nfft, fs,
                                   keel::MutArrayRef<keel::f64>(P.doubleDataMut(), nf),
                                   keel::MutArrayRef<keel::f64>(F.doubleDataMut(), nf),
                                   scratch.allocator());
    }
    return keel::make_tuple(keel::move(P), keel::move(F));
}

} // namespace

keel::Tuple<Value, Value>
pmusic(const Value &x, keel::i32 p, keel::i32 nfft, keel::f64 fs, keel::Allocator *mr)
{
    return pseudospectrum(x, p, nfft, fs, /*eigWeight=*/false, mr);
}

keel::Tuple<Value, Value>
peig(const Value &x, keel::i32 p, keel::i32 nfft, keel::f64 fs, keel::Allocator *mr)
{
    return pseudospectrum(x, p, nfft, fs, /*eigWeight=*/true, mr);
}

} // namespace numkit::signal
