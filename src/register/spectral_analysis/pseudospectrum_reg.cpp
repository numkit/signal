// toolboxes/signal/src/spectral_analysis/pseudospectrum_reg.cpp
//
// CallContext register half of spectral_analysis/pseudospectrum.cpp
// (pmusic / peig). Parses (x, p[, nfft[, fs]]) and threads (P, F) by nargout.
#include <keel/core.h>
#include <keel/ref.h>
#include <keel/math.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>



namespace numkit::signal {

namespace detail {

void pmusic_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "pmusic");  // no max: name-value tail in the reference
    const keel::i32    p    = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "p", "pmusic"), 2, "p", "pmusic");
    const keel::i32    nfft = (args.size() >= 3 && !args[2].isEmpty())
                       ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "pmusic"), 3, "nfft", "pmusic") : 256;
    const keel::f64 fs   = (args.size() >= 4 && !args[3].isEmpty()) ? args[3].toScalar() : 2.0 * keel::kPi;
    auto [P, F] = pmusic(args[0], p, nfft, fs, ctx.engine->allocator());
    outs[0] = keel::move(P);
    if (nargout > 1) outs[1] = keel::move(F);
}

void peig_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 9, "peig");
    const keel::i32    p    = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "p", "peig"), 2, "p", "peig");
    const keel::i32    nfft = (args.size() >= 3 && !args[2].isEmpty())
                       ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "peig"), 3, "nfft", "peig") : 256;
    const keel::f64 fs   = (args.size() >= 4 && !args[3].isEmpty()) ? args[3].toScalar() : 2.0 * keel::kPi;
    auto [P, F] = peig(args[0], p, nfft, fs, ctx.engine->allocator());
    outs[0] = keel::move(P);
    if (nargout > 1) outs[1] = keel::move(F);
}

} // namespace detail

} // namespace numkit::signal
