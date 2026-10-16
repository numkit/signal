// toolboxes/signal/src/spectral_analysis/pseudospectrum_reg.cpp
//
// CallContext register half of spectral_analysis/pseudospectrum.cpp
// (pmusic / peig). Parses (x, p[, nfft[, fs]]) and threads (P, F) by nargout.
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/spectral_analysis/signal_modeling.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/span.hpp>

#include <tuple>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::signal {

namespace detail {

void pmusic_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "pmusic");  // no max: name-value tail in the reference
    const int    p    = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "p", "pmusic"), 2, "p", "pmusic");
    const int    nfft = (args.size() >= 3 && !args[2].isEmpty())
                       ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "pmusic"), 3, "nfft", "pmusic") : 256;
    const double fs   = (args.size() >= 4 && !args[3].isEmpty()) ? args[3].toScalar() : 2.0 * M_PI;
    auto [P, F] = pmusic(args[0], p, nfft, fs, ctx.engine->resource());
    outs[0] = std::move(P);
    if (nargout > 1) outs[1] = std::move(F);
}

void peig_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 9, "peig");
    const int    p    = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "p", "peig"), 2, "p", "peig");
    const int    nfft = (args.size() >= 3 && !args[2].isEmpty())
                       ? argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "peig"), 3, "nfft", "peig") : 256;
    const double fs   = (args.size() >= 4 && !args[3].isEmpty()) ? args[3].toScalar() : 2.0 * M_PI;
    auto [P, F] = peig(args[0], p, nfft, fs, ctx.engine->resource());
    outs[0] = std::move(P);
    if (nargout > 1) outs[1] = std::move(F);
}

} // namespace detail

} // namespace numkit::signal
