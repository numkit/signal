// toolboxes/signal/src/spectral_analysis/ar_reg.cpp
//
// CallContext register half of spectral_analysis/ar.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/spectral_analysis/periodogram_pwelch.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

void pyulear_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 8, "pyulear");
    const int p     = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "p", "pyulear"), 2, "p", "pyulear");
    const keel::usize nf = (args.size() >= 3 && !args[2].isEmpty())
                      ? static_cast<keel::usize>(argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "pyulear"), 3, "nfft", "pyulear")) : 0;
    auto [Pxx, F] = pyulear(args[0], p, nf, ctx.engine->allocator());
    outs[0] = std::move(Pxx);
    if (nargout > 1) outs[1] = std::move(F);
}

void pburg_reg(keel::ArrayRef<Value> args, keel::usize nargout,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 8, "pburg");
    const int p     = argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "p", "pburg"), 2, "p", "pburg");
    const keel::usize nf = (args.size() >= 3 && !args[2].isEmpty())
                      ? static_cast<keel::usize>(argcheck::expectedPositiveInteger(argcheck::expectedReal(args[2], 3, "nfft", "pburg"), 3, "nfft", "pburg")) : 0;
    auto [Pxx, F] = pburg(args[0], p, nf, ctx.engine->allocator());
    outs[0] = std::move(Pxx);
    if (nargout > 1) outs[1] = std::move(F);
}

// aryule_reg / lpc_reg live in signal_modeling.cpp.

} // namespace detail

} // namespace numkit::signal
