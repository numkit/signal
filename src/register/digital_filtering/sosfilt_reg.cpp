// toolboxes/signal/src/digital_filtering/sosfilt_reg.cpp
//
// Register half of the signal sosfilt builtins: the CallContext wrappers
// delegating to the engine-free compute in sosfilt.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/signal/digital_filtering/sosfilt.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void sosfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "sosfilt");
    outs[0] = sosfilt(args[0], args[1], ctx.engine->allocator());
}

void sosfiltfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "sosfiltfilt");  // absent from the R2025b reference build; historical signature
    outs[0] = sosfiltfilt(args[0], args[1], ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
