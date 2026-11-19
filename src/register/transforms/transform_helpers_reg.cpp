// toolboxes/signal/src/transforms/transform_helpers_reg.cpp
//
// CallContext register half of transforms/transform_helpers.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/transforms/transform_helpers.hpp>
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

void nextpow2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "nextpow2");
    outs[0] = nextpow2(args[0], ctx.engine->allocator());
}

void fftshift_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "fftshift");
    if (args.size() >= 2) {
        const int dim = argcheck::dimArg(args[1], "fftshift");
        if (dim < 1 || dim > 3)
            throw Error("fftshift: dim must be 1, 2, or 3",
                         0, 0, "fftshift", "", "numkit:fftshift:dim");
        outs[0] = fftshift(args[0], dim, ctx.engine->allocator());
    } else {
        outs[0] = fftshift(args[0], ctx.engine->allocator());
    }
}

void ifftshift_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "ifftshift");
    if (args.size() >= 2) {
        const int dim = argcheck::dimArg(args[1], "ifftshift");
        if (dim < 1 || dim > 3)
            throw Error("ifftshift: dim must be 1, 2, or 3",
                         0, 0, "ifftshift", "", "numkit:ifftshift:dim");
        outs[0] = ifftshift(args[0], dim, ctx.engine->allocator());
    } else {
        outs[0] = ifftshift(args[0], ctx.engine->allocator());
    }
}

} // namespace detail

} // namespace numkit::signal
