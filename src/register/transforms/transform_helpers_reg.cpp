// toolboxes/signal/src/transforms/transform_helpers_reg.cpp
//
// CallContext register half of transforms/transform_helpers.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/plugin.hpp>
#include <numkit/signal/transforms/transform_helpers.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::signal {

namespace detail {

void nextpow2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "nextpow2");
    outs[0] = nextpow2(args[0], ctx.engine->resource());
}

void fftshift_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "fftshift");
    if (args.size() >= 2) {
        const int dim = argcheck::dimArg(args[1], "fftshift");
        if (dim < 1 || dim > 3)
            throw Error("fftshift: dim must be 1, 2, or 3",
                         0, 0, "fftshift", "", "nk:fftshift:dim");
        outs[0] = fftshift(args[0], dim, ctx.engine->resource());
    } else {
        outs[0] = fftshift(args[0], ctx.engine->resource());
    }
}

void ifftshift_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "ifftshift");
    if (args.size() >= 2) {
        const int dim = argcheck::dimArg(args[1], "ifftshift");
        if (dim < 1 || dim > 3)
            throw Error("ifftshift: dim must be 1, 2, or 3",
                         0, 0, "ifftshift", "", "nk:ifftshift:dim");
        outs[0] = ifftshift(args[0], dim, ctx.engine->resource());
    } else {
        outs[0] = ifftshift(args[0], ctx.engine->resource());
    }
}

} // namespace detail

} // namespace numkit::signal
