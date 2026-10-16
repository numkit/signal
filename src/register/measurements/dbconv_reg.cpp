// toolboxes/signal/src/measurements/dbconv_reg.cpp
//
// CallContext register half of measurements/dbconv.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/measurements/dbconv.hpp>
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
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void db_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "db");
    std::string mode = "voltage";
    if (args.size() >= 2) {
        if (!args[1].isChar() && !args[1].isString())
            throw Error("db: 2nd argument must be a string",
                         0, 0, "db", "", "numkit:db:badType");
        mode = args[1].toString();
    }
    outs[0] = db(args[0], mode, ctx.engine->resource());
}

void db2mag_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "db2mag");
    outs[0] = db2mag(args[0], ctx.engine->resource());
}

void mag2db_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "mag2db");
    outs[0] = mag2db(args[0], ctx.engine->resource());
}

void db2pow_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "db2pow");
    outs[0] = db2pow(args[0], ctx.engine->resource());
}

void pow2db_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "pow2db");
    outs[0] = pow2db(args[0], ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
