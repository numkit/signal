// toolboxes/signal/src/measurements/sigroi_reg.cpp
//
// CallContext register half of measurements/sigroi.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <numkit/plugin.hpp>
#include <numkit/signal/measurements/sigroi.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
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

// Marshalling for the sigroi scalar lengths: scalarity/integrality here
// (nk ids); range/content contracts (positive, nonnegative) live in the
// compute argchecks.
static int64_t sigroiLen(const Value &v, int argNum, const char *name,
                         const char *fn)
{
    return static_cast<int64_t>(argcheck::expectedInteger(
        argcheck::expectedReal(v, argNum, name, fn), argNum, name, fn));
}

void binmask2sigroi_reg(Span<const Value> args, size_t /*nargout*/,
                        Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "binmask2sigroi");
    outs[0] = binmask2sigroi(args[0], ctx.engine->resource());
}

void sigroi2binmask_reg(Span<const Value> args, size_t /*nargout*/,
                        Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "sigroi2binmask");
    int64_t len = -1;
    if (args.size() >= 2) len = sigroiLen(args[1], 2, "L", "sigroi2binmask");
    outs[0] = sigroi2binmask(args[0], len, ctx.engine->resource());
}

void extendsigroi_reg(Span<const Value> args, size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "extendsigroi");
    outs[0] = extendsigroi(args[0],
                           sigroiLen(args[1], 2, "SL", "extendsigroi"),
                           sigroiLen(args[2], 3, "SR", "extendsigroi"),
                           ctx.engine->resource());
}

void shortensigroi_reg(Span<const Value> args, size_t /*nargout*/,
                       Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "shortensigroi");
    outs[0] = shortensigroi(args[0],
                            sigroiLen(args[1], 2, "Lpre", "shortensigroi"),
                            sigroiLen(args[2], 3, "Lpost", "shortensigroi"),
                            ctx.engine->resource());
}

void mergesigroi_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "mergesigroi");
    outs[0] = mergesigroi(args[0],
                          sigroiLen(args[1], 2, "S", "mergesigroi"),
                          ctx.engine->resource());
}

void removesigroi_reg(Span<const Value> args, size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "removesigroi");
    outs[0] = removesigroi(args[0],
                           sigroiLen(args[1], 2, "maxLen", "removesigroi"),
                           ctx.engine->resource());
}

void extractsigroi_reg(Span<const Value> args, size_t /*nargout*/,
                       Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "extractsigroi");
    bool concat = false;
    if (args.size() >= 3) {
        concat = argcheck::expectedReal(args[2], 3, "concat",
                                        "extractsigroi") != 0.0;
    }
    outs[0] = extractsigroi(args[0], args[1], concat, ctx.engine->resource());
}

void sigrangebinmask_reg(Span<const Value> args, size_t /*nargout*/,
                         Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "sigrangebinmask");  // reference has an option-string tail (unimplemented)
    auto *mr = ctx.engine->resource();
    const Value &bound = args[1];
    if (bound.numel() == 1)
        outs[0] = sigrangebinmask(args[0], bound.toScalar(), mr);
    else if (bound.numel() == 2)
        outs[0] = sigrangebinmask(args[0], bound.elemAsDouble(0),
                                  bound.elemAsDouble(1), mr);
    else
        throw Error("sigrangebinmask: bound must be scalar or 2-element vector",
                    0, 0, "sigrangebinmask", "", "nk:sigrangebinmask:BadBound");
}

} // namespace detail

} // namespace numkit::signal
