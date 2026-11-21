// toolboxes/signal/src/measurements/sig_utils_reg.cpp
//
// CallContext register half of measurements/sig_utils.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/measurements/sig_utils.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void seqperiod_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "seqperiod");
    keel::f64 tol = 1e-10;
    if (args.size() >= 2) tol = args[1].toScalar();
    auto [p, nr] = seqperiod(args[0], tol, ctx.engine->allocator());
    outs[0] = p;
    if (nargout >= 2 && outs.size() >= 2) outs[1] = nr;
}

void zerocrossrate_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "zerocrossrate");  // no max: name-value tail in the reference
    keel::f64 level = 0.0;
    if (args.size() >= 2) level = args[1].toScalar();
    auto [r, c] = zerocrossrate(args[0], level, ctx.engine->allocator());
    outs[0] = r;
    if (nargout >= 2 && outs.size() >= 2) outs[1] = c;
}

void cusum_reg(keel::ArrayRef<Value> args, keel::usize nargout,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 6, "cusum");
    const keel::f64 climit = (args.size() >= 2) ? args[1].toScalar() : 5.0;
    const keel::f64 mshift = (args.size() >= 3) ? args[2].toScalar() : 1.0;
    CusumResult R = cusum(args[0], climit, mshift,
                          (args.size() >= 4) ? args[3] : Value::Empty,
                          (args.size() >= 5) ? args[4] : Value::Empty,
                          ctx.engine->allocator());
    outs[0] = R.iupper;
    if (nargout >= 2 && outs.size() >= 2) outs[1] = R.ilower;
    if (nargout >= 3 && outs.size() >= 3) outs[2] = R.uppersum;
    if (nargout >= 4 && outs.size() >= 4) outs[3] = R.lowersum;
}

} // namespace detail

} // namespace numkit::signal
