// toolboxes/signal/src/transforms/hilbert_reg.cpp
//
// CallContext register half of transforms/hilbert.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>     // interp1 (spline)
#include <numkit/signal/transforms/hilbert.hpp>
#include <numkit/signal/windows/windows.hpp>          // kaiser
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <numkit/value/argcheck.hpp>

namespace numkit::signal {

namespace detail {

void hilbert_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "hilbert");
    outs[0] = hilbert(args[0], ctx.engine->allocator());
}

void envelope_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "envelope");
    auto *mr = ctx.engine->allocator();
    keel::i32 mode = 0;     // 0=default, 1=analytic, 2=rms, 3=peak
    keel::usize n = 0;
    if (args.size() >= 2 && !args[1].isEmpty())
        n = (keel::usize)args[1].toScalar();
    if (args.size() >= 3 && (args[2].isChar() || args[2].isString())) {
        keel::String m = args[2].toString();
        m.to_ascii_lower_in_place();
        if      (m == "analytic")           mode = 1;
        else if (m == "rms")                mode = 2;
        else if (m == "peak" || m == "peaks") mode = 3;
        else throw Error(keel::format("envelope: unknown method '{}' (expected 'analytic', 'rms', or 'peak')", m),
                         0, 0, "envelope", "", "numkit:envelope:badmethod");
    } else if (args.size() >= 2) {
        // n given but no method â†’ MATLAB defaults to 'analytic'.
        mode = 1;
    }
    if ((mode == 1 || mode == 2 || mode == 3) && n == 0)
        throw Error("envelope: n must be a positive integer for "
                    "'analytic'/'rms'/'peak' modes",
                    0, 0, "envelope", "", "numkit:envelope:badn");
    auto [up, lo] = envelope_full(args[0], mode, n, mr);
    outs[0] = std::move(up);
    if (nargout > 1) outs[1] = std::move(lo);
}

} // namespace detail

} // namespace numkit::signal
