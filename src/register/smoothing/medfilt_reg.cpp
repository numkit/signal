// toolboxes/signal/src/smoothing/medfilt_reg.cpp
//
// Register half of the signal medfilt builtins: the CallContext wrappers
// delegating to the engine-free compute in medfilt.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/signal/smoothing/medfilt.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>


namespace numkit::signal {

namespace detail {

void medfilt1_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 6, "medfilt1");
    keel::usize k = 3;
    if (args.size() >= 2 && !args[1].isEmpty()
        && !args[1].isChar() && !args[1].isString()) {
        // k = 0 is LEGAL (the reference answers an all-NaN signal â€”
        // probed R2025b); only negative k is rejected.
        const keel::i64 kv = argcheck::expectedInteger(
            argcheck::expectedReal(args[1], 2, "k", "medfilt1"), 2, "k", "medfilt1");
        if (kv < 0)
            throw Error("medfilt1: window length must be nonnegative",
                         0, 0, "medfilt1", "", "numkit:medfilt1:badK");
        k = keel::as<keel::usize>(kv);
    }

    // Padding mode: MATLAB default 'zeropad'; a trailing 'truncate' string
    // (medfilt1(x,n,'truncate') or medfilt1(x,n,[],dim,'truncate')) clips
    // the window at the ends instead. (blksz/dim/nanflag args are accepted
    // but ignored for now â€” noted as a gap.)
    bool zeropad = true;
    for (keel::usize a = args.size(); a-- > 1;) {
        if (args[a].isChar() || args[a].isString()) {
            keel::String s = args[a].toString();
            s.to_ascii_lower_in_place();
            if (s == "truncate") zeropad = false;
            else if (s == "zeropad") zeropad = true;
            break;
        }
    }
    outs[0] = medfilt1(args[0], k, zeropad, ctx.engine->allocator());
}

void fillgaps_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "fillgaps");
    keel::usize maxlen = 0;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const keel::f64 ml = argcheck::expectedReal(args[1], 2, "MAXLEN", "fillgaps");
        if (keel::is_inf(ml) && ml > 0.0) {
            maxlen = 0;
        } else {
            maxlen = keel::as<keel::usize>(
                argcheck::expectedPositiveInteger(ml, 2, "MAXLEN", "fillgaps"));
        }
    }

    keel::usize order = 0;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (args[2].isChar() || args[2].isString()) {
            keel::String raw = args[2].toString();
            keel::String s = raw.clone();
            s.to_ascii_lower_in_place();
            if (s != "aic")
                throw Error(keel::format("fillgaps: ORDER string must be 'aic' (got '{}')", raw),
                            0, 0, "fillgaps", "", "numkit:fillgaps:invalidOrder");
            order = 0;
        } else {
            const keel::f64 ord = argcheck::expectedReal(args[2], 3, "ORDER", "fillgaps");
            if (keel::is_inf(ord) && ord > 0.0) {
                order = 0;
            } else {
                order = keel::as<keel::usize>(
                    argcheck::expectedPositiveInteger(ord, 3, "ORDER", "fillgaps"));
            }
        }
    }

    outs[0] = fillgaps(args[0], maxlen, order, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
