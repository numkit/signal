// toolboxes/signal/src/digital_filtering/quantizers_reg.cpp
//
// Register half of the signal quantizers builtins: the CallContext wrappers
// delegating to the engine-free compute in quantizers.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/signal/digital_filtering/quantizers.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

void uencode_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "uencode");
    // Marshalling only â€” the [2, 32] range contract lives in the
    // uencode compute (nk:uencode:BadN).
    const keel::i32 N = keel::as<keel::i32>(argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "N", "uencode"), 2, "N", "uencode"));
    keel::f64 V = 1.0;
    if (args.size() >= 3 && !args[2].isEmpty())
        V = argcheck::expectedReal(args[2], 3, "V", "uencode");
    bool signedOut = false;
    if (args.size() >= 4 && !args[3].isEmpty()) {
        keel::String s = args[3].toString();
        std::transform(s.begin(), s.end(), s.begin(),
                        [](keel::u8 c) { return keel::to_ascii_lower(c); });
        if (s == "signed")        signedOut = true;
        else if (s == "unsigned") signedOut = false;
        else throw Error("uencode: 4th arg must be 'signed' or 'unsigned'",
                          0, 0, "uencode", "", "numkit:uencode:Polarity");
    }
    outs[0] = uencode(args[0], N, V, signedOut, ctx.engine->allocator());
}

void udecode_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "udecode");
    // Marshalling only: the [2, 32] range and V positivity contracts
    // live in udecode_argcheck (compute side).
    const keel::i32 N = keel::as<keel::i32>(argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "N", "udecode"), 2, "N", "udecode"));
    keel::f64 V = 1.0;
    if (args.size() >= 3 && !args[2].isEmpty())
        V = argcheck::expectedReal(args[2], 3, "V", "udecode");
    bool wrap = false;
    if (args.size() >= 4 && !args[3].isEmpty()) {
        keel::String s = args[3].toString();
        std::transform(s.begin(), s.end(), s.begin(),
                        [](keel::u8 c) { return keel::to_ascii_lower(c); });
        if (s == "wrap")          wrap = true;
        else if (s == "saturate") wrap = false;
        else throw Error("udecode: 4th arg must be 'saturate' or 'wrap'",
                          0, 0, "udecode", "", "numkit:udecode:BadOpt");
    }
    outs[0] = udecode(args[0], N, V, wrap, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
