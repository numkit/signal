// toolboxes/signal/src/digital_filtering/quantizers_reg.cpp
//
// Register half of the signal quantizers builtins: the CallContext wrappers
// delegating to the engine-free compute in quantizers.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/signal/digital_filtering/quantizers.hpp>

#include <numkit/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::signal {

namespace detail {

void uencode_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "uencode");
    // Marshalling only — the [2, 32] range contract lives in the
    // uencode compute (nk:uencode:BadN).
    const int N = static_cast<int>(argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "N", "uencode"), 2, "N", "uencode"));
    double V = 1.0;
    if (args.size() >= 3 && !args[2].isEmpty())
        V = argcheck::expectedReal(args[2], 3, "V", "uencode");
    bool signedOut = false;
    if (args.size() >= 4 && !args[3].isEmpty()) {
        std::string s = args[3].toString();
        std::transform(s.begin(), s.end(), s.begin(),
                        [](unsigned char c) { return std::tolower(c); });
        if (s == "signed")        signedOut = true;
        else if (s == "unsigned") signedOut = false;
        else throw Error("uencode: 4th arg must be 'signed' or 'unsigned'",
                          0, 0, "uencode", "", "nk:uencode:Polarity");
    }
    outs[0] = uencode(args[0], N, V, signedOut, ctx.engine->resource());
}

void udecode_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "udecode");
    // Marshalling only: the [2, 32] range and V positivity contracts
    // live in udecode_argcheck (compute side).
    const int N = static_cast<int>(argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "N", "udecode"), 2, "N", "udecode"));
    double V = 1.0;
    if (args.size() >= 3 && !args[2].isEmpty())
        V = argcheck::expectedReal(args[2], 3, "V", "udecode");
    bool wrap = false;
    if (args.size() >= 4 && !args[3].isEmpty()) {
        std::string s = args[3].toString();
        std::transform(s.begin(), s.end(), s.begin(),
                        [](unsigned char c) { return std::tolower(c); });
        if (s == "wrap")          wrap = true;
        else if (s == "saturate") wrap = false;
        else throw Error("udecode: 4th arg must be 'saturate' or 'wrap'",
                          0, 0, "udecode", "", "nk:udecode:BadOpt");
    }
    outs[0] = udecode(args[0], N, V, wrap, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::signal
