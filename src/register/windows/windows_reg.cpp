// toolboxes/signal/src/windows/windows_reg.cpp
//
// Register half of the signal windows builtins: the CallContext wrappers
// delegating to the engine-free compute in windows.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/signal/windows/windows.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::signal {

namespace detail {

namespace {

// Detect MATLAB's `'symmetric'` (default) / `'periodic'` window flag at
// the trailing argument position. Returns the effective positional arg
// count; sets `periodic = true` for the periodic form.
keel::usize parseSflag(keel::ArrayRef<Value> args, bool &periodic)
{
    periodic = false;
    if (args.empty()) return 0;
    const Value &last = args[args.size() - 1];
    if (!last.isChar() && !last.isString()) return args.size();
    keel::String s = last.toString();
    s.to_ascii_lower_in_place();
    if (s == "symmetric") return args.size() - 1;
    if (s == "periodic") { periodic = true; return args.size() - 1; }
    return args.size();   // not an sflag â€” leave for caller
}

// Periodic-form trick: for any window function f(N) computing the
// symmetric variant, the MATLAB periodic variant is the first N samples
// of f(N+1). This avoids modifying every window's implementation.
template <typename Fn>
Value applySflag(keel::usize N, bool periodic, Fn impl, keel::Allocator *mr)
{
    if (!periodic) return impl(N);
    Value full = impl(N + 1);
    Value out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    if (N > 0)
        std::copy(full.doubleData(), full.doubleData() + N, out.doubleDataMut());
    return out;
}

// Some windows (bartlett, triang, parzenwin, bohmanwin, barthannwin,
// rectwin) accept ONLY a `typeName` flag ('double' / 'single') â€” they
// reject 'periodic' explicitly with the documented MATLAB error.
// `single` output cast is currently a no-op: nk emits double
// regardless of the requested typeName.
void parseTypeNameOnly(keel::ArrayRef<Value> args, const char *fn)
{
    if (args.size() < 2) return;
    const Value &last = args[args.size() - 1];
    if (!last.isChar() && !last.isString()) return;
    keel::String s = last.toString();
    s.to_ascii_lower_in_place();
    if (s != "double" && s != "single")
        throw Error(keel::format("{}: Expected TYPENAME to match one of these values: 'double', 'single' (got '{}')", fn, s),
                    0, 0, fn, "", keel::format("numkit:{}:typeName", fn));
}

} // anonymous

void hamming_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "hamming");
    bool periodic = false; (void)parseSflag(args, periodic);
    auto *mr = ctx.engine->allocator();
    outs[0] = applySflag(argcheck::roundLength(args[0], 1, "N", "hamming"), periodic, [&](keel::usize M){ return hamming(M, mr); }, mr);
}

void hann_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "hann");  // alias hann,hanning: union of probed bounds
    bool periodic = false; (void)parseSflag(args, periodic);
    auto *mr = ctx.engine->allocator();
    outs[0] = applySflag(argcheck::roundLength(args[0], 1, "N", "hann"), periodic, [&](keel::usize M){ return hann(M, mr); }, mr);
}

void blackman_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "blackman");
    bool periodic = false; (void)parseSflag(args, periodic);
    auto *mr = ctx.engine->allocator();
    outs[0] = applySflag(argcheck::roundLength(args[0], 1, "N", "blackman"), periodic, [&](keel::usize M){ return blackman(M, mr); }, mr);
}

void kaiser_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "kaiser");
    const keel::usize N = argcheck::roundLength(args[0], 1, "N", "kaiser");
    const keel::f64 beta = (args.size() >= 2) ? args[1].toScalar() : 0.5;
    outs[0] = kaiser(N, beta, ctx.engine->allocator());
}

void rectwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "rectwin");
    parseTypeNameOnly(args, "rectwin");
    outs[0] = rectwin(argcheck::roundLength(args[0], 1, "N", "rectwin"), ctx.engine->allocator());
}

void bartlett_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "bartlett");
    parseTypeNameOnly(args, "bartlett");
    outs[0] = bartlett(argcheck::roundLength(args[0], 1, "N", "bartlett"), ctx.engine->allocator());
}

// Local helper: extract N from arg[0] with a `name` for error messages.
static keel::usize windowN(const Value &a, const char *name)
{
    return argcheck::roundLength(a, 1, "N", name);
}

void triang_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "triang");
    parseTypeNameOnly(args, "triang");
    outs[0] = triang(windowN(args[0], "triang"), ctx.engine->allocator());
}

void tukeywin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "tukeywin");
    const keel::usize N = windowN(args[0], "tukeywin");
    const keel::f64 r = (args.size() >= 2) ? args[1].toScalar() : 0.5;
    outs[0] = tukeywin(N, r, ctx.engine->allocator());
}

void flattopwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "flattopwin");
    bool periodic = false; (void)parseSflag(args, periodic);
    auto *mr = ctx.engine->allocator();
    outs[0] = applySflag(windowN(args[0], "flattopwin"), periodic, [&](keel::usize M){ return flattopwin(M, mr); }, mr);
}

void gausswin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "gausswin");
    const keel::usize N = windowN(args[0], "gausswin");
    const keel::f64 alpha = (args.size() >= 2) ? args[1].toScalar() : 2.5;
    outs[0] = gausswin(N, alpha, ctx.engine->allocator());
}

void chebwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "chebwin");
    const keel::usize N = windowN(args[0], "chebwin");
    const keel::f64 at = (args.size() >= 2) ? args[1].toScalar() : 100.0;
    outs[0] = chebwin(N, at, ctx.engine->allocator());
}

void parzenwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "parzenwin");
    parseTypeNameOnly(args, "parzenwin");
    outs[0] = parzenwin(windowN(args[0], "parzenwin"), ctx.engine->allocator());
}

void nuttallwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "nuttallwin");
    bool periodic = false; (void)parseSflag(args, periodic);
    auto *mr = ctx.engine->allocator();
    outs[0] = applySflag(windowN(args[0], "nuttallwin"), periodic, [&](keel::usize M){ return nuttallwin(M, mr); }, mr);
}

void taylorwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "taylorwin");
    const keel::usize N = windowN(args[0], "taylorwin");
    const keel::i32 nbar = (args.size() >= 2)
        ? argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[1], 2, "nbar", "taylorwin"), 2, "nbar", "taylorwin")
        : 4;
    const keel::f64 sll = (args.size() >= 3) ? args[2].toScalar() : -30.0;
    outs[0] = taylorwin(N, nbar, sll, ctx.engine->allocator());
}

void blackmanharris_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "blackmanharris");
    bool periodic = false; (void)parseSflag(args, periodic);
    auto *mr = ctx.engine->allocator();
    outs[0] = applySflag(windowN(args[0], "blackmanharris"), periodic, [&](keel::usize M){ return blackmanharris(M, mr); }, mr);
}

void bohmanwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "bohmanwin");
    parseTypeNameOnly(args, "bohmanwin");
    outs[0] = bohmanwin(windowN(args[0], "bohmanwin"), ctx.engine->allocator());
}

void barthannwin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "barthannwin");
    parseTypeNameOnly(args, "barthannwin");
    outs[0] = barthannwin(windowN(args[0], "barthannwin"), ctx.engine->allocator());
}

void dpss_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "dpss");
    const keel::usize N = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[0], 1, "N", "dpss"), 1, "N", "dpss"));
    const keel::f64 nw = args[1].toScalar();
    keel::usize k = std::min(N, std::max<keel::usize>(1, keel::as<keel::usize>(keel::round_i64(2.0 * nw))));
    if (args.size() >= 3 && args[2].isNumeric() && !args[2].isEmpty()) {
        k = std::min(N, keel::as<keel::usize>(std::max(1.0, keel::round(args[2].elemAsDouble(args[2].numel() - 1)))));
    }
    auto [E, V] = dpss(N, nw, k, ctx.engine->allocator());
    outs[0] = keel::move(E);
    if (nargout >= 2) outs[1] = keel::move(V);
}

} // namespace detail

} // namespace numkit::signal
