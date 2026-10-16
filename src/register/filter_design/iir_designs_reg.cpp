// toolboxes/signal/src/filter_design/iir_designs_reg.cpp
//
// CallContext register half of filter_design/iir_designs.cpp (Phase 2b compute/register split).
// The detail{{}} *_reg fns were interleaved with compute in the original
// TU; collected here verbatim (reg-side arg parsers nested in their anon
// namespaces ride along). See project_layering_refactor.
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/internal/shims/polyfun.hpp>
#include <numkit/signal/internal/shims/specfun.hpp>
#include <numkit/signal/filter_design/analog_filters.hpp>
#include <numkit/signal/filter_design/iir_designs.hpp>
#include <numkit/signal/filter_implementation/conversions_extras.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::signal {

namespace detail {

namespace {
inline FilterType parseFtype(const Value &v) {
    if (!v.isChar() && !v.isString()) return FilterType::Lowpass;
    auto s = v.toString();
    // Case-insensitive compare
    auto eq = [&](const char *t) {
        if (s.size() != std::strlen(t)) return false;
        for (size_t i = 0; i < s.size(); ++i)
            if (std::tolower((unsigned char)s[i]) != (unsigned char)t[i]) return false;
        return true;
    };
    if (eq("low"))      return FilterType::Lowpass;
    if (eq("high"))     return FilterType::Highpass;
    if (eq("bandpass")) return FilterType::Bandpass;
    if (eq("stop"))     return FilterType::Bandstop;
    return FilterType::Lowpass;
}

inline bool isAnalogFlag(const Value &v) {
    if (!v.isChar() && !v.isString()) return false;
    auto s = v.toString();
    return s.size() == 1 && (s[0] == 's' || s[0] == 'S');
}

// Parse trailing string args: any combination of ftype string + 's' analog
// flag, in any order.
struct Trailing {
    FilterType ftype{FilterType::Lowpass};
    bool ftype_set{false};
    bool analog{false};
};

inline Trailing parseTrailing(Span<const Value> args, size_t start) {
    Trailing t;
    for (size_t i = start; i < args.size(); ++i) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        if (isAnalogFlag(args[i])) t.analog = true;
        else                       { t.ftype = parseFtype(args[i]); t.ftype_set = true; }
    }
    return t;
}

inline FilterType defaultFtype(const Value &Wn, FilterType picked, bool was_set) {
    if (was_set) return picked;
    return Wn.numel() == 2 ? FilterType::Bandpass : FilterType::Lowpass;
}
} // anonymous

namespace {

// IIR-family order extraction â€” the Value-boundary half of the order
// contract (positive lives in each compute's first-line
// iirOrder_argcheck). Note the family's own rejection classes: a
// complex order is rejected as not-real, a fractional one as
// not-integer (probed 2026-09-17).
int iirOrder(const Value &v, const char *fn)
{
    if (!v.isScalar()) {
        const Dims &d = v.dims();
        throw Error(std::string(fn) + ": filter order N must be a scalar "
                    "(argument is " + std::to_string(d.dim(0)) + "x"
                        + std::to_string(d.dim(1)) + ").",
                    0, 0, fn, "", std::string("numkit:") + fn + ":orderNotScalar");
    }
    if (v.isComplex())
        throw Error(std::string(fn) + ": filter order N must be real "
                        "(argument is complex).",
                    0, 0, fn, "", std::string("numkit:") + fn + ":orderNotReal");
    const double n = v.toScalar();
    if (n != std::floor(n))
        throw Error(std::string(fn) + ": filter order N must be a positive "
                        "integer (got " + argcheck::detail::num(n) + ").",
                    0, 0, fn, "", std::string("numkit:") + fn + ":orderNotInteger");
    return static_cast<int>(n);
}

} // namespace

void cheby1_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "cheby1");
    const int N = iirOrder(args[0], "cheby1");
    const double R = args[1].toScalar();
    const Value &Wn = args[2];
    auto t = parseTrailing(args, 3);
    auto ftype = defaultFtype(Wn, t.ftype, t.ftype_set);
    auto [b, a] = cheby1(N, R, Wn, ftype, t.analog, ctx.engine->resource());
    if (outs.size() >= 3) {   // [z, p, k] = cheby1(...): digital ZPK via tf2zpk
        auto [z, p, k] = tf2zpk(b, a, ctx.engine->resource());
        outs[0] = std::move(z); outs[1] = std::move(p); outs[2] = Value::scalar(k, ctx.engine->resource());
        return;
    }
    outs[0] = std::move(b);
    if (outs.size() > 1) outs[1] = std::move(a);
}

void cheby2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "cheby2");
    const int N = iirOrder(args[0], "cheby2");
    const double R = args[1].toScalar();
    const Value &Wn = args[2];
    auto t = parseTrailing(args, 3);
    auto ftype = defaultFtype(Wn, t.ftype, t.ftype_set);
    auto [b, a] = cheby2(N, R, Wn, ftype, t.analog, ctx.engine->resource());
    if (outs.size() >= 3) {   // [z, p, k] = cheby2(...): digital ZPK via tf2zpk
        auto [z, p, k] = tf2zpk(b, a, ctx.engine->resource());
        outs[0] = std::move(z); outs[1] = std::move(p); outs[2] = Value::scalar(k, ctx.engine->resource());
        return;
    }
    outs[0] = std::move(b);
    if (outs.size() > 1) outs[1] = std::move(a);
}

void besself_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "besself");
    const int N = iirOrder(args[0], "besself");
    const Value &Wn = args[1];
    auto t = parseTrailing(args, 2);
    auto ftype = defaultFtype(Wn, t.ftype, t.ftype_set);
    // besself is ALWAYS analog â€” MATLAB has no digital Bessel filter (the
    // bilinear transform destroys the maximally-flat group delay). The 's'
    // flag is therefore redundant; ignore the digital path entirely.
    (void)t.analog;
    auto [b, a] = besself(N, Wn, ftype, /*analog=*/true, ctx.engine->resource());
    outs[0] = std::move(b);
    if (outs.size() > 1) outs[1] = std::move(a);
}

void ellip_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 7, "ellip");
    const int N = iirOrder(args[0], "ellip");
    const double Rp  = args[1].toScalar();
    const double Rs  = args[2].toScalar();
    const Value &Wn  = args[3];
    auto t = parseTrailing(args, 4);
    auto ftype = defaultFtype(Wn, t.ftype, t.ftype_set);
    auto [b, a] = ellip(N, Rp, Rs, Wn, ftype, t.analog, ctx.engine->resource());
    if (outs.size() >= 3) {   // [z, p, k] = ellip(...): digital ZPK via tf2zpk
        auto [z, p, k] = tf2zpk(b, a, ctx.engine->resource());
        outs[0] = std::move(z); outs[1] = std::move(p); outs[2] = Value::scalar(k, ctx.engine->resource());
        return;
    }
    outs[0] = std::move(b);
    if (outs.size() > 1) outs[1] = std::move(a);
}

namespace {
inline bool parseAnalogFromOrdArgs(Span<const Value> args, size_t start) {
    for (size_t i = start; i < args.size(); ++i)
        if ((args[i].isChar() || args[i].isString()) && isAnalogFlag(args[i]))
            return true;
    return false;
}
}

#define NUMKIT_ORD_REG(name)                                                          \
    void name##_reg(Span<const Value> args, size_t nargout,                       \
                    Span<Value> outs, CallContext &ctx)                           \
    {                                                                              \
        argcheck::checkNargin(args.size(), 4, 5, #name);                          \
        const Value &Wp = args[0];                                                \
        const Value &Ws = args[1];                                                \
        const double Rp = args[2].toScalar();                                     \
        const double Rs = args[3].toScalar();                                     \
        const bool analog = parseAnalogFromOrdArgs(args, 4);                      \
        auto [N, Wn] = name(Wp, Ws, Rp, Rs, analog, ctx.engine->resource());     \
        outs[0] = Value::scalar(static_cast<double>(N), ctx.engine->resource()); \
        if (nargout > 1) outs[1] = std::move(Wn);                                 \
    }

NUMKIT_ORD_REG(buttord)
NUMKIT_ORD_REG(cheb1ord)
NUMKIT_ORD_REG(cheb2ord)

#undef NUMKIT_ORD_REG

} // namespace detail

namespace detail {

void ellipord_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "ellipord");
    bool analog = false;
    if (args.size() >= 5) {
        std::string s = args[4].toString();
        analog = (s == "s" || s == "S");
    }
    auto [N, Wn] = ellipord(args[0], args[1], args[2].toScalar(), args[3].toScalar(), analog, ctx.engine->resource());
    outs[0] = Value::scalar(static_cast<double>(N), ctx.engine->resource());
    if (nargout > 1) outs[1] = std::move(Wn);
}

} // namespace detail

namespace detail {

void firpmord_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "firpmord");
    double fs = 2.0;
    if (args.size() >= 4 && !args[3].isEmpty()) fs = args[3].toScalar();
    auto [N, ff, aa, wts] = firpmord(args[0], args[1], args[2], fs, ctx.engine->resource());
    outs[0] = Value::scalar(static_cast<double>(N), ctx.engine->resource());
    if (nargout >= 2 && outs.size() >= 2) outs[1] = std::move(ff);
    if (nargout >= 3 && outs.size() >= 3) outs[2] = std::move(aa);
    if (nargout >= 4 && outs.size() >= 4) outs[3] = std::move(wts);
}

} // namespace detail

namespace detail {

void kaiserord_reg(Span<const Value> args, size_t nargout,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "kaiserord");
    double fs = 2.0;
    if (args.size() >= 4 && !args[3].isEmpty()) fs = args[3].toScalar();
    auto [N, Wn, beta, ftype] = kaiserord(args[0], args[1], args[2], fs, ctx.engine->resource());
    auto *mr = ctx.engine->resource();
    outs[0] = Value::scalar(static_cast<double>(N), mr);
    if (nargout >= 2 && outs.size() >= 2) outs[1] = std::move(Wn);
    if (nargout >= 3 && outs.size() >= 3) outs[2] = Value::scalar(beta, mr);
    if (nargout >= 4 && outs.size() >= 4) {
        Value f = Value::matrix(1, ftype.size(), ValueType::CHAR, mr);
        char *cd = f.charDataMut();
        for (size_t i = 0; i < ftype.size(); ++i) cd[i] = ftype[i];
        outs[3] = std::move(f);
    }
}


void butter_reg(Span<const Value> args, size_t nargout, Span<Value> outs, CallContext &ctx)
{
    // butter(N, Wn[, ftype][, 's']) â€” routed through the shared analog/
    // digital pipeline (bugs/closed/signal/butter-analog-flag-wn-domain.md).
    argcheck::checkNargin(args.size(), 2, 5, "butter");
    const int N = iirOrder(args[0], "butter");
    const Value &Wn = args[1];
    auto t = parseTrailing(args, 2);
    auto ftype = defaultFtype(Wn, t.ftype, t.ftype_set);
    auto [b, a] = butter(N, Wn, ftype, t.analog, ctx.engine->resource());
    if (nargout >= 3) {   // [z, p, k]: ZPK via tf2zpk (order may differ
        auto [z, p, k] = tf2zpk(b, a, ctx.engine->resource()); // from MATLAB)
        outs[0] = std::move(z); outs[1] = std::move(p);
        outs[2] = Value::scalar(k, ctx.engine->resource());
        return;
    }
    outs[0] = std::move(b);
    if (nargout > 1) outs[1] = std::move(a);
}

} // namespace detail

} // namespace numkit::signal
