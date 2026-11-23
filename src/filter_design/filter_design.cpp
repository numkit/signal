// toolboxes/signal/src/filter_design/filter_design.cpp
//
// Butterworth IIR (butter) and FIR filter design (fir1 / fir2 / firls / firpm),
// delegating numerical kernels to keel::signal, plus cell2sos marshaling.

#include <keel/arena.h>
#include <keel/tuple.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/signal/filter_design/filter_design.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>

namespace numkit::signal {

namespace {

keel::Array<keel::f64> valueToDoubleRow(const Value &v, const keel::ArenaTemp &arena)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> dst(n, arena.allocator());
    for (keel::usize i = 0; i < n; ++i) dst[i] = v.elemAsDouble(i);
    return dst;
}

void fir1_argcheck(keel::i32 N, keel::f64 Wn, keel::StringRef type)
{
    argcheck::expectedPositive(keel::as<keel::f64>(N), 1, "n", "fir1");
    argcheck::expectedPositive(Wn, 2, "Wn", "fir1");
    if (Wn >= 1.0) {
        const auto numStr = argcheck::detail::num(Wn);
        throw Error(keel::format("fir1: cutoff frequency Wn must be inside the open interval (0, 1) (got {}).", numStr),
                    0, 0, "fir1", "", "numkit:fir1:wnOutOfRange");
    }
    if (type != "low" && type != "high")
        throw Error(keel::format("fir1: type must be one of: 'low', 'high' (got '{}').", type),
                    0, 0, "fir1", "", "numkit:fir1:invalidType");
}

} // anonymous namespace

keel::Tuple<Value, Value>
butter(keel::i32 N, keel::f64 Wn, keel::StringRef type, keel::Allocator *mr)
{
    if (Wn <= 0.0 || Wn >= 1.0)
        throw Error("butter: Wn must be between 0 and 1",
                    0, 0, "butter", "", "numkit:butter:badWn");
    if (type != "low" && type != "high")
        throw Error("butter: type must be 'low' or 'high'",
                    0, 0, "butter", "", "numkit:butter:badType");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::f64 wn_arr[1] = {Wn};
    const auto ftype = (type == "low") ? keel::signal::FilterType::kLowpass
                                       : keel::signal::FilterType::kHighpass;
    keel::signal::TfResult tf{
        keel::Array<keel::f64>(scratch.allocator()),
        keel::Array<keel::f64>(scratch.allocator()),
    };
    (void)keel::signal::butter(N, keel::ArrayRef<keel::f64>(wn_arr, 1), ftype, false, tf,
                               scratch.allocator());

    Value bv = Value::matrix(1, tf.b.size(), ValueType::DOUBLE, mr);
    Value av = Value::matrix(1, tf.a.size(), ValueType::DOUBLE, mr);
    keel::copy_n(bv.doubleDataMut(), tf.b.data(), tf.b.size());
    keel::copy_n(av.doubleDataMut(), tf.a.data(), tf.a.size());
    return keel::make_tuple(keel::move(bv), keel::move(av));
}

Value firls(keel::i32 N, const Value &Farg, const Value &Aarg,
            const Value &Warg, keel::StringRef ftype,
            keel::Allocator *mr)
{
    if (N < 1)
        throw Error("firls: filter order must be a positive integer",
                    0, 0, "firls", "", "numkit:firls:badOrder");

    keel::signal::FirFtype kftype = keel::signal::FirFtype::kStandard;
    if (!ftype.empty()) {
        keel::String fl;
        fl.reserve(ftype.size());
        for (char c : ftype) fl.push_back(keel::to_ascii_lower(c));
        if (fl == "hilbert" || fl == "h")             kftype = keel::signal::FirFtype::kHilbert;
        else if (fl == "differentiator" || fl == "d") kftype = keel::signal::FirFtype::kDifferentiator;
        else
            throw Error("firls: ftype must be 'hilbert' or 'differentiator'",
                        0, 0, "firls", "", "numkit:firls:badFtype");
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto Fv = valueToDoubleRow(Farg, scratch);
    auto Av = valueToDoubleRow(Aarg, scratch);
    const keel::usize Fn = Fv.size();
    const keel::usize An = Av.size();

    if (Fn == 0 || Fn != An)
        throw Error("firls: F and A must have the same non-empty length",
                    0, 0, "firls", "", "numkit:firls:badLen");
    if ((Fn % 2) != 0)
        throw Error("firls: F must have even length (band-edge pairs)",
                    0, 0, "firls", "", "numkit:firls:badLen");
    for (keel::usize i = 0; i < Fn; ++i)
        if (Fv[i] < 0.0 || Fv[i] > 1.0)
            throw Error("firls: F values must be in [0, 1]",
                        0, 0, "firls", "", "numkit:firls:badF");
    for (keel::usize i = 1; i < Fn; ++i)
        if (Fv[i] < Fv[i - 1])
            throw Error("firls: F must be non-decreasing",
                        0, 0, "firls", "", "numkit:firls:badF");

    const keel::usize numBands = Fn / 2;
    keel::Array<keel::f64> Wv(scratch.allocator());
    if (!Warg.isEmpty()) {
        if (Warg.numel() != numBands)
            throw Error("firls: W must have one weight per frequency band",
                        0, 0, "firls", "", "numkit:firls:badWLen");
        Wv.resize(numBands);
        for (keel::usize k = 0; k < numBands; ++k) {
            const keel::f64 wk = Warg.elemAsDouble(k);
            if (!(wk > 0.0) || !keel::is_finite(wk))
                throw Error("firls: weights W must be positive and finite",
                            0, 0, "firls", "", "numkit:firls:badW");
            Wv[k] = wk;
        }
    }

    const keel::usize len = keel::as<keel::usize>(N + 1);
    Value bv = Value::matrix(1, len, ValueType::DOUBLE, mr);
    const auto st = keel::signal::firls(
        N, Fv.as_ref(), Av.as_ref(), Wv.as_ref(), kftype,
        keel::MutArrayRef<keel::f64>(bv.doubleDataMut(), len),
        scratch.allocator());
    if (st != keel::signal::SignalStatus::kOk)
        throw Error("firls: Q matrix is not positive-definite",
                    0, 0, "firls", "", "numkit:firls:singular");
    return bv;
}

Value firls(keel::i32 N, const Value &Farg, const Value &Aarg, keel::Allocator *mr)
{
    return firls(N, Farg, Aarg, Value::Empty, "", mr);
}

Value fir1(keel::i32 N, keel::f64 Wn, keel::StringRef type, keel::Allocator *mr)
{
    fir1_argcheck(N, Wn, type);
    const keel::usize filtLen = keel::as<keel::usize>(N + 1);
    Value bv = Value::matrix(1, filtLen, ValueType::DOUBLE, mr);
    (void)keel::signal::fir1_single(N, Wn, type == "high",
                                    keel::MutArrayRef<keel::f64>(bv.doubleDataMut(), filtLen));
    return bv;
}

Value fir1(keel::i32 N, const Value &Wn, keel::StringRef typeIn,
           const Value &window, bool scale, keel::Allocator *mr)
{
    argcheck::expectedPositive(keel::as<keel::f64>(N), 1, "n", "fir1");
    if (!Wn.isNumeric() || Wn.isComplex() || Wn.isEmpty() || !Wn.dims().isVector())
        throw Error("fir1: cutoff frequency Wn must be a real non-empty vector.",
                    0, 0, "fir1", "", "numkit:fir1:invalidWn");

    const keel::usize nw = Wn.numel();
    keel::String type = keel::String::from(typeIn);
    type.to_ascii_lower_in_place();
    if (type.empty())
        type.assign((nw == 1) ? "low" : (nw == 2 ? "bandpass" : "dc-0"));

    if (nw == 1 && window.isEmpty() && scale && (type == "low" || type == "high"))
        return fir1(N, Wn.elemAsDouble(0), type, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> w(nw, scratch.allocator());
    for (keel::usize k = 0; k < nw; ++k) {
        w[k] = Wn.elemAsDouble(k);
        if (!(w[k] > 0.0 && w[k] < 1.0))
            throw Error("fir1: cutoff frequencies Wn must be inside (0, 1).",
                        0, 0, "fir1", "", "numkit:fir1:wnOutOfRange");
        if (k > 0 && !(w[k] > w[k - 1]))
            throw Error("fir1: cutoff frequencies Wn must be strictly increasing.",
                        0, 0, "fir1", "", "numkit:fir1:wnNotIncreasing");
    }

    bool firstBandPass = false;
    if (nw == 1) {
        if (type == "low") firstBandPass = true;
        else if (type == "high") firstBandPass = false;
        else throw Error(keel::format("fir1: invalid filter type '{}'.", typeIn), 0, 0, "fir1", "", "numkit:fir1:invalidType");
    } else {
        if (type == "bandpass" || type == "dc-0") firstBandPass = false;
        else if (type == "stop" || type == "dc-1") firstBandPass = true;
        else throw Error(keel::format("fir1: invalid filter type '{}'.", typeIn), 0, 0, "fir1", "", "numkit:fir1:invalidType");
    }

    const bool passAtNyquist = ((nw % 2 == 1) ? !firstBandPass : firstBandPass);
    const keel::i32 effN = (passAtNyquist && (N % 2 == 1) && window.isEmpty()) ? (N + 1) : N;
    const keel::usize filtLen = keel::as<keel::usize>(effN + 1);
    keel::Array<keel::f64> winVec(scratch.allocator());
    if (!window.isEmpty()) {
        if (window.numel() != filtLen)
            throw Error("fir1: window length must equal N + 1.",
                        0, 0, "fir1", "", "numkit:fir1:badWindowLength");
        winVec = valueToDoubleRow(window, scratch);
    }

    keel::Array<keel::f64> h(scratch.allocator());
    (void)keel::signal::fir1_multiband(N, w.as_ref(), firstBandPass, winVec.as_ref(), scale, h,
                                       scratch.allocator());
    Value bv = Value::matrix(1, h.size(), ValueType::DOUBLE, mr);
    keel::copy_n(bv.doubleDataMut(), h.data(), h.size());
    return bv;
}

keel::Tuple<Value, Value>
cell2sos(const Value &C, keel::Allocator *mr)
{
    if (!C.isCell())
        throw Error("cell2sos: input must be a cell array",
                    0, 0, "cell2sos", "", "numkit:cell2sos:NotCell");
    const keel::usize L = C.numel();
    if (L == 0) {
        Value empty = Value::matrix(0, 6, ValueType::DOUBLE, mr);
        return {empty, Value::scalar(1.0, mr)};
    }

    keel::f64 g = 1.0;
    keel::usize startIdx = 0;
    {
        const Value &first = C.cellAt(0);
        if (first.isCell() && first.numel() == 2) {
            const Value &b0 = first.cellAt(0);
            const Value &a0 = first.cellAt(1);
            if (b0.numel() == 1 && a0.numel() == 1) {
                const keel::f64 bv = b0.toScalar();
                const keel::f64 av = a0.toScalar();
                if (av != 0.0) {
                    g = bv / av;
                    startIdx = 1;
                }
            }
        }
    }

    const keel::usize outRows = L - startIdx;
    Value S = Value::matrix(outRows, 6, ValueType::DOUBLE, mr);
    if (outRows == 0) return {S, Value::scalar(g, mr)};

    keel::f64 *Sd = S.doubleDataMut();
    std::fill(Sd, Sd + outRows * 6, 0.0);
    for (keel::usize i = 0; i < outRows; ++i) {
        const Value &pair = C.cellAt(startIdx + i);
        if (!pair.isCell() || pair.numel() != 2)
            throw Error("cell2sos: each cell must be {B, A} pair",
                        0, 0, "cell2sos", "", "numkit:cell2sos:BadPair");
        const Value &bv = pair.cellAt(0);
        const Value &av = pair.cellAt(1);
        if (bv.numel() > 3 || av.numel() > 3)
            throw Error("cell2sos: each B/A must have at most 3 elements",
                        0, 0, "cell2sos", "", "numkit:cell2sos:OrderTooHigh");
        for (keel::usize k = 0; k < bv.numel(); ++k)
            Sd[i + k * outRows] = bv.elemAsDouble(k);
        for (keel::usize k = 0; k < av.numel(); ++k)
            Sd[i + (3 + k) * outRows] = av.elemAsDouble(k);
    }
    return {S, Value::scalar(g, mr)};
}

Value fir2(keel::i32 n, const Value &f, const Value &m,
           const Fir2Options &opts, keel::Allocator *mr)
{
    if (n <= 0)
        throw numkit::Error("n must be a positive integer.", 0, 0, "fir2", "",
                            "numkit:fir2:BadN");

    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto fv = valueToDoubleRow(f, arena);
    auto mv = valueToDoubleRow(m, arena);

    if (fv.size() != mv.size())
        throw numkit::Error("The frequency and magnitude vectors must be the same length.",
                            0, 0, "fir2", "", "numkit:fir2:MismatchedDimensions");
    if (fv.size() < 2)
        throw numkit::Error("The frequency vector must have at least 2 elements.",
                            0, 0, "fir2", "", "numkit:fir2:BadFLen");

    const keel::usize nf = fv.size();
    if (fv[0] != 0.0 || fv[nf - 1] != 1.0)
        throw numkit::Error("The first frequency must be 0 and the last 1.",
                            0, 0, "fir2", "", "numkit:fir2:InvalidRange");
    for (keel::usize i = 1; i < nf; ++i) {
        if (fv[i] < fv[i - 1])
            throw numkit::Error("Frequencies must be non-decreasing.", 0, 0,
                                "fir2", "", "numkit:fir2:InvalidFreqVec");
    }

    const keel::i32 effN = ((n % 2) == 1 && mv[nf - 1] != 0.0) ? (n + 1) : n;
    const keel::i32 nn = effN + 1;
    if (opts.npt > 0 && 2 * opts.npt < nn)
        throw numkit::Error("The number of grid points must be greater than or equal to ceil(nn/2).",
                            0, 0, "fir2", "", "numkit:fir2:InvalidNpt");

    keel::Array<keel::f64> winVec(arena.allocator());
    if (!opts.window.isEmpty()) {
        if (keel::as<keel::i32>(opts.window.numel()) != nn)
            throw numkit::Error("The window length must equal the filter length.",
                                0, 0, "fir2", "", "numkit:fir2:BadWindow");
        winVec = valueToDoubleRow(opts.window, arena);
    }

    keel::Array<keel::f64> h(arena.allocator());
    const auto st = keel::signal::fir2(n, fv.as_ref(), mv.as_ref(), opts.npt, opts.lap,
                                       winVec.as_ref(), h, arena.allocator());
    if (st != keel::signal::SignalStatus::kOk)
        throw numkit::Error("Internal grid index out of range.", 0, 0,
                            "fir2", "", "numkit:fir2:SignalErr");

    Value b = Value::matrix(1, h.size(), ValueType::DOUBLE, mr);
    keel::copy_n(b.doubleDataMut(), h.data(), h.size());
    return b;
}

keel::Tuple<Value, keel::f64>
firpm(keel::i32 N, const Value &Farg, const Value &Aarg, const Value &Warg,
      keel::StringRef ftype, keel::Allocator *mr)
{
    if (N < 3)
        throw Error("Filter order must be 3 or more",
                    0, 0, "firpm", "", "numkit:firpm:badOrder");

    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto Fv = valueToDoubleRow(Farg, arena);
    auto Av = valueToDoubleRow(Aarg, arena);
    auto Wv = valueToDoubleRow(Warg, arena);
    const keel::usize Fn = Fv.size();
    const keel::usize An = Av.size();

    if (Fn == 0 || Fn != An || (Fn % 2) != 0)
        throw Error("firpm: F and A must be non-empty equal-length even-length",
                    0, 0, "firpm", "", "numkit:firpm:badLen");
    const keel::usize numBands = Fn / 2;
    if (!Wv.empty() && Wv.size() != numBands)
        throw Error("firpm: W must have one weight per band",
                    0, 0, "firpm", "", "numkit:firpm:badW");
    for (keel::usize i = 0; i < Fn; ++i)
        if (Fv[i] < 0.0 || Fv[i] > 1.0)
            throw Error("firpm: F values must be in [0, 1]",
                        0, 0, "firpm", "", "numkit:firpm:badF");
    for (keel::usize i = 1; i < Fn; ++i)
        if (Fv[i] < Fv[i - 1])
            throw Error("firpm: F must be non-decreasing",
                        0, 0, "firpm", "", "numkit:firpm:badF");

    keel::signal::FirFtype kftype = keel::signal::FirFtype::kStandard;
    if (!ftype.empty()) {
        if (ftype.equals_ignore_ascii_case("hilbert"))
            kftype = keel::signal::FirFtype::kHilbert;
        else if (ftype.equals_ignore_ascii_case("differentiator"))
            kftype = keel::signal::FirFtype::kDifferentiator;
        else
            throw Error("firpm: ftype must be 'hilbert' or 'differentiator'",
                        0, 0, "firpm", "", "numkit:firpm:badFtype");
    }

    keel::f64 totalWidth = 0.0;
    for (keel::usize k = 0; k < numBands; ++k) totalWidth += Fv[2 * k + 1] - Fv[2 * k];
    if (totalWidth <= 0.0)
        throw Error("firpm: all bands have zero width",
                    0, 0, "firpm", "", "numkit:firpm:badF");
    for (keel::usize k = 0; k < Wv.size(); ++k)
        if (Wv[k] <= 0.0)
            throw Error("firpm: weights must be positive",
                        0, 0, "firpm", "", "numkit:firpm:badW");

    const keel::usize len = keel::as<keel::usize>(N + 1);
    Value bOut = Value::matrix(1, len, ValueType::DOUBLE, mr);
    keel::f64 delta = 0.0;
    const auto st = keel::signal::firpm(
        N, Fv.as_ref(), Av.as_ref(), Wv.as_ref(), kftype,
        keel::MutArrayRef<keel::f64>(bOut.doubleDataMut(), len),
        delta, arena.allocator());
    if (st != keel::signal::SignalStatus::kOk)
        throw Error("firpm: dense grid too small for filter order",
                    0, 0, "firpm", "", "numkit:firpm:internal");
    return keel::make_tuple(keel::move(bOut), delta);
}

} // namespace numkit::signal
