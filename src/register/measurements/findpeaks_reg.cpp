// toolboxes/signal/src/measurements/findpeaks_reg.cpp
//
// CallContext register half of measurements/findpeaks.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/utf8.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/measurements/findpeaks.hpp>
#include "measurements/findpeaks_detail.hpp"
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

void findpeaks_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                   CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "findpeaks");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    const Value &y = args[0];

    if (nargout > 4)
        throw Error("findpeaks: at most 4 outputs (pks, locs, w, p)",
                     0, 0, "findpeaks", "", "numkit:findpeaks:nargout");

    PeakOpts opt;
    keel::ArenaTemp locScratch = keel::get_scratch(mr);
    auto locStore = keel::Array<keel::f64>(locScratch.allocator()); // for the Fs form
    const keel::f64 *locArr = nullptr;

    keel::usize argi = 1;

    // Optional positional X-vector or Fs-scalar (before any Name-Value pair).
    if (args.size() > 1 && !args[1].isChar() && !args[1].isString()) {
        const Value &second = args[1];
        const keel::usize n = y.numel();
        if (second.numel() == 1) {
            // Sampling frequency: location(i) = i / Fs  (0-based sample index).
            const keel::f64 fs = second.toScalar();
            if (!(fs > 0.0))
                throw Error("findpeaks: sample rate Fs must be positive",
                             0, 0, "findpeaks", "", "numkit:findpeaks:badFs");
            locStore.assign(n, 0.0);
            for (keel::usize i = 0; i < n; ++i)
                locStore[i] = keel::as<keel::f64>(i) / fs;
            locArr = locStore.data();
        } else {
            if (second.numel() != n)
                throw Error("findpeaks: X must have the same length as Y",
                             0, 0, "findpeaks", "", "numkit:findpeaks:badX");
            locArr = second.doubleData();
        }
        argi = 2;
    }

    // Name-Value options.
    for (; argi < args.size(); argi += 2) {
        if (!args[argi].isChar() && !args[argi].isString())
            throw Error("findpeaks: expected an option name",
                         0, 0, "findpeaks", "", "numkit:findpeaks:badOpt");
        keel::String rawName = args[argi].toString();
        keel::String name = rawName.clone();
        name.to_ascii_lower_in_place();
        if (argi + 1 >= args.size())
            throw Error(keel::format("findpeaks: option '{}' needs a value", rawName),
                         0, 0, "findpeaks", "", "numkit:findpeaks:badOpt");
        const Value &v = args[argi + 1];

        if (name == "minpeakheight") {
            opt.minHeight = v.toScalar();
        } else if (name == "threshold") {
            opt.threshold = v.toScalar();
        } else if (name == "minpeakdistance") {
            opt.minDist = v.toScalar();
            opt.useMinDist = true;
        } else if (name == "minpeakprominence") {
            opt.minProm = v.toScalar();
            opt.useMinProm = true;
        } else if (name == "npeaks") {
            opt.nPeaks = keel::as<keel::i64>(argcheck::expectedPositiveInteger(
                argcheck::expectedReal(v, argi ? keel::as<keel::i32>(argi) + 2 : 2,
                                     "NPeaks", "findpeaks"),
                argi ? keel::as<keel::i32>(argi) + 2 : 2, "NPeaks", "findpeaks"));
        } else if (name == "sortstr") {
            keel::String s = v.toString();
            s.to_ascii_lower_in_place();
            if      (s == "none")    opt.sortMode = 0;
            else if (s == "ascend")  opt.sortMode = 1;
            else if (s == "descend") opt.sortMode = 2;
            else
                throw Error("findpeaks: SortStr must be 'none'/'ascend'/'descend'",
                             0, 0, "findpeaks", "", "numkit:findpeaks:badOpt");
        } else if (name == "widthreference") {
            opt.widthRef = keel::as<keel::i32>(argcheck::expectedChoice(
                v, {"halfprom", "halfheight"}, keel::as<keel::i32>(argi) + 2,
                "WidthReference", "findpeaks"));
        } else if (name == "minpeakwidth") {
            opt.minWidth = v.toScalar();
            opt.useWidthFilter = true;
        } else if (name == "maxpeakwidth") {
            opt.maxWidth = v.toScalar();
            opt.useWidthFilter = true;
        } else if (name == "annotate") {
            (void)argcheck::expectedChoice(
                v, {"peaks", "extents"}, keel::as<keel::i32>(argi) + 2,
                "Annotate", "findpeaks");
        } else {
            throw Error(keel::format("findpeaks: option '{}' not supported", rawName),
                         0, 0, "findpeaks", "", "numkit:findpeaks:unsupportedOpt");
        }
    }

    auto [vals, idxs, wids, proms] = findpeaksImpl(y, locArr, opt, mr);
    outs[0] = std::move(vals);
    if (nargout > 1) outs[1] = std::move(idxs);
    if (nargout > 2) outs[2] = std::move(wids);
    if (nargout > 3) outs[3] = std::move(proms);
}

} // namespace detail

} // namespace numkit::signal
