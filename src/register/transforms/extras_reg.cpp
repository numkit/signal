// toolboxes/signal/src/transforms/extras_reg.cpp
//
// CallContext register half of transforms/extras.cpp (Phase 2b compute/register split).
// Engine-coupled glue: marshals CallContext args/outs into the engine-free
// compute API declared in the headers below. See project_layering_refactor.
#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/sys/plugin.hpp>
#include <numkit/signal/filter_analysis/unwrap.hpp>      // (used externally; not needed here)
#include <numkit/signal/transforms/extras.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include "transforms/extras_detail.hpp"
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numkit::signal {

namespace detail {

void dftmtx_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "dftmtx");
    outs[0] = dftmtx(keel::as<keel::usize>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[0], 1, "n", "dftmtx"), 1, "n", "dftmtx")),
        ctx.engine->allocator());
}

void bitrevorder_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "bitrevorder");
    auto *mr = ctx.engine->allocator();
    outs[0] = bitrevorder(args[0], mr);
    // 2nd output: 1-based index vector I such that Y(k) = X(I(k)).
    // Same permutation applied to (1:N), preserving the input shape.
    if (nargout > 1) {
        const keel::usize n = args[0].numel();
        if (n == 0) {
            outs[1] = Value::matrix(0, 0, ValueType::DOUBLE, mr);
            return;
        }
        keel::usize bits = 0;
        for (keel::usize v = n; v > 1; v >>= 1) ++bits;
        const bool isRow = (args[0].dims().rows() == 1);
        Value I = isRow
                    ? Value::matrix(1, n, ValueType::DOUBLE, mr)
                    : Value::matrix(n, 1, ValueType::DOUBLE, mr);
        keel::f64 *id = I.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) {
            // dst[bitReverse(i, bits)] = i+1 (1-based MATLAB index).
            id[bitReverse(i, bits)] = keel::as<keel::f64>(i + 1);
        }
        outs[1] = std::move(I);
    }
}

void dst_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "dst");
    outs[0] = dst(args[0], ctx.engine->allocator());
}

void idst_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "idst");
    outs[0] = idst(args[0], ctx.engine->allocator());
}

void rceps_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "rceps");
    auto *mr = ctx.engine->allocator();
    if (nargout >= 2) {
        auto [y, ym] = rcepsMinPhase(args[0], mr);
        outs[0] = std::move(y);
        outs[1] = std::move(ym);
        return;
    }
    outs[0] = rceps(args[0], mr);
}

void cceps_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "cceps");
    auto *mr = ctx.engine->allocator();
    keel::f64 nd = 0.0;
    outs[0] = cceps(args[0], &nd, mr);
    if (nargout >= 2)   // [xhat, nd] = cceps(x): samples of (circular) delay removed
        outs[1] = Value::scalar(nd, mr);
}

void icceps_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "icceps");
    outs[0] = icceps(args[0], ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::signal
