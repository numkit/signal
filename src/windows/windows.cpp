// toolboxes/signal/src/windows/windows.cpp
//
// Window generators & DPSS tapers, delegated to keel::signal.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/signal/windows/windows.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::signal {

namespace {

void gausswin_argcheck(keel::f64 alpha)
{
    if (alpha <= 0)
        throw Error("gausswin: alpha must be positive",
                     0, 0, "gausswin", "", "numkit:gausswin:badAlpha");
}

} // namespace

Value hamming(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::hamming({r.doubleDataMut(), N});
    return r;
}

Value hann(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::hann({r.doubleDataMut(), N});
    return r;
}

Value blackman(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::blackman({r.doubleDataMut(), N});
    return r;
}

Value kaiser(keel::usize N, keel::f64 beta, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::kaiser({r.doubleDataMut(), N}, beta);
    return r;
}

Value rectwin(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::rectwin({r.doubleDataMut(), N});
    return r;
}

Value bartlett(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::bartlett({r.doubleDataMut(), N});
    return r;
}

Value triang(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::triang({r.doubleDataMut(), N});
    return r;
}

Value tukeywin(keel::usize N, keel::f64 r, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::tukeywin({out.doubleDataMut(), N}, r);
    return out;
}

Value flattopwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::flattopwin({out.doubleDataMut(), N});
    return out;
}

Value gausswin(keel::usize N, keel::f64 alpha, keel::Allocator *mr)
{
    gausswin_argcheck(alpha);
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::gausswin({out.doubleDataMut(), N}, alpha);
    return out;
}

Value chebwin(keel::usize N, keel::f64 at, keel::Allocator *mr)
{
    if (at <= 0)
        throw Error("chebwin: at must be positive (sidelobe attenuation in dB)",
                     0, 0, "chebwin", "", "numkit:chebwin:badAt");
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::chebwin({out.doubleDataMut(), N}, at, mr);
    return out;
}

Value parzenwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::parzenwin({out.doubleDataMut(), N});
    return out;
}

Value nuttallwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::nuttallwin({out.doubleDataMut(), N});
    return out;
}

Value taylorwin(keel::usize N, keel::i32 nbar, keel::f64 sll, keel::Allocator *mr)
{
    if (sll >= 0)
        throw Error("taylorwin: sll must be negative (peak sidelobe in dB)",
                     0, 0, "taylorwin", "", "numkit:taylorwin:badSll");
    if (nbar < 2)
        throw Error("taylorwin: nbar must be >= 2",
                     0, 0, "taylorwin", "", "numkit:taylorwin:badNbar");
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    (void)keel::signal::taylorwin({out.doubleDataMut(), N}, keel::as<keel::i32>(nbar), sll, mr);
    return out;
}

Value blackmanharris(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::blackmanharris({out.doubleDataMut(), N});
    return out;
}

Value bohmanwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::bohmanwin({out.doubleDataMut(), N});
    return out;
}

Value barthannwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::barthannwin({out.doubleDataMut(), N});
    return out;
}

keel::Pair<Value, Value> dpss(keel::usize N, keel::f64 nw, keel::usize k,
                             keel::Allocator *mr)
{
    if (N == 0 || k == 0) {
        return {Value::matrix(N, k, ValueType::DOUBLE, mr),
                Value::matrix(k, 1, ValueType::DOUBLE, mr)};
    }
    if (k > N) k = N;
    Value E = Value::matrix(N, k, ValueType::DOUBLE, mr);
    Value V = Value::matrix(k, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::dpss(N, nw, {E.doubleDataMut(), N * k}, {V.doubleDataMut(), k}, mr);
    return {keel::move(E), keel::move(V)};
}

} // namespace numkit::signal
