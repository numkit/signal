// toolboxes/signal/src/windows/windows.cpp
//
// Window generators & DPSS tapers, delegated to keel::signal.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/signal.h>
#include <numkit/signal/windows/windows.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <utility>

namespace numkit::signal {

namespace {

void gausswin_argcheck(double alpha)
{
    if (alpha <= 0)
        throw Error("gausswin: alpha must be positive",
                     0, 0, "gausswin", "", "numkit:gausswin:badAlpha");
}

} // namespace

Value hamming(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::hamming(N, r.doubleDataMut());
    return r;
}

Value hann(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::hann(N, r.doubleDataMut());
    return r;
}

Value blackman(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::blackman(N, r.doubleDataMut());
    return r;
}

Value kaiser(keel::usize N, double beta, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::kaiser(N, beta, r.doubleDataMut());
    return r;
}

Value rectwin(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::rectwin(N, r.doubleDataMut());
    return r;
}

Value bartlett(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::bartlett(N, r.doubleDataMut());
    return r;
}

Value triang(keel::usize N, keel::Allocator *mr)
{
    auto r = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::triang(N, r.doubleDataMut());
    return r;
}

Value tukeywin(keel::usize N, double r, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::tukeywin(N, r, out.doubleDataMut());
    return out;
}

Value flattopwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::flattopwin(N, out.doubleDataMut());
    return out;
}

Value gausswin(keel::usize N, double alpha, keel::Allocator *mr)
{
    gausswin_argcheck(alpha);
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::gausswin(N, alpha, out.doubleDataMut());
    return out;
}

Value chebwin(keel::usize N, double at, keel::Allocator *mr)
{
    if (at <= 0)
        throw Error("chebwin: at must be positive (sidelobe attenuation in dB)",
                     0, 0, "chebwin", "", "numkit:chebwin:badAt");
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::chebwin(N, at, out.doubleDataMut(), mr);
    return out;
}

Value parzenwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::parzenwin(N, out.doubleDataMut());
    return out;
}

Value nuttallwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::nuttallwin(N, out.doubleDataMut());
    return out;
}

Value taylorwin(keel::usize N, int nbar, double sll, keel::Allocator *mr)
{
    if (sll >= 0)
        throw Error("taylorwin: sll must be negative (peak sidelobe in dB)",
                     0, 0, "taylorwin", "", "numkit:taylorwin:badSll");
    if (nbar < 2)
        throw Error("taylorwin: nbar must be >= 2",
                     0, 0, "taylorwin", "", "numkit:taylorwin:badNbar");
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    (void)keel::signal::taylorwin(N, static_cast<keel::i32>(nbar), sll, out.doubleDataMut(), mr);
    return out;
}

Value blackmanharris(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::blackmanharris(N, out.doubleDataMut());
    return out;
}

Value bohmanwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::bohmanwin(N, out.doubleDataMut());
    return out;
}

Value barthannwin(keel::usize N, keel::Allocator *mr)
{
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::signal::barthannwin(N, out.doubleDataMut());
    return out;
}

std::pair<Value, Value> dpss(keel::usize N, double nw, keel::usize k,
                             keel::Allocator *mr)
{
    if (N == 0 || k == 0) {
        return {Value::matrix(N, k, ValueType::DOUBLE, mr),
                Value::matrix(k, 1, ValueType::DOUBLE, mr)};
    }
    if (k > N) k = N;
    Value E = Value::matrix(N, k, ValueType::DOUBLE, mr);
    Value V = Value::matrix(k, 1, ValueType::DOUBLE, mr);
    (void)keel::signal::dpss(N, nw, k, E.doubleDataMut(), V.doubleDataMut(), mr);
    return {std::move(E), std::move(V)};
}

} // namespace numkit::signal
