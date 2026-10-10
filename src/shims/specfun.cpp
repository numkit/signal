// src/shims/specfun.cpp
#include <numkit/signal/internal/shims/specfun.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <limits>
#include <algorithm>

namespace numkit::builtin {

namespace {

double erfinvScalar(double y)
{
    if (std::isnan(y))            return y;
    if (y >  1.0 || y < -1.0)     return std::nan("");
    if (y ==  1.0)                return std::numeric_limits<double>::infinity();
    if (y == -1.0)                return -std::numeric_limits<double>::infinity();
    if (y ==  0.0)                return 0.0;

    constexpr double kA  = 0.147;
    constexpr double k2P = 2.0 / 3.14159265358979323846;
    const double s   = (y < 0) ? -1.0 : 1.0;
    const double ay  = std::abs(y);
    const double L   = std::log(1.0 - ay * ay);
    const double t   = k2P / kA + 0.5 * L;
    double z = s * std::sqrt(std::sqrt(t * t - L / kA) - t);

    constexpr double kInvSqrtPi = 0.56418958354775628694;
    for (int i = 0; i < 3; ++i) {
        const double err = std::erf(z) - y;
        const double dz  = err / (2.0 * kInvSqrtPi * std::exp(-z * z));
        z -= dz;
    }
    return z;
}

std::pair<double, double> ellipKEScalar(double m)
{
    if (std::isnan(m)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    if (m == 1.0) return {std::numeric_limits<double>::infinity(), 1.0};
    if (m < 0.0 || m > 1.0)
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};

    double a = 1.0, g = std::sqrt(1.0 - m);
    double sum = 0.5 * m;
    double scale = 0.25;
    constexpr int kMaxIter = 64;
    for (int i = 0; i < kMaxIter; ++i) {
        const double diff = a - g;
        if (std::abs(diff) < 1e-17 * a) break;
        sum += scale * diff * diff;
        const double aNew = 0.5 * (a + g);
        const double gNew = std::sqrt(a * g);
        a = aNew;
        g = gNew;
        scale *= 2.0;
    }
    constexpr double kPiOver2 = 1.57079632679489661923;
    const double K = kPiOver2 / a;
    const double E = K * (1.0 - sum);
    return {K, E};
}

} // namespace

Value erfinv(const Value &x, std::pmr::memory_resource *mr)
{
    return unaryDouble(x, [](double v) { return erfinvScalar(v); }, mr);
}

EllipKE ellipke(const Value &m, std::pmr::memory_resource *mr)
{
    auto K = createLike(m, ValueType::DOUBLE, mr);
    auto E = createLike(m, ValueType::DOUBLE, mr);
    if (m.isScalar()) {
        const auto [k, e] = ellipKEScalar(m.toScalar());
        return { Value::scalar(k, mr), Value::scalar(e, mr) };
    }
    const size_t n = m.numel();
    for (size_t i = 0; i < n; ++i) {
        const auto [k, e] = ellipKEScalar(m.doubleData()[i]);
        K.doubleDataMut()[i] = k;
        E.doubleDataMut()[i] = e;
    }
    return { std::move(K), std::move(E) };
}

} // namespace numkit::builtin
