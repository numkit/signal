// include/nk/builtin/specfun.hpp
// Standalone specfun header for numkit modules
#pragma once

#include <nk/value/value.hpp>
#include <memory_resource>

namespace nk::builtin {

struct EllipKE {
    Value K;
    Value E;
};

Value erfinv(const Value &x, std::pmr::memory_resource *mr = nullptr);
EllipKE ellipke(const Value &m, std::pmr::memory_resource *mr = nullptr);

} // namespace nk::builtin
