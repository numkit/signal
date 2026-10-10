// include/nk/builtin/polyfun.hpp
// Standalone polyfun header for numkit modules
#pragma once

#include <nk/value/value.hpp>
#include <tuple>
#include <string>
#include <memory_resource>

namespace nk::builtin {

Value roots(const Value &p, std::pmr::memory_resource *mr = nullptr);
Value poly(const Value &r, std::pmr::memory_resource *mr = nullptr);
std::tuple<Value, Value, Value> tf2zp(const Value &b, const Value &a, std::pmr::memory_resource *mr = nullptr);
std::tuple<Value, Value> zp2tf(const Value &z, const Value &p, double k, std::pmr::memory_resource *mr = nullptr);
Value interp1(const Value &x, const Value &y, const Value &xq, const std::string &method, std::pmr::memory_resource *mr = nullptr);

} // namespace nk::builtin
