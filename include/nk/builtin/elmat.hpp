// include/nk/builtin/elmat.hpp
// Standalone elmat header for numkit modules
#pragma once

#include <nk/value/value.hpp>
#include <nk/value/span.hpp>
#include <cstddef>
#include <memory_resource>

namespace nk::builtin {

struct ShiftDimAuto {
    Value v;
    std::size_t dropped;
};

Value permute(const Value &x, Span<const int> perm, std::pmr::memory_resource *mr = nullptr);
Value ipermute(const Value &x, Span<const int> perm, std::pmr::memory_resource *mr = nullptr);
Value shiftdim(const Value &x, int n, std::pmr::memory_resource *mr = nullptr);
ShiftDimAuto shiftdimAuto(const Value &x, std::pmr::memory_resource *mr = nullptr);

} // namespace nk::builtin
