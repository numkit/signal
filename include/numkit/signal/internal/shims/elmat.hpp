// include/nk/builtin/elmat.hpp
// Standalone elmat header for numkit modules
#pragma once

#include <numkit/value/value.hpp>
#include <numkit/value/span.hpp>
#include <cstddef>
#include <memory_resource>

namespace numkit::builtin {

struct ShiftDimAuto {
    Value v;
    std::size_t dropped;
};

Value permute(const Value &x, Span<const int> perm, std::pmr::memory_resource *mr = nullptr);
Value ipermute(const Value &x, Span<const int> perm, std::pmr::memory_resource *mr = nullptr);
Value shiftdim(const Value &x, int n, std::pmr::memory_resource *mr = nullptr);
ShiftDimAuto shiftdimAuto(const Value &x, std::pmr::memory_resource *mr = nullptr);

} // namespace numkit::builtin
