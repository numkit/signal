// include/numkit/builtin/elmat.hpp
// Standalone elmat header for numkit modules
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>
#include <cstddef>

namespace numkit::builtin {

struct ShiftDimAuto {
    Value v;
    int dropped = 0;
};

Value permute(const Value &x, keel::ArrayRef<int> perm, keel::Allocator *mr = nullptr);
Value ipermute(const Value &x, keel::ArrayRef<int> perm, keel::Allocator *mr = nullptr);
Value shiftdim(const Value &x, int n, keel::Allocator *mr = nullptr);
ShiftDimAuto shiftdimAuto(const Value &x, keel::Allocator *mr = nullptr);

} // namespace numkit::builtin
