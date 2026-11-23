// include/numkit/builtin/elmat.hpp
// Standalone elmat header for numkit modules
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit::builtin {

struct ShiftDimAuto {
    Value v;
    keel::i32 dropped = 0;
};

Value permute(const Value &x, keel::ArrayRef<keel::i32> perm, keel::Allocator *mr = nullptr);
Value ipermute(const Value &x, keel::ArrayRef<keel::i32> perm, keel::Allocator *mr = nullptr);
Value shiftdim(const Value &x, keel::i32 n, keel::Allocator *mr = nullptr);
ShiftDimAuto shiftdimAuto(const Value &x, keel::Allocator *mr = nullptr);

} // namespace numkit::builtin
