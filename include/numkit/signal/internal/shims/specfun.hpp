// include/numkit/builtin/specfun.hpp
// Standalone specfun header for numkit modules
#pragma once

#include <keel/memory.h>
#include <numkit/value/value.hpp>

namespace numkit::builtin {

struct EllipKE {
    Value K;
    Value E;
};

Value erfinv(const Value &x, keel::Allocator *mr = nullptr);
EllipKE ellipke(const Value &m, keel::Allocator *mr = nullptr);

} // namespace numkit::builtin
