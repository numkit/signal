// include/numkit/builtin/polyfun.hpp
// Standalone polyfun header for numkit modules
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>
#include <keel/tuple.h>

namespace numkit::builtin {

Value roots(const Value &p, keel::Allocator *mr = nullptr);
Value poly(const Value &r, keel::Allocator *mr = nullptr);
keel::Tuple<Value, Value, Value> tf2zp(const Value &b, const Value &a, keel::Allocator *mr = nullptr);
keel::Tuple<Value, Value> zp2tf(const Value &z, const Value &p, keel::f64 k, keel::Allocator *mr = nullptr);
Value interp1(const Value &x, const Value &y, const Value &xq, keel::StringRef method = "linear", keel::Allocator *mr = nullptr);

} // namespace numkit::builtin
