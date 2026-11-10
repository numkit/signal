/* include/numkit/signal/c_api.h
 *
 * Self-contained C-ABI for the `numkit/signal` toolbox.
 *
 * Supports both:
 *   1. Pre-resolved O(1) dispatch (`numkit_signal_resolve` + `numkit_signal_invoke`)
 *      with zero string hashing on the hot path.
 *   2. Dynamic name/method dispatch (`numkit_signal_call`) for MATLAB overloads,
 *      Name-Value pairs, and toolbox object methods.
 */
#ifndef NUMKIT_SIGNAL_C_API_H
#define NUMKIT_SIGNAL_C_API_H

#include <numkit/value/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(NUMKIT_PLUGIN_BUILD)
#    define NUMKIT_SIGNAL_API __declspec(dllexport)
#  elif defined(NUMKIT_SIGNAL_STATIC)
#    define NUMKIT_SIGNAL_API
#  else
#    define NUMKIT_SIGNAL_API __declspec(dllimport)
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define NUMKIT_SIGNAL_API __attribute__((visibility("default")))
#  else
#    define NUMKIT_SIGNAL_API
#  endif
#endif

/* Opaque isolated `signal` toolbox context */
typedef struct numkit_signal_ctx_s numkit_signal_ctx;

/* Create an isolated `signal` toolbox context (engine-free, zero interpreter overhead). */
NUMKIT_SIGNAL_API numkit_signal_ctx *numkit_signal_ctx_create(numkit_error *err);

/* Destroy `signal` toolbox context and release its resources. */
NUMKIT_SIGNAL_API void numkit_signal_ctx_destroy(numkit_signal_ctx *ctx);

/* Seed the `signal` context's random number stream (MATLAB `rng(seed)` semantics). */
NUMKIT_SIGNAL_API void numkit_signal_seed_rng(numkit_signal_ctx *ctx, uint64_t seed);

/* Return 1 if function `name` is registered in `signal`, 0 otherwise. */
NUMKIT_SIGNAL_API int numkit_signal_has_function(const numkit_signal_ctx *ctx, const char *name);

/* Register a host-language callback (`numkit_host_fn`) into this context. */
NUMKIT_SIGNAL_API int numkit_signal_register_fn(numkit_signal_ctx *ctx,
                                            const char *name,
                                            numkit_host_fn fn,
                                            void *user_data,
                                            numkit_error *err);

/* Resolve function `name` once into an O(1) function pointer (`NULL` if not found). */
NUMKIT_SIGNAL_API const numkit_fn *numkit_signal_resolve(const numkit_signal_ctx *ctx,
                                                     const char *name);

/* Invoke a pre-resolved function pointer directly with zero lookup overhead. */
NUMKIT_SIGNAL_API int numkit_signal_invoke(numkit_signal_ctx *ctx,
                                       const numkit_fn *fn,
                                       const numkit_val *const *args,
                                       size_t nargs,
                                       size_t nargout,
                                       numkit_val **outs,
                                       numkit_error *err);

/* Dynamically invoke `signal` function or object method `name`. */
NUMKIT_SIGNAL_API int numkit_signal_call(numkit_signal_ctx *ctx,
                                     const char *name,
                                     const numkit_val *const *args,
                                     size_t nargs,
                                     size_t nargout,
                                     numkit_val **outs,
                                     numkit_error *err);

/* Format a value or `signal` class object (e.g. `sym`, `tf`, `ss`) to UTF-8 text. */
NUMKIT_SIGNAL_API size_t numkit_signal_format(const numkit_signal_ctx *ctx,
                                          const numkit_val *v,
                                          char *buf,
                                          size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_SIGNAL_C_API_H */
