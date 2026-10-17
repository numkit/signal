/// @file medfilt.hpp
/// @ingroup group_signal
// toolboxes/signal/include/nk/signal/smoothing/medfilt.hpp
#pragma once

#include <memory_resource>
#include <numkit/value/value.hpp>

namespace numkit::signal {

/// @addtogroup group_signal
/// @{


/// 1-D median filter.
///
/// Sliding-window median over a window of `k` samples centered on each
/// output sample. Effective for removing salt-and-pepper noise and
/// short outliers while preserving step-like edges (unlike linear
/// smoothing, which blurs edges).
///
/// Boundary handling uses truncation (the `'truncate'`
/// mode): the window is shortened at the signal edges rather than
/// zero-padded.
///
/// @param x   Real 1-D signal.
/// @param k   Window length (samples). Default 3. Typically odd; even k
///            uses the lower-median convention.
/// @param mr  Memory resource (nullptr → process default).
/// @return    Filtered signal, same shape as `x`.
///
/// @code
/// // Remove single-sample outliers in a noisy measurement:
/// Value clean = medfilt1(noisy, 5);
/// @endcode
///
/// @see sgolayfilt
/// @param zeropad  true (MATLAB default): out-of-range samples count as 0
///                 (window always `k` long). false: 'truncate' — clip the
///                 window to the valid range at the signal ends.
Value medfilt1(const Value &                x,
               size_t                       k        = 3,
               bool                         zeropad  = true,
               std::pmr::memory_resource *  mr       = nullptr);

/// Fill missing samples (NaNs) using autoregressive modeling.
///
/// Replaces contiguous runs of NaN values in `x` by extrapolating
/// forward and backward Burg AR models fit to the adjacent non-NaN
/// samples (up to `maxlen` samples on each side) and blending them
/// with a linear ramp across interior gaps.
///
/// @param x       1-D or 2-D float/complex array (vector or column-wise matrix).
/// @param maxlen  Maximum context length on each side (`0` = unlimited).
/// @param order   AR model order (`0` = automatic AIC selection).
/// @param mr      Memory resource (nullptr -> process default).
/// @return        Gap-filled array with same shape and type as `x`.
Value fillgaps(const Value &                x,
               size_t                       maxlen = 0,
               size_t                       order  = 0,
               std::pmr::memory_resource *  mr     = nullptr);


/// @}
} // namespace numkit::signal
