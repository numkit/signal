// toolboxes/signal/include/numkit/signal/measurements/sigroi.hpp
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>

#include <cstdint>

namespace numkit::signal {

/// @addtogroup group_signal
/// @{


/// @file
/// @ingroup group_signal
/// @brief Signal-region-of-interest utilities.
///
/// ROIs are represented as `N × 2` matrices where each row is a 1-based
/// inclusive `[start, end]` pair. Binary masks are 1-D LOGICAL columns
/// where consecutive `true` runs correspond to ROIs.

/// @brief Binary mask → ROI pairs (`roi = binmask2sigroi(m)`).
///
/// @param m   Logical column.
/// @param mr  Memory resource (nullptr → process default).
/// @return    `N × 2` matrix of 1-based inclusive `[start, end]` runs;
///            empty mask → `0 × 0` matrix.
/// @see sigroi2binmask
Value binmask2sigroi(const Value &m,
                     keel::Allocator *mr = nullptr);

/// @brief ROI pairs → binary mask (`m = sigroi2binmask(roi, len)`).
///
/// @param roi  `N × 2` ROI matrix.
/// @param len  Output length: `-1` (default) → `max(roi(:, 2))`;
///             otherwise pad with zeros (or clamp ROIs) to exact length.
/// @param mr   Memory resource (nullptr → process default).
/// @return     LOGICAL column.
/// @see binmask2sigroi
Value sigroi2binmask(const Value &roi, keel::i64 len = -1,
                     keel::Allocator *mr = nullptr);

/// @brief Extend each ROI (`roi = extendsigroi(roi, Lpre, Lpost)`).
///
/// Extends each ROI by `Lpre` samples on the left and `Lpost` on the
/// right; clamps start to `>= 1`.
///
/// @param roi    `N × 2` ROI matrix.
/// @param Lpre   Left extension.
/// @param Lpost  Right extension.
/// @param mr     Memory resource (nullptr → process default).
/// @return       Extended ROI matrix.
/// @see shortensigroi
Value extendsigroi(const Value &roi, keel::i64 Lpre, keel::i64 Lpost,
                   keel::Allocator *mr = nullptr);

/// @brief Shrink each ROI (`roi = shortensigroi(roi, Lpre, Lpost)`).
///
/// Shrinks each ROI by `Lpre` on the left and `Lpost` on the right.
/// ROIs that collapse (`end < start`) are dropped.
///
/// @param roi    `N × 2` ROI matrix.
/// @param Lpre   Left shrink.
/// @param Lpost  Right shrink.
/// @param mr     Memory resource (nullptr → process default).
/// @return       Shrunken ROI matrix.
/// @see extendsigroi
Value shortensigroi(const Value &roi, keel::i64 Lpre, keel::i64 Lpost,
                    keel::Allocator *mr = nullptr);

/// @brief Merge near-adjacent ROIs (`roi = mergesigroi(roi, sep)`).
///
/// `sep == 0` → only strictly overlapping ROIs merge.
/// `sep >= 1` → ROIs within `sep` samples of each other merge.
///
/// @param roi  `N × 2` ROI matrix.
/// @param sep  Merge gap threshold.
/// @param mr   Memory resource (nullptr → process default).
/// @return     Merged ROI matrix.
/// @see extendsigroi
Value mergesigroi(const Value &roi, keel::i64 sep,
                  keel::Allocator *mr = nullptr);

/// @brief Drop short ROIs (`roi = removesigroi(roi, maxLen)`).
///
/// Drops ROIs whose length (`end - start + 1`) is `<= maxLen`.
///
/// @param roi     `N × 2` ROI matrix.
/// @param maxLen  Length threshold (ROIs `<= maxLen` are dropped).
/// @param mr      Memory resource (nullptr → process default).
/// @return        Filtered ROI matrix.
Value removesigroi(const Value &roi, keel::i64 maxLen,
                   keel::Allocator *mr = nullptr);

/// @brief Extract signal slices indexed by ROIs
/// (`y = extractsigroi(x, roi, concat)`).
///
/// @param x       Source signal (column vector).
/// @param roi     `N × 2` ROI matrix.
/// @param concat  `false` (default) → return a cell of column-vector
///                slices; `true` → return a single concatenated column.
/// @param mr      Memory resource (nullptr → process default).
/// @return        Cell or concatenated column.
Value extractsigroi(const Value &x, const Value &roi, bool concat = false,
                    keel::Allocator *mr = nullptr);

/// @brief Range-based binary mask, single-threshold form
/// (`m = sigrangebinmask(x, threshold)` ≡ `x > threshold`).
///
/// @param x          Source signal column.
/// @param threshold  Scalar threshold.
/// @param mr         Memory resource (nullptr → process default).
/// @return           LOGICAL column, same length as `x`.
/// @see binmask2sigroi
Value sigrangebinmask(const Value &x, keel::f64 threshold,
                      keel::Allocator *mr = nullptr);

/// @brief Range-based binary mask, interval form
/// (`m = sigrangebinmask(x, [lo, hi])` ≡ `lo <= x <= hi`).
///
/// @param x   Source signal column.
/// @param lo  Lower bound (inclusive).
/// @param hi  Upper bound (inclusive).
/// @param mr  Memory resource (nullptr → process default).
/// @return    LOGICAL column, same length as `x`.
/// @see binmask2sigroi
Value sigrangebinmask(const Value &x, keel::f64 lo, keel::f64 hi,
                      keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::signal
