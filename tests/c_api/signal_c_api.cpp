#include <numkit/signal/c_api.h>
#include <gtest/gtest.h>
#include <cmath>

TEST(NumkitSignalCApi, ButterResolveAndInvokeZeroLeak)
{
    const int64_t base = numkit_debug_live_values();
    numkit_error err{};
    numkit_signal_ctx *ctx = numkit_signal_ctx_create(&err);
    ASSERT_NE(ctx, nullptr) << err.message;
    EXPECT_EQ(numkit_signal_has_function(ctx, "butter"), 1);

    const numkit_fn *butterFn = numkit_signal_resolve(ctx, "butter");
    ASSERT_NE(butterFn, nullptr);

    numkit_val *order = numkit_val_scalar(4.0);
    numkit_val *wn = numkit_val_scalar(0.25);
    const numkit_val *args[2] = {order, wn};
    numkit_val *outs[2] = {nullptr, nullptr};
    ASSERT_EQ(numkit_signal_invoke(ctx, butterFn, args, 2, 2, outs, &err), 0) << err.message;
    EXPECT_EQ(numkit_val_numel(outs[0]), 5u);
    EXPECT_EQ(numkit_val_numel(outs[1]), 5u);

    const auto *bPtr = static_cast<const double *>(numkit_val_raw_data(outs[0]));
    const auto *aPtr = static_cast<const double *>(numkit_val_raw_data(outs[1]));
    double sumB = 0.0, sumA = 0.0;
    for (size_t i = 0; i < 5; ++i) { sumB += bPtr[i]; sumA += aPtr[i]; }
    EXPECT_NEAR(sumB / sumA, 1.0, 1e-12);

    numkit_val_release(outs[0]);
    numkit_val_release(outs[1]);
    numkit_val_release(wn);
    numkit_val_release(order);
    numkit_signal_ctx_destroy(ctx);
    EXPECT_EQ(numkit_debug_live_values(), base);
}
