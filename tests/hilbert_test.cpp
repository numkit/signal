// toolboxes/signal/tests/hilbert_test.cpp
// — sign convention fix.

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class HilbertTest : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {}
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

// MATLAB R2025b reference: hilbert([1:8])
//   real: [1 2 3 4 5 6 7 8]
//   imag: [3.8284 -1 -1 -1.8284 -1.8284 -1 -1 3.8284]

TEST_F(HilbertTest, RealPartEqualsInput)
{
    eval("h = hilbert([1:8]'); r = real(h);");
    for (int k = 1; k <= 8; ++k) {
        EXPECT_DOUBLE_EQ(evalScalar("r(" + std::to_string(k) + ")"),
                         static_cast<double>(k));
    }
}

TEST_F(HilbertTest, ImagPartMatchesMATLABSign)
{
    eval("h = hilbert([1:8]'); im = imag(h);");
    EXPECT_NEAR(evalScalar("im(1)"),  3.8284271247, 1e-9);
    EXPECT_NEAR(evalScalar("im(2)"), -1.0, 1e-12);
    EXPECT_NEAR(evalScalar("im(3)"), -1.0, 1e-12);
    EXPECT_NEAR(evalScalar("im(4)"), -1.8284271247, 1e-9);
    EXPECT_NEAR(evalScalar("im(5)"), -1.8284271247, 1e-9);
    EXPECT_NEAR(evalScalar("im(6)"), -1.0, 1e-12);
    EXPECT_NEAR(evalScalar("im(7)"), -1.0, 1e-12);
    EXPECT_NEAR(evalScalar("im(8)"),  3.8284271247, 1e-9);
}

TEST_F(HilbertTest, EnvelopeMagnitudePositive)
{
    // envelope(x) = |hilbert(x)| — sign-flip-invariant; should still
    // produce positive values.
    eval("e = envelope([1:8]');");
    for (int k = 1; k <= 8; ++k) {
        EXPECT_GT(evalScalar("e(" + std::to_string(k) + ")"), 0.0);
    }
}

// bugs/signal/hilbert-nonpow2 — a non-power-of-two length must be
// transformed at length N, not zero-padded to nextPow2 (which corrupted the
// analytic signal). MATLAB R2025b: hilbert([1:6]') imag =
//   [2.3094, -1.1547, -1.1547, -1.1547, -1.1547, 2.3094].
TEST_F(HilbertTest, NonPowerOfTwoLength)
{
    eval("h = hilbert([1:6]'); r = real(h); im = imag(h);");
    for (int k = 1; k <= 6; ++k)   // real part == input
        EXPECT_NEAR(evalScalar("r(" + std::to_string(k) + ")"),
                    static_cast<double>(k), 1e-13);
    EXPECT_NEAR(evalScalar("im(1)"),  2.3094010768, 1e-9);
    EXPECT_NEAR(evalScalar("im(2)"), -1.1547005384, 1e-9);
    EXPECT_NEAR(evalScalar("im(4)"), -1.1547005384, 1e-9);
    EXPECT_NEAR(evalScalar("im(6)"),  2.3094010768, 1e-9);
}

// Constant-envelope property at a non-pow2 length: a pure tone at an exact
// DFT bin has |hilbert(x)| == 1 everywhere (this was ~0.75 before the fix).
TEST_F(HilbertTest, NonPowerOfTwoConstantEnvelope)
{
    eval("n=(0:99)'; x=cos(2*pi*5*n/100); e=abs(hilbert(x));");
    for (int k = 1; k <= 100; k += 17)
        EXPECT_NEAR(evalScalar("e(" + std::to_string(k) + ")"), 1.0, 1e-9);
}

TEST_F(HilbertTest, PreservesVectorOrientationAndOperatesColumnwiseOnMatrices)
{
    // Row vector stays 1xN, column vector stays Nx1.
    eval("hr = hilbert([1 2 3 4]); hc = hilbert([1; 2; 3; 4]);");
    EXPECT_EQ(evalScalar("size(hr, 1)"), 1.0);
    EXPECT_EQ(evalScalar("size(hr, 2)"), 4.0);
    EXPECT_EQ(evalScalar("size(hc, 1)"), 4.0);
    EXPECT_EQ(evalScalar("size(hc, 2)"), 1.0);

    // 2-D matrix operates along columns (first non-singleton dim).
    eval("A = [1 2; 3 4; 5 6]; H = hilbert(A);");
    EXPECT_EQ(evalScalar("size(H, 1)"), 3.0);
    EXPECT_EQ(evalScalar("size(H, 2)"), 2.0);
    EXPECT_NEAR(evalScalar("real(H(1,1))"), 1.0, 1e-12);
    EXPECT_NEAR(evalScalar("imag(H(1,1))"), 1.1547005383792515, 1e-9);
    EXPECT_NEAR(evalScalar("real(H(2,2))"), 4.0, 1e-12);
    EXPECT_NEAR(evalScalar("imag(H(2,2))"), -2.309401076758503, 1e-9);

    // Explicit n zero-pads or truncates along the first non-singleton dim.
    eval("Hp = hilbert(A, 8); Ht = hilbert(A, 2);");
    EXPECT_EQ(evalScalar("size(Hp, 1)"), 8.0);
    EXPECT_EQ(evalScalar("size(Hp, 2)"), 2.0);
    EXPECT_EQ(evalScalar("size(Ht, 1)"), 2.0);
    EXPECT_EQ(evalScalar("size(Ht, 2)"), 2.0);
}
