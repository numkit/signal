// toolboxes/signal/tests/findpeaks_width_test.cpp
//
// 'WidthReference' option ('halfprom' default / 'halfheight') — every
// expected value below was validated against the reference engine
// (R2025b, probed 2026-09-17; derivation + probe tables in
// bugs/closed/signal/findpeaks-widthreference.md). Dual-engine: the width
// computation runs inside findpeaksImpl for both backends.

#include "dual_engine_fixture.hpp"
#include <numkit/value/error.hpp>

#include <cmath>
#include <string>

using namespace m_test;

class FindpeaksWidthTest : public DualEngineTest
{
public:
    // w(i) of the third output of findpeaks(y, ...)
    double wOf(const std::string &y, const std::string &opts, size_t i)
    {
        eval("[~,~,w] = findpeaks(" + y + opts + ");");
        auto *w = getVarPtr("w");
        return w->doubleData()[i];
    }
    size_t wNumel(const std::string &y, const std::string &opts)
    {
        eval("[~,~,w] = findpeaks(" + y + opts + ");");
        return getVarPtr("w")->numel();
    }
};

// The anchor vector from the original bug file — three peaks, every width
// distinct between the two references.
TEST_P(FindpeaksWidthTest, AnchorVectorBothReferences)
{
    EXPECT_NEAR(wOf("[1 3 2 5 1 6 1]", "", 0), 0.75, 1e-12);            // halfprom default
    EXPECT_NEAR(wOf("[1 3 2 5 1 6 1]", "", 1), 1.1666666666666667, 1e-12);
    EXPECT_NEAR(wOf("[1 3 2 5 1 6 1]", "", 2), 1.0, 1e-12);
    EXPECT_NEAR(wOf("[1 3 2 5 1 6 1]", ", 'WidthReference', 'halfheight'", 0), 1.75, 1e-12);
    EXPECT_NEAR(wOf("[1 3 2 5 1 6 1]", ", 'WidthReference', 'halfheight'", 1), 1.4583333333333333, 1e-12);
    EXPECT_NEAR(wOf("[1 3 2 5 1 6 1]", ", 'WidthReference', 'halfheight'", 2), 1.2, 1e-12);
    // Peak 3@2: ref 1.5 lies BELOW the right shoulder's rise — the walk
    // stops at the rise and the bound is the first plateau (sample 3),
    // NOT the later 1@5 crossing. Distinguishes halfheight from
    // "halfprom machinery with a different ref".
}

// Explicit 'halfprom' is the default.
TEST_P(FindpeaksWidthTest, ExplicitHalfpromEqualsDefault)
{
    eval("[~,~,w1] = findpeaks([5 5 5 6 9 6 5 5 5]);"
         "[~,~,w2] = findpeaks([5 5 5 6 9 6 5 5 5], 'WidthReference', 'halfprom');");
    EXPECT_DOUBLE_EQ(getVarPtr("w1")->doubleData()[0],
                      getVarPtr("w2")->doubleData()[0]);
}

// Pedestal family — the no-crossing case that blocked the fix for two
// sessions: the bounds are the flat's far end (left) and first flat
// sample (right), NOT the naive full inter-border span.
TEST_P(FindpeaksWidthTest, HalfHeightPedestalFamily)
{
    const std::string hh = ", 'WidthReference', 'halfheight'";
    EXPECT_NEAR(wOf("[5 5 5 6 9 6 5 5 5]", hh, 0), 6.0, 1e-12); // classic pedestal
    EXPECT_NEAR(wOf("[5 5 5 6 9 6 5]", hh, 0), 6.0, 1e-12);     // right flat of one sample
    EXPECT_NEAR(wOf("[5 6 9 6 5 5 5]", hh, 0), 4.0, 1e-12);     // left flat of one sample
    EXPECT_NEAR(wOf("[5 5 6 9 6 5 5 5]", hh, 0), 5.0, 1e-12);
    EXPECT_NEAR(wOf("[5 5 5 5 6 9 6 5 5 5]", hh, 0), 7.0, 1e-12);
    EXPECT_NEAR(wOf("[5 5 5 9 5 5 5]", hh, 0), 4.0, 1e-12);     // no shoulders at all
    EXPECT_NEAR(wOf("[5 5 5 6 9 7 5 5 5]", hh, 0), 6.0, 1e-12); // asymmetric shoulders
    EXPECT_NEAR(wOf("[6 5 5 5 6 9 6 5 5 5 6]", hh, 0), 6.0, 1e-12); // sub-peak walls ignored
}

// The reference level exactly EQUAL to the flat: the "crossing" lands on
// the first touching sample (interpolation f = 0), even when a strictly
// lower sample follows — the first event on the walk wins.
TEST_P(FindpeaksWidthTest, HalfHeightTouchingReference)
{
    const std::string hh = ", 'WidthReference', 'halfheight'";
    EXPECT_NEAR(wOf("[5 5 5 6 10 6 5 5 5]", hh, 0), 4.0, 1e-12); // ref == flat
    EXPECT_NEAR(wOf("[5 5 5 6 10 6 5 5 4]", hh, 0), 4.0, 1e-12); // touch before deep dip
}

// One side crosses, the other falls back to the plateau bound.
TEST_P(FindpeaksWidthTest, HalfHeightMixedCrossingAndFallback)
{
    const std::string hh = ", 'WidthReference', 'halfheight'";
    EXPECT_NEAR(wOf("[5 5 5 6 9 6 5 5 4]", hh, 0), 7.5, 1e-12);  // left fallback, right crossing 8.5
    EXPECT_NEAR(wOf("[5 5 4 5 5 5 6 9 6 5 5 5]", hh, 0), 6.5, 1e-12); // left crossing 3.5, right fallback
    EXPECT_NEAR(wOf("[5 5 5 5 5 5 6 9 6 5 4]", hh, 0), 9.5, 1e-12);
    EXPECT_NEAR(wOf("[4 5 5 5 5 5 6 9 6 5 5 5 5 5 5]", hh, 0), 8.5, 1e-12);
    EXPECT_NEAR(wOf("[5 5 4 5 5 5 6 9 6 5 5 5 5 5]", hh, 0), 6.5, 1e-12);
}

// A rise before the reference level is reached ends the walk: a deeper
// sample beyond the rise is never a bound (peak 9 in t1/t2), and a
// higher neighbouring peak bounds its own width by crossing.
TEST_P(FindpeaksWidthTest, HalfHeightWalkStopsAtRise)
{
    const std::string hh = ", 'WidthReference', 'halfheight'";
    ASSERT_EQ(wNumel("[5 6 5 5 5 6 9 6 5]", hh), 2u);
    EXPECT_NEAR(wOf("[5 6 5 5 5 6 9 6 5]", hh, 0), 2.0, 1e-12);  // peak 6
    EXPECT_NEAR(wOf("[5 6 5 5 5 6 9 6 5]", hh, 1), 6.0, 1e-12);  // peak 9: plateau [3..5]
    ASSERT_EQ(wNumel("[5 5 5 6 9 6 5 7 4.7]", hh), 2u);
    EXPECT_NEAR(wOf("[5 5 5 6 9 6 5 7 4.7]", hh, 0), 6.0, 1e-12); // deeper 4.7 ignored
    EXPECT_NEAR(wOf("[5 5 5 6 9 6 5 7 4.7]", hh, 1), 2.0, 1e-12);
    ASSERT_EQ(wNumel("[5 5 5 6 9 6 5 5 5 12 5]", hh), 2u);
    EXPECT_NEAR(wOf("[5 5 5 6 9 6 5 5 5 12 5]", hh, 0), 6.0, 1e-12);
    EXPECT_NEAR(wOf("[5 5 5 6 9 6 5 5 5 12 5]", hh, 1), 12.0 / 7.0, 1e-12); // higher peak crosses
}

// Pure-crossing sanity (descendant base) + far-edge geometries.
TEST_P(FindpeaksWidthTest, HalfHeightCrossingAndEdges)
{
    const std::string hh = ", 'WidthReference', 'halfheight'";
    EXPECT_NEAR(wOf("[3 4 5 6 9 6 5 4 3]", hh, 0), 5.0, 1e-12);  // ordinary interpolation
    EXPECT_NEAR(wOf("[4 5 5 5 6 9 6 5 5 5 4]", hh, 0), 9.0, 1e-12);
    EXPECT_NEAR(wOf("[5 5 9 5 5 5]", hh, 0), 3.0, 1e-12);        // short left flat
    EXPECT_NEAR(wOf("[5 5 5 6 9 5]", hh, 0), 5.0, 1e-12);        // single right sample
}

// halfprom searches the crossing across the FULL prominence interval —
// a rise before the interval minimum does not stop it (the mirror
// behaviour of halfheight's walk; both values reference-validated).
TEST_P(FindpeaksWidthTest, HalfPromIntervalSearchIgnoresRise)
{
    ASSERT_EQ(wNumel("[7.5 8 9 8.5 8.6 2]", ""), 2u);
    EXPECT_NEAR(wOf("[7.5 8 9 8.5 8.6 2]", "", 0), 2.803030303030303, 1e-12);
    EXPECT_NEAR(wOf("[7.5 8 9 8.5 8.6 2]", "", 1), 0.5075757575757576, 1e-12);
    const std::string hh = ", 'WidthReference', 'halfheight'";
    EXPECT_NEAR(wOf("[7.5 8 9 8.5 8.6 2]", hh, 0), 3.0, 1e-12);
    EXPECT_NEAR(wOf("[7.5 8 9 8.5 8.6 2]", hh, 1), 1.6515151515151516, 1e-12);
}

// Widths come out in location units for both the X-vector and Fs forms
// (bounds are sample positions mapped through the location array).
TEST_P(FindpeaksWidthTest, HalfHeightLocationUnits)
{
    const std::string hh = ", 'WidthReference', 'halfheight'";
    eval("xx = [0 0.5 1 1.5 2 2.5 3]; yy = [5 5 5 6 9 6 5];");
    EXPECT_NEAR(wOf("yy, xx", hh, 0), 3.0, 1e-12);
    EXPECT_NEAR(wOf("yy, 2", hh, 0), 3.0, 1e-12); // Fs = 2
}

// Invalid reference value — nk-voiced diagnostic blaming the option.
TEST_P(FindpeaksWidthTest, BadWidthReferenceValue)
{
    bool threw = false;
    try {
        eval("[~,~,w] = findpeaks([1 3 2], 'WidthReference', 'bogus');");
    } catch (const numkit::Error &e) {
        threw = true;
        const std::string msg = e.what();
        EXPECT_EQ(e.identifier(), "numkit:findpeaks:WidthReferenceNotChoice");
        EXPECT_TRUE(msg.find("halfprom") != std::string::npos);
        EXPECT_TRUE(msg.find("bogus") != std::string::npos);
    }
    EXPECT_TRUE(threw);
}

INSTANTIATE_DUAL(FindpeaksWidthTest);
