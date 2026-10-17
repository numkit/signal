// signal/tests/argcheck_signal_test.cpp
//
// Guards for the argcheck migration of the signal pilot
// (dev-docs/todo/argcheck_separation.md):
//   * filter — FilterPlan resolver + the [y,zf] loop extracted from
//     filter_reg into compute (filter_argcheck/runFilter);
//   * fir1 / firls — expectedReal/expectedPositive extraction,
//     fir1_argcheck first line;
//   * butter + IIR family — order contracts.
// The REJECTION behavior (which inputs fail, which argument is blamed)
// is validated against the reference engine; expected strings are
// nk-voiced with the received value/shape (user rule 2026-09-17).

#include <gtest/gtest.h>
#include <numkit/core/standard_engine.hpp>
#include <numkit/signal/digital_filtering/filter.hpp>
#include <numkit/signal/filter_design/filter_design.hpp>
#include <numkit/signal/filter_design/iir_designs.hpp>
#include <numkit/signal/multirate/multirate.hpp>
#include <numkit/signal/smoothing/medfilt.hpp>
#include <numkit/value/value.hpp>

#include <cmath>

using namespace numkit;

namespace {

class ArgcheckSignalTest : public ::testing::Test
{
public:
    StandardEngine engine;
    Value eval(const std::string &code) { return engine.eval(code); }
    double evalScalar(const std::string &code) { return eval(code).toScalar(); }

    void expectError(const char *code, const char *id)
    {
        expectError(code, id, nullptr);
    }
    void expectError(const char *code, const char *id, const char *msg)
    {
        try {
            eval(code);
        } catch (const Error &e) {
            EXPECT_EQ(e.identifier(), id) << code;
            if (msg)
                EXPECT_STREQ(e.what(), msg) << code;
            return;
        }
        ADD_FAILURE() << code << ": expected numkit::Error " << id;
    }
};

} // namespace

// ── filter: argument-contract diagnostics ────────────────────────────────
TEST_F(ArgcheckSignalTest, FilterArgcheckDiagnostics)
{
    expectError("filter();", "numkit:filter:nargin",
                "filter: not enough input arguments — at least 3 required "
                "(got 0).");
    expectError("filter('x', 1, 1:3);", "numkit:filter:inputType",
                "filter: b must be numeric or logical (type is char).");
    expectError("filter(eye(2), 1, 1:3);", "numkit:filter:notVector",
                "filter: b and a must be non-empty vectors (b is 2x2).");
    expectError("filter(1, [], 1:3);", "numkit:filter:notVector",
                "filter: b and a must be non-empty vectors (a is 0x0).");
    expectError("filter(1, 0, 1:3);", "numkit:filter:zeroLead",
                "filter: first denominator coefficient a(1) must be "
                "non-zero (got 0).");
    expectError("filter(1i, 0i, 1:3);", "numkit:filter:zeroLead",
                "filter: first denominator coefficient a(1) must be "
                "non-zero (got 0).");
    expectError("[y, zf] = filter(1, 1, 1:3, [0 0]);",
                "numkit:filter:invalidInitialConditions",
                "filter: zi must have max(length(a),length(b))-1 = 0 "
                "elements per signal (got 2).");
}

// Logical inputs are legal filter inputs (the old real path required a
// double array).
TEST_F(ArgcheckSignalTest, FilterLogicalInputWorks)
{
    eval("y = filter(1, 1, logical([1 0 1]));");
    EXPECT_EQ(evalScalar("y(1)"), 1.0);
    EXPECT_EQ(evalScalar("y(2)"), 0.0);
    EXPECT_EQ(evalScalar("y(3)"), 1.0);
}

// ── filter: direct C++ (both-paths guarantee) ────────────────────────────
TEST_F(ArgcheckSignalTest, FilterDirectCApiContract)
{
    std::pmr::memory_resource *mr = std::pmr::get_default_resource();

    Value b = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    b.doubleDataMut()[0] = 1.0;
    b.doubleDataMut()[1] = 2.0;
    Value a = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    a.doubleDataMut()[0] = 1.0;
    a.doubleDataMut()[1] = 0.5;
    Value x = Value::matrix(1, 4, ValueType::DOUBLE, mr);
    for (int i = 0; i < 4; ++i)
        x.doubleDataMut()[i] = double(i + 1);
    Value zi = Value::scalar(0.0, mr);

    // Reference-validated values: filter([1 2], [1 0.5], 1:4, 0) gives
    // zf = 4.3125 and y = [1 3.5 5.25 7.375].
    signal::FilterState st = signal::filter(b, a, x, zi, mr);
    ASSERT_EQ(st.zf.numel(), 1u);
    EXPECT_NEAR(st.zf.doubleData()[0], 4.3125, 1e-12);
    ASSERT_EQ(st.y.numel(), 4u);
    EXPECT_NEAR(st.y.doubleData()[0], 1.0, 1e-12);
    EXPECT_NEAR(st.y.doubleData()[1], 3.5, 1e-12);
    EXPECT_NEAR(st.y.doubleData()[2], 5.25, 1e-12);
    EXPECT_NEAR(st.y.doubleData()[3], 7.375, 1e-12);

    // a(1) == 0 throws with the nk identifier on the direct path.
    Value a0z = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    a0z.doubleDataMut()[0] = 0.0;
    a0z.doubleDataMut()[1] = 1.0;
    try {
        (void)signal::filter(b, a0z, x, mr);
        FAIL() << "expected zeroLead";
    } catch (const Error &e) {
        EXPECT_EQ(e.identifier(), "numkit:filter:zeroLead");
    }
}

// ── fir1: argument-contract diagnostics ──────────────────────────────────
TEST_F(ArgcheckSignalTest, Fir1ArgcheckDiagnostics)
{
    expectError("fir1();", "numkit:fir1:nargin",
                "fir1: not enough input arguments — at least 2 required "
                "(got 0).");
    expectError("fir1(10);", "numkit:fir1:nargin",
                "fir1: not enough input arguments — at least 2 required "
                "(got 1).");
    expectError("fir1([2 3], 0.5);", "numkit:fir1:nNotScalar",
                "fir1: input 1 (n) must be a scalar (argument is 1x2).");
    expectError("fir1(2+3i, 0.5);", "numkit:fir1:nNotReal",
                "fir1: input 1 (n) must be real (argument is complex).");
    expectError("fir1(2.5, 0.5);", "numkit:fir1:orderNotInteger",
                "fir1: filter order n must be a positive integer (got 2.5).");
    expectError("fir1(-5, 0.5);", "numkit:fir1:nNotPositive",
                "fir1: input 1 (n) must be positive (got -5).");
    expectError("fir1(10, 0);", "numkit:fir1:WnNotPositive",
                "fir1: input 2 (Wn) must be positive (got 0).");
    expectError("fir1(10, -0.5);", "numkit:fir1:WnNotPositive",
                "fir1: input 2 (Wn) must be positive (got -0.5).");
    expectError("fir1(10, 1.0);", "numkit:fir1:wnOutOfRange",
                "fir1: cutoff frequency Wn must be inside the open "
                "interval (0, 1) (got 1).");
    expectError("fir1(10, 1.5);", "numkit:fir1:wnOutOfRange",
                "fir1: cutoff frequency Wn must be inside the open "
                "interval (0, 1) (got 1.5).");
    expectError("fir1(10, 0.5, 'bad');", "numkit:fir1:invalidType",
                "fir1: type must be one of: 'low', 'high' (got 'bad').");
}

// fir1 direct C++ call: the negative-order crash hole (size_t wrap on
// N+1) is closed by fir1_argcheck's first line.
TEST_F(ArgcheckSignalTest, Fir1DirectCApiContract)
{
    try {
        (void)signal::fir1(-5, 0.5, "low", std::pmr::get_default_resource());
        FAIL() << "expected nNotPositive";
    } catch (const Error &e) {
        EXPECT_EQ(e.identifier(), "numkit:fir1:nNotPositive");
        EXPECT_STREQ(e.what(), "fir1: input 1 (n) must be positive (got -5).");
    }
}

// ── butter / IIR family: order contract ──────────────────────────────────
// The family blames a complex order as not-real and a fractional one as
// not-integer (reference-validated classes); every message carries the
// received value.
TEST_F(ArgcheckSignalTest, ButterOrderArgcheckDiagnostics)
{
    expectError("butter();", "numkit:butter:nargin",
                "butter: not enough input arguments — at least 2 required "
                "(got 0).");
    expectError("butter([2 3], 0.5);", "numkit:butter:orderNotScalar",
                "butter: filter order N must be a scalar (argument is 1x2).");
    expectError("butter(2.5, 0.5);", "numkit:butter:orderNotInteger",
                "butter: filter order N must be a positive integer (got 2.5).");
    expectError("butter(2+3i, 0.5);", "numkit:butter:orderNotReal",
                "butter: filter order N must be real (argument is complex).");
    expectError("butter(-5, 0.5);", "numkit:butter:orderNotPositive",
                "butter: filter order N must be a positive integer (got -5).");
    expectError("butter(0, 0.5);", "numkit:butter:orderNotPositive",
                "butter: filter order N must be a positive integer (got 0).");
    // The rest of the IIR family shares the contract with its own fn id.
    expectError("cheby1(2.5, 0.5, 0.5);", "numkit:cheby1:orderNotInteger",
                "cheby1: filter order N must be a positive integer (got 2.5).");
    expectError("cheby2(2.5, 20, 0.5);", "numkit:cheby2:orderNotInteger",
                "cheby2: filter order N must be a positive integer (got 2.5).");
    expectError("ellip(2.5, 1, 40, 0.5);", "numkit:ellip:orderNotInteger",
                "ellip: filter order N must be a positive integer (got 2.5).");
    expectError("besself(2.5, 0.5);", "numkit:besself:orderNotInteger",
                "besself: filter order N must be a positive integer (got 2.5).");
    expectError("butter(-5, 0.5, 's');", "numkit:butter:orderNotPositive",
                "butter: filter order N must be a positive integer (got -5).");
}

// Direct C++ call: butter's compute-side first line guards the order for
// callers that bypass the interpreter.
TEST_F(ArgcheckSignalTest, ButterDirectCApiContract)
{
    try {
        std::pmr::memory_resource *mr = std::pmr::get_default_resource();
        Value wn = Value::scalar(0.5, mr);
        (void)signal::butter(-5, wn, signal::FilterType::Lowpass, false, mr);
        FAIL() << "expected orderNotPositive";
    } catch (const Error &e) {
        EXPECT_EQ(e.identifier(), "numkit:butter:orderNotPositive");
        EXPECT_STREQ(e.what(),
                     "butter: filter order N must be a positive integer "
                     "(got -5).");
    }
}

// ── 2026-09-17 signal sweep: count/order/window contracts ────────────
// Reference-probed (R2025b): counts are positive integers except where
// noted; maxlag is sign-free; window LENGTHS and lpc orders ROUND a
// fractional value instead of rejecting (rejection parity requires
// accepting them); uencode/udecode N is an integer in [2, 32].
TEST_F(ArgcheckSignalTest, SignalSweepBranchMatrix)
{
    expectError("fft(1:8, 2.5);", "numkit:fft:nNotInteger");
    expectError("fftshift(1:8, 1.5);", "numkit:fftshift:dimNotInteger");
    expectError("dct(1:8, 2.5);", "numkit:dct:nNotInteger");
    expectError("fwht(1:8, 2.5);", "numkit:fwht:nNotInteger");
    expectError("buffer(1:10, 2.5);", "numkit:buffer:nNotInteger");
    expectError("buffer(1:10, 3, 2.5);", "numkit:buffer:pNotInteger");
    expectError("uencode(1:10, 2.5);", "numkit:uencode:NNotInteger");
    expectError("uencode(1:10, 40);", "numkit:uencode:BadN");
    expectError("xcorr(1:8, 2.5);", "numkit:xcorr:maxlagNotInteger");
    expectError("downsample(1:8, 2.5);", "numkit:downsample:nNotInteger");
    expectError("decimate(1:8, 0);", "numkit:decimate:qNotPositive");
    expectError("freqz([1 0.5], 1, 2.5);", "numkit:freqz:nNotInteger");
    expectError("sgolay(2.5, 5);", "numkit:sgolay:orderNotInteger");
    expectError("sgolay(2, 4.5);", "numkit:sgolay:framelenNotInteger");
    expectError("periodogram(1:8, [], 2.5);", "numkit:periodogram:nfftNotInteger");
    expectError("spectrogram(1:16, 2.5);", "numkit:spectrogram:windowNotInteger");
    expectError("fir2(2.5, [0 1], [1 1]);", "numkit:fir2:NNotInteger");
}

TEST_F(ArgcheckSignalTest, SignalSweepHappyPaths)
{
    EXPECT_DOUBLE_EQ(evalScalar("numel(hamming(2.5))"), 3.0);   // rounds, not truncates
    EXPECT_DOUBLE_EQ(evalScalar("numel(kaiser(4.7))"), 5.0);
    EXPECT_DOUBLE_EQ(evalScalar("numel(xcorr(1:8, -3))"), 7.0); // windows to ±|maxlag|
    EXPECT_DOUBLE_EQ(evalScalar("numel(cconv(1:4, 1:4, 6))"), 6.0);
    EXPECT_DOUBLE_EQ(evalScalar("numel(decimate(1:16, 2))"), 8.0);
    EXPECT_DOUBLE_EQ(evalScalar("numel(interp(1:9, 2))"), 18.0);  // n*r (reference-probed, min length 2*4+1=9)
}

// ── 2026-09-18 structural portion: type contracts for the 33 ──────────
// Reference-probed against R2025b (build/probe_signal_argcheck.m):
// every rejection below matches the reference's rejection CLASS (which
// inputs are rejected and which argument is blamed), with nk-voiced
// identifiers and richer messages.

// conv family: numeric-or-logical coerces to double; char/cell are one
// inputType class; matrices/empties are notVector; a numeric shape
// argument is rejected (unknownShapeParameter class).
TEST_F(ArgcheckSignalTest, ConvFamilyTypeContract)
{
    eval("y = conv([true false true], [1 1]);");           // logical → double
    EXPECT_EQ(evalScalar("y(1)"), 1.0);
    eval("y = conv(int8([1 2 3]), [1 1 1]);");
    EXPECT_EQ(evalScalar("y(3)"), 6.0);                    // 1+2+3
    expectError("conv('ab', [1 1]);", "numkit:conv:inputType");
    expectError("conv({1 2}, [1 1]);", "numkit:conv:inputType");
    expectError("conv(single([1 2]), [1 1]);", "numkit:conv:singleUnsupported");
    expectError("conv(ones(2), [1 1]);", "numkit:conv:notVector");
    expectError("conv([], [1 1]);", "numkit:conv:notVector");
    expectError("conv([1 2], [1], 'zzz');", "numkit:conv:badShape");
    expectError("conv([1 2], [1], 0);", "numkit:conv:badShape");   // numeric shape rejected
}

// conv2 / filter2 / convn: logical+integer inputs coerce to double
// (reference-verified result class double); single/complex are voiced
// capability gaps; N-D rejected.
TEST_F(ArgcheckSignalTest, Conv2FamilyCoercionAndGaps)
{
    eval("y = conv2(true(2), ones(2));");
    EXPECT_EQ(evalScalar("numel(y)"), 9.0);
    EXPECT_EQ(evalScalar("y(1)"), 1.0);
    eval("y = filter2(int8(ones(3)), ones(4));");
    EXPECT_EQ(evalScalar("numel(y)"), 16.0);
    eval("y = convn(logical(ones(2)), ones(2));");
    EXPECT_EQ(evalScalar("numel(y)"), 9.0);
    expectError("conv2(single(ones(2)), ones(2));", "numkit:conv2:singleUnsupported");
    expectError("conv2(ones(2)+1i, ones(2));", "numkit:conv2:complexUnsupported");
    expectError("filter2(ones(3), ones(2,2,2));", "numkit:filter2:nd");
    expectError("conv2(ones(2), ones(2), 0);", "numkit:conv2:firstTwoInputsNotVectors");
    expectError("conv2(ones(2), ones(2), 'zzz');", "numkit:conv2:badShape");
    expectError("filter2(ones(3), ones(4), 'zzz');", "numkit:filter2:badShape");
}

// convmtx: double/single/complex pass with class-preserving results;
// logical is rejected (reference mustBeA list has no logical);
// integer classes are a voiced gap; h must be a vector.
TEST_F(ArgcheckSignalTest, ConvmtxClassPreserving)
{
    eval("h = single([1 1]); y = convmtx(h, 3);");
    EXPECT_EQ(evalScalar("y(1,2)"), 1.0);
    eval("y = convmtx([1 1i], 3);");
    EXPECT_EQ(evalScalar("y(1,1)"), 1.0);                  // h(1) = 1 lands on the diagonal
    EXPECT_EQ(evalScalar("imag(y(1,2))"), 1.0);            // h(2) = 1i shifted right
    expectError("convmtx([true true], 3);", "numkit:convmtx:inputType");
    expectError("convmtx(int8([1 1]), 3);", "numkit:convmtx:integerUnsupported");
    expectError("convmtx(ones(2), 3);", "numkit:convmtx:notVector");
    expectError("convmtx('aa', 3);", "numkit:convmtx:inputType");
}

// xcorr2: numeric-or-logical coerced; complex/single/N-D voiced.
TEST_F(ArgcheckSignalTest, Xcorr2TypeContract)
{
    eval("y = xcorr2(logical(ones(3)));");
    EXPECT_EQ(evalScalar("numel(y)"), 25.0);
    eval("y = xcorr2(int8(ones(3)));");
    EXPECT_EQ(evalScalar("y(13)"), 9.0);                   // center of the 5×5 peak
    expectError("xcorr2(single(ones(3)));", "numkit:xcorr2:singleUnsupported");
    expectError("xcorr2(ones(3)+1i);", "numkit:xcorr2:complexUnsupported");
    expectError("xcorr2('abc');", "numkit:xcorr2:inputType");
    expectError("xcorr2(ones(2,2,2));", "numkit:xcorr2:nd");
}

// polyscale / polystab: double/complex pass; logical/integer rejected
// (reference sigcheckfloattype / eig inputType classes).
TEST_F(ArgcheckSignalTest, PolyUtilsTypeContract)
{
    expectError("polyscale([true true true], 2);", "numkit:polyscale:inputType");
    expectError("polyscale(int8([1 2 3]), 2);", "numkit:polyscale:inputType");
    expectError("polyscale(single([1 2 3]), 2);", "numkit:polyscale:singleUnsupported");
    expectError("polystab(logical([1 1 1]));", "numkit:polystab:inputType");
    expectError("polystab(int8([1 2 3]));", "numkit:polystab:inputType");
    expectError("polystab(single([1 2 3]));", "numkit:polystab:singleUnsupported");
    expectError("polystab(ones(2));", "numkit:polystab:notVector");
    expectError("polystab(complex([1 2], [0 1]));", "numkit:polystab:complexUnsupported");
    eval("y = polyscale(complex([1 2], [0 1]), 2);");      // complex p passes
}

// unshiftdata: perm validated for THIS entry; nshifts scalar.
TEST_F(ArgcheckSignalTest, UnshiftdataPermContract)
{
    expectError("unshiftdata(ones(2,3,2), [1 1 2], 1);", "numkit:unshiftdata:badPerm");
    expectError("unshiftdata(ones(2,3,2), [1.5 2 3], 1);", "numkit:unshiftdata:badPerm");
    expectError("unshiftdata(ones(2,3,2), [2 3 1], [1 2]);", "numkit:unshiftdata:nshifts");
    eval("y = unshiftdata(ones(2,3,2), [2 3 1], 1);");     // valid perm passes
}

// sosfilt / sosfiltfilt: floating-point x only (reference VectMustBeFloat
// class); single/complex voiced gaps; SOS shape voiced per entry.
TEST_F(ArgcheckSignalTest, SosfiltTypeContract)
{
    eval("sos = [1 1 1 1 0.5 0.25];");
    expectError("sosfilt(sos, logical(1:8));", "numkit:sosfilt:xType");
    expectError("sosfilt(sos, int8(1:8));", "numkit:sosfilt:xType");
    expectError("sosfilt(sos, 'abcdefgh');", "numkit:sosfilt:xType");
    expectError("sosfilt(sos, single(1:8));", "numkit:sosfilt:singleUnsupported");
    expectError("sosfilt(sos, (1:8)+1i);", "numkit:sosfilt:complexUnsupported");
    expectError("sosfilt([1 1 1 1 0.5], 1:8);", "numkit:sosfilt:sosShape");
    expectError("sosfiltfilt(sos, logical(1:8));", "numkit:sosfiltfilt:xType");
    expectError("sosfiltfilt([1 1 1 1 0.5], 1:8);", "numkit:sosfiltfilt:sosShape");
    expectError("sosfiltfilt(sos, int8(1:8));", "numkit:sosfiltfilt:xType");
}

// zp2sos / tf2sos: coefficient type contracts (logical rejected by the
// reference; single/complex voiced gaps).
TEST_F(ArgcheckSignalTest, Zp2sosTf2sosTypeContract)
{
    expectError("zp2sos(logical([0; 0]), [0.2; 0.3], 1);", "numkit:zp2sos:inputType");
    expectError("zp2sos(int8([1; 2]), [0.2; 0.3], 1);", "numkit:zp2sos:inputType");
    expectError("zp2sos(single([0.5; 0.6]), [0.2; 0.3], 1);", "numkit:zp2sos:singleUnsupported");
    expectError("tf2sos(logical([1 1]), [1 -0.5]);", "numkit:tf2sos:type");
    expectError("tf2sos(int8([1 1]), [1 -0.5]);", "numkit:tf2sos:type");
    expectError("tf2sos(single([1 1]), [1 -0.5]);", "numkit:tf2sos:singleUnsupported");
    expectError("tf2sos([1 1i], [1 -0.5]);", "numkit:tf2sos:complexUnsupported");
}

// sigroi family: N×2 roi contract (reference incorrectNumcols class);
// scalar content contracts; extractsigroi x vector + complex slices.
TEST_F(ArgcheckSignalTest, SigroiContracts)
{
    expectError("sigroi2binmask([1 3 5 7], 10);", "numkit:sigroi2binmask:roiCols");
    expectError("sigroi2binmask(logical([1 0; 1 1]), 10);", "numkit:sigroi2binmask:inputType");
    expectError("sigroi2binmask([1 3; 5 7], 0);", "numkit:sigroi2binmask:lenPositive");
    eval("y = sigroi2binmask(int8([1 3; 5 7]), 10);");      // int8 roi OK (reference)
    expectError("extendsigroi([1 3 5 7], 1, 1);", "numkit:extendsigroi:roiCols");
    expectError("extendsigroi([1 3; 5 7], -1, 1);", "numkit:extendsigroi:nonnegative");
    expectError("mergesigroi([1 3 5 7], 2);", "numkit:mergesigroi:roiCols");
    expectError("mergesigroi([1 3; 5 7], -1);", "numkit:mergesigroi:nonnegative");
    expectError("removesigroi([1 3 5 7], 3);", "numkit:removesigroi:roiCols");
    expectError("shortensigroi([1 5 7 9], 1, 2);", "numkit:shortensigroi:roiCols");
    expectError("extractsigroi(1:10, [1 3 5 7]);", "numkit:extractsigroi:roiCols");
    expectError("extractsigroi(logical(1:10), [1 3; 5 7]);", "numkit:extractsigroi:inputType");
    expectError("extractsigroi(reshape(1:12, 3, 4), [1 3; 5 7]);", "numkit:extractsigroi:notVector");
    expectError("extractsigroi(single(1:10), [1 3; 5 7]);", "numkit:extractsigroi:singleUnsupported");
    // Complex x keeps its imaginary part through both output forms.
    eval("c = extractsigroi((1:10)+1i, [1 3; 5 7]);");
    EXPECT_NEAR(evalScalar("imag(c{1}(1))"), 1.0, 1e-12);
    eval("cc = extractsigroi((1:10)+1i, [1 3; 5 7], true);");
    EXPECT_NEAR(evalScalar("imag(cc(1))"), 1.0, 1e-12);
}

// multirate: typed gathers preserve class and handle matrices
// column-wise (reference-verified); integer classes and N-D are voiced
// gaps; phase contract now lives in the compute.
TEST_F(ArgcheckSignalTest, MultirateTypedGathers)
{
    eval("y = downsample(single(1:8), 2);");
    EXPECT_EQ(evalScalar("y(1)"), 1.0);
    eval("y = upsample(logical([1 0 1]), 2);");
    EXPECT_EQ(evalScalar("y(5)"), 1.0);
    eval("y = downsample((1:8)+1i, 2);");                   // complex preserved
    EXPECT_NEAR(evalScalar("imag(y(1))"), 1.0, 1e-12);
    eval("m = reshape(1:8, 4, 2); y = downsample(m, 2);");
    EXPECT_EQ(evalScalar("size(y, 1) * 10 + size(y, 2)"), 22.0);
    EXPECT_EQ(evalScalar("y(1,2)"), 5.0);                   // column 2 starts at 5
    eval("m = reshape(1:6, 3, 2); y = upsample(m, 2);");
    EXPECT_EQ(evalScalar("size(y, 1)"), 6.0);
    expectError("downsample(int8(1:8), 2);", "numkit:downsample:integerUnsupported");
    expectError("upsample(int8(1:3), 2);", "numkit:upsample:integerUnsupported");
    expectError("downsample({1 2}, 2);", "numkit:downsample:inputType");
    expectError("decimate(logical(1:12), 3);", "numkit:decimate:inputType");
    expectError("decimate(int8(1:12), 3);", "numkit:decimate:inputType");
    expectError("decimate(single(1:12), 3);", "numkit:decimate:singleUnsupported");
    expectError("resample(logical(1:10), 3, 2);", "numkit:resample:inputType");
    expectError("resample(single(1:10), 3, 2);", "numkit:resample:singleUnsupported");
}

// medfilt1: real-double contract; complex rejected (reference
// expectedReal class); k = 0 answers all-NaN (reference-probed).
TEST_F(ArgcheckSignalTest, Medfilt1Contract)
{
    expectError("medfilt1(logical(1:8), 3);", "numkit:medfilt1:inputType");
    expectError("medfilt1(int8(1:8), 3);", "numkit:medfilt1:inputType");
    expectError("medfilt1((1:8)+1i, 3);", "numkit:medfilt1:inputReal");
    expectError("medfilt1(single(1:8), 3);", "numkit:medfilt1:singleUnsupported");
    eval("y = medfilt1(1:8, 0);");
    EXPECT_TRUE(std::isnan(evalScalar("y(4)")));
}

// bandpower: fs requires freqrange (reference FreqRangeMissing class);
// freqrange exactly 2 elements; complex 1-arg power = mean(|x|^2);
// 'psd' forms voiced as unimplemented.
TEST_F(ArgcheckSignalTest, BandpowerContract)
{
    eval("x = cos(1:100);");
    expectError("bandpower(x, 10);", "numkit:bandpower:freqRangeRequired");
    expectError("bandpower(x, 10, [1 2 3]);", "numkit:bandpower:freqRange");
    expectError("bandpower(x, 'a');", "numkit:bandpower:freqRangeRequired");
    expectError("bandpower(logical(1:100));", "numkit:bandpower:inputType");
    expectError("bandpower(int8(1:100));", "numkit:bandpower:inputType");
    expectError("bandpower(single(1:100));", "numkit:bandpower:singleUnsupported");
    expectError("bandpower(x, 10, [1 5], 'psd');", "numkit:bandpower:freqVecLength");
    // Complex 1-arg: mean(|x|^2) — reference-exact (probed 26.5 for
    // x = (1:8)+1i).
    EXPECT_NEAR(evalScalar("bandpower((1:8)+1i)"), 26.5, 1e-12);
}

// stft: double/complex pass; single/logical/integer rejected
// (reference InvalidInputDataType); 3-D rejected (expected2D); matrix
// input is a voiced gap (the reference treats columns as channels).
TEST_F(ArgcheckSignalTest, StftTypeContract)
{
    eval("s = stft(sin(1:128));");                          // double OK
    eval("s = stft((1:128)+1i);");                          // complex OK
    expectError("stft(single(sin(1:128)));", "numkit:stft:singleUnsupported");
    expectError("stft(logical(1:128));", "numkit:stft:inputType");
    expectError("stft(int8(1:128));", "numkit:stft:inputType");
    expectError("stft(ones(16, 4, 2));", "numkit:stft:expected2D");
    expectError("stft(ones(64, 2));", "numkit:stft:matrixUnsupported");
}

// czt / ifft('symmetric') / goertzel / bitrevorder.
TEST_F(ArgcheckSignalTest, TransformsTypeContract)
{
    expectError("czt(logical(1:8));", "numkit:czt:inputType");
    expectError("czt(int8(1:8));", "numkit:czt:inputType");
    expectError("czt(single(1:8));", "numkit:czt:singleUnsupported");
    expectError("czt(1:8, 8, [1 1], 1);", "numkit:czt:wNotScalar");
    eval("y = czt(1:8);");                                  // double passes
    eval("y = czt((1:8)+1i);");                             // complex passes
    expectError("ifft('abcd', 'symmetric');", "numkit:ifft:inputType");
    eval("y = ifft(logical([1 0 1 0]), 'symmetric');");     // logical coerces
    expectError("goertzel(logical(1:8), [2 5]);", "numkit:goertzel:inputType");
    expectError("goertzel(int8(1:8), [2 5]);", "numkit:goertzel:inputType");
    expectError("goertzel(single(1:8), [2 5]);", "numkit:goertzel:singleUnsupported");
    expectError("goertzel({1 2}, [2 5]);", "numkit:goertzel:inputType");
    // Complex goertzel: reference-exact values (probed R2025b).
    eval("g = goertzel((1:8)+1i, [2 5]);");
    EXPECT_NEAR(evalScalar("real(g(1))"), -4.0, 1e-9);
    EXPECT_NEAR(evalScalar("imag(g(1))"), 9.65685424949238, 1e-9);
    expectError("bitrevorder('abcdefgh');", "numkit:bitrevorder:inputType");
    expectError("bitrevorder(int8(1:8));", "numkit:bitrevorder:integerUnsupported");
    eval("y = bitrevorder(single(1:8));");                  // single preserved
    EXPECT_EQ(evalScalar("y(2)"), 5.0);
    eval("y = bitrevorder(logical([1 0 0 0]));");           // logical preserved
    EXPECT_EQ(evalScalar("y(1)"), 1.0);
}

// modulate / demod: Fc/Fs real-scalar marshalling; x float contract.
TEST_F(ArgcheckSignalTest, ModulateDemodContract)
{
    expectError("modulate(1:10, 'a', 100, 'am');", "numkit:modulate:FcNotReal");
    expectError("modulate(1:10, [1 2], 100, 'am');", "numkit:modulate:FcNotScalar");
    expectError("modulate(1:10, 10i, 100, 'am');", "numkit:modulate:FcNotReal");
    expectError("modulate(logical(1:10), 10, 100, 'am');", "numkit:modulate:inputType");
    expectError("modulate(int8(1:10), 10, 100, 'am');", "numkit:modulate:inputType");
    expectError("modulate(single(1:10), 10, 100, 'am');", "numkit:modulate:singleUnsupported");
    expectError("demod(1:10, 'a', 100, 'am');", "numkit:demod:FcNotReal");
    expectError("demod(1:10, 10, 100, 'zz');", "numkit:demod:UnsupportedMethod");
}

// Direct C++ path: the contracts live in the compute, so a caller that
// bypasses the interpreter crosses the same checks (both-paths rule).
TEST_F(ArgcheckSignalTest, DirectCApiContracts)
{
    std::pmr::memory_resource *mr = std::pmr::get_default_resource();
    try {
        (void)signal::downsample(Value::cell(1, 2, mr), 2, 0, mr);
        FAIL() << "expected downsample inputType";
    } catch (const Error &e) {
        EXPECT_EQ(e.identifier(), "numkit:downsample:inputType");
    }
    try {
        Value x = Value::matrix(1, 4, ValueType::DOUBLE, mr);
        (void)signal::medfilt1(x, 0, true, mr);             // k=0 legal now
    } catch (const Error &) {
        FAIL() << "medfilt1 k=0 must not throw";
    }
}
