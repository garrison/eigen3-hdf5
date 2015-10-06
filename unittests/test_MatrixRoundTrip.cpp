#include <complex>
#include <iostream>

#include <Eigen/Dense>
#include <H5Cpp.h>

#include "eigen3-hdf5.hpp"

#include <gtest/gtest.h>

TEST(MatrixRoundTrip, Double) {
    Eigen::MatrixXd mat(3, 4), mat2;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_Double.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "double_matrix", mat);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_Double.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "double_matrix", mat2);
    }
    ASSERT_EQ(mat, mat2);
}

TEST(MatrixRoundTrip, LongDouble) {
    Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> mat(3, 4), mat2;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_LongDouble.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "longdouble_matrix", mat);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_LongDouble.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "longdouble_matrix", mat2);
    }
    ASSERT_EQ(mat, mat2);
}

TEST(MatrixRoundTrip, Int) {
    Eigen::MatrixXi mat(3, 4), mat2;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_Int.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "int_matrix", mat);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_Int.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "int_matrix", mat2);
    }
    ASSERT_EQ(mat, mat2);
}

TEST(MatrixRoundTrip, ULongLong) {
    Eigen::Matrix<unsigned long long, Eigen::Dynamic, Eigen::Dynamic> mat(3, 4), mat2;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_ULongLong.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "ull_matrix", mat);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_ULongLong.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "ull_matrix", mat2);
    }
    ASSERT_EQ(mat, mat2);
}

TEST(MatrixRoundTrip, ComplexDouble) {
    Eigen::MatrixXcd mat(3, 4), mat2;
    mat << 1, std::complex<double>(0, 2), 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_ComplexDouble.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "complex_matrix", mat);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_ComplexDouble.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "complex_matrix", mat2);
    }
    ASSERT_EQ(mat, mat2);
}

namespace Eigen {

// utility function to print an eigen object to an ostream; gtest will use then when it
// outputs a matrix. Without it, gtest seems to dump some kind of byte representation,
// which is not very helpful. 
template <class Derived>
void PrintTo(const Eigen::EigenBase<Derived>& mat, ::std::ostream* os)
{
    (*os) << mat.derived() << "\n";
}

// utility function for gtest to check if two eigen objects are identical. returns
// assertion success when they are; returns assertion failure along with a nicely
// formatted message with the matrix contents when they are not identical. The C++ and/or
// gtest rules requires that the method used by ASSERT_PRED_FORMAT2 be in the namespace
// of its argument.
// 
// I put this function in this location of this matrix test cpp file for a few reasons:
// 1) there is not already a header file to put common test code for eigen3-hdf5, 2)
// because all the other tests were already passing, so they would not be helped by this
// new assert, and 3) because I needed it to help me debug test failures for
// DoubleSkipInternalCopyBlock (It took me a few attempts to implement EigenHDF5::save
// for a row major block matrix with a stride that is not equal to the number of columns
// in the original matrix.) I really think that (2, a header for common test code) should
// be addressed at some point, and then this function (and its companion PrintTo) should be moved
// there. 

template <class DerivedExp, class DerivedAct>
::testing::AssertionResult assert_same(const char* exp_expr,
    const char* act_expr,
    const Eigen::EigenBase<DerivedExp>& exp,
    const Eigen::EigenBase<DerivedAct>& act)
{
    if (exp.rows() == act.rows() &&
        exp.cols() == act.cols() &&
        exp.derived() == act.derived())
    {
        return ::testing::AssertionSuccess();
    }

    // if eigen did not define the == operator, you could use
    // exp.derived().cwiseEqual(act.derived()).all();

    ::testing::AssertionResult result = ::testing::AssertionFailure()
        << "Eigen objects are not the same: ("
        << exp_expr << ", " << act_expr << ")\n"
        << exp_expr << ":\n"
        << ::testing::PrintToString(exp)
        << "\n---and\n" << act_expr << ":\n"
        << ::testing::PrintToString(act)
        << "\n---are not equal!\n";

    return result;
}
} // namespace Eigen

TEST(MatrixRoundTrip, IntBlock) {
    Eigen::Matrix4i mat(Eigen::Matrix4i::Zero());
    Eigen::Matrix4i mat2(Eigen::Matrix4i::Zero());
    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(1, 0) = 3;
    mat(1, 1) = 4;
    mat(2, 2) = 5;
    mat2(2, 2) = 5;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_IntBlock.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "int_block", mat.block(0, 0, 2, 2));
    }
    {
        H5::H5File file("test_MatrixRoundTrip_IntBlock.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "int_block", mat2.block(0, 0, 2, 2));
    }
    ASSERT_PRED_FORMAT2(assert_same, mat, mat2);
}

TEST(MatrixRoundTrip, IntBlockRowMajor) {
    typedef Eigen::Matrix<int, 4, 4, Eigen::RowMajor> Matrix4RowMajor;
    Matrix4RowMajor mat(Eigen::Matrix4i::Zero());
    Matrix4RowMajor mat2(Eigen::Matrix4i::Zero());
    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(1, 0) = 3;
    mat(1, 1) = 4;
    mat(2, 2) = 5;
    mat2(2, 2) = 5;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_IntBlockRowMajor.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "int_block", mat.block(0, 0, 2, 2));
    }
    {
        H5::H5File file("test_MatrixRoundTrip_IntBlockRowMajor.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "int_block", mat2.block(0, 0, 2, 2));
    }
    ASSERT_PRED_FORMAT2(assert_same, mat, mat2);
}

TEST(MatrixRoundTrip, DoubleSkipInternalCopy) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(3, 4); // , mat2;
    Eigen::MatrixXd mat2;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleSkipInternalCopy.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "double_matrix", mat);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleSkipInternalCopy.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "double_matrix", mat2);
    }
    ASSERT_PRED_FORMAT2(assert_same, mat, mat2);
}

TEST(MatrixRoundTrip, DoubleSkipInternalCopyBlock) {
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MyMatrixXdRowMajor;
    MyMatrixXdRowMajor mat(3, 4);
    Eigen::Block<MyMatrixXdRowMajor> matblock = mat.block(1, 1, 2, 3);
    Eigen::MatrixXd mat2;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleSkipInternalCopy.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "double_matrix", matblock);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleSkipInternalCopy.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "double_matrix", mat2);
    }
    ASSERT_PRED_FORMAT2(assert_same, matblock, mat2);
}

TEST(MatrixRoundTrip, DoubleFixedRow) {
    typedef Eigen::Matrix<double, 4, 6, Eigen::RowMajor> MyMatrix46RowMajor;
    MyMatrix46RowMajor mat;
    Eigen::Block<MyMatrix46RowMajor> matblock = mat.block(1, 2, 2, 3);
    MyMatrix46RowMajor fmat2;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> fmatblock2;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dmat2, dmatblock2;

    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleFixedRow.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "double_matrix", mat);
        EigenHDF5::save(file, "matrix_block", matblock);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleFixedRow.h5", H5F_ACC_RDONLY);
#if 0
        // this won't compile because load has a transposeInPlace, which is not allowed for
        // fixed size matrices. 
        EigenHDF5::load(file, "double_matrix", fmat2);
        EigenHDF5::load(file, "matrix_block", fmatblock2);
#else
        // read into a dynamic sized matrix and then copy into fixed size
        EigenHDF5::load(file, "double_matrix", dmat2);
        EigenHDF5::load(file, "matrix_block", dmatblock2);
        fmat2 = dmat2;
        fmatblock2 = dmatblock2;
#endif
    }
    ASSERT_PRED_FORMAT2(assert_same, mat, fmat2);
    ASSERT_PRED_FORMAT2(assert_same, matblock, fmatblock2);
}

TEST(MatrixRoundTrip, DoubleFixedCol) {
    typedef Eigen::Matrix<double, 4, 6, Eigen::ColMajor> MyMatrix46ColMajor;
    MyMatrix46ColMajor mat;
    Eigen::Block<MyMatrix46ColMajor> matblock = mat.block(1, 2, 2, 3);
    MyMatrix46ColMajor fmat2;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> fmatblock2;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dmat2, dmatblock2;

    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
#ifdef LOGGING
    std::cout << mat << std::endl;
#endif
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleFixedRow.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "double_matrix", mat);
        EigenHDF5::save(file, "matrix_block", matblock);
    }
    {
        H5::H5File file("test_MatrixRoundTrip_DoubleFixedRow.h5", H5F_ACC_RDONLY);
#if 0
        // this won't compile because load has a transposeInPlace, which is not allowed for
        // fixed size matrices. 
        EigenHDF5::load(file, "double_matrix", fmat2);
        EigenHDF5::load(file, "matrix_block", fmatblock2);
#else
        // read into a dynamic sized matrix and then copy into fixed size
        EigenHDF5::load(file, "double_matrix", dmat2);
        EigenHDF5::load(file, "matrix_block", dmatblock2);
        fmat2 = dmat2;
        fmatblock2 = dmatblock2;
#endif
    }
    ASSERT_PRED_FORMAT2(assert_same, mat, fmat2);
    ASSERT_PRED_FORMAT2(assert_same, matblock, fmatblock2);
}

// To run all of the EigenHDF5 tests use:
// -- --gtest_filter=Attribute*:Matrix*:Vector*
