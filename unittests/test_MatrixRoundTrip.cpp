#include <complex>
#include <iostream>

#include <Eigen/Dense>
#include <H5Cpp.h>

#include "eigen3-hdf5.hpp"
#include "gtest-helpers.hpp"

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
    ASSERT_PRED_FORMAT2(assert_same, mat, mat2);
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
    mat <<
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16;

    mat2(0, 2) = 3; mat2(0, 3) = 4;
    mat2(1, 2) = 7; mat2(1, 3) = 8;
    mat2(2, 0) = 9; mat2(2, 1) = 10; mat2(2, 2) = 11; mat2(2, 3) = 12;
    mat2(3, 0) = 13; mat2(3, 1) = 14; mat2(3, 2) = 15; mat2(3, 3) = 16;
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
        // read into a dynamic sized matrix and then copy into fixed size
        EigenHDF5::load(file, "double_matrix", dmat2);
        EigenHDF5::load(file, "matrix_block", dmatblock2);
        fmat2 = dmat2;
        fmatblock2 = dmatblock2;
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
        EigenHDF5::load(file, "double_matrix", fmat2);
        EigenHDF5::load(file, "matrix_block", fmatblock2);
    }
    ASSERT_PRED_FORMAT2(assert_same, mat, fmat2);
    ASSERT_PRED_FORMAT2(assert_same, matblock, fmatblock2);
}
