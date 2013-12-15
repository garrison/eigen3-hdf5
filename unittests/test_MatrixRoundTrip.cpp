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
        H5::H5File file("/tmp/test_MatrixRoundTrip_Double.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "double_matrix", mat);
    }
    {
        H5::H5File file("/tmp/test_MatrixRoundTrip_Double.h5", H5F_ACC_RDONLY);
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
        H5::H5File file("/tmp/test_MatrixRoundTrip_LongDouble.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "longdouble_matrix", mat);
    }
    {
        H5::H5File file("/tmp/test_MatrixRoundTrip_LongDouble.h5", H5F_ACC_RDONLY);
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
        H5::H5File file("/tmp/test_MatrixRoundTrip_Int.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "int_matrix", mat);
    }
    {
        H5::H5File file("/tmp/test_MatrixRoundTrip_Int.h5", H5F_ACC_RDONLY);
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
        H5::H5File file("/tmp/test_MatrixRoundTrip_ULongLong.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "ull_matrix", mat);
    }
    {
        H5::H5File file("/tmp/test_MatrixRoundTrip_ULongLong.h5", H5F_ACC_RDONLY);
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
        H5::H5File file("/tmp/test_MatrixRoundTrip_ComplexDouble.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "complex_matrix", mat);
    }
    {
        H5::H5File file("/tmp/test_MatrixRoundTrip_ComplexDouble.h5", H5F_ACC_RDONLY);
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
        H5::H5File file("/tmp/test_MatrixRoundTrip_IntBlock.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "int_block", mat.block(0, 0, 2, 2));
    }
    {
        H5::H5File file("/tmp/test_MatrixRoundTrip_IntBlock.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "int_block", mat2.block(0, 0, 2, 2));
    }
    ASSERT_EQ(mat, mat2);
}
