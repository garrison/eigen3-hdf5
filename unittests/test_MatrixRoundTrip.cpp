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
