#include <complex>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <H5Cpp.h>

#include "eigen3-hdf5-sparse.hpp"

#include <gtest/gtest.h>

TEST(SparseMatrix, Double) {
    Eigen::SparseMatrix<double> mat(3, 3), mat2;
    mat.insert(0, 1) = 2.7;
    mat.insert(2, 0) = 82;
    {
        H5::H5File file("/tmp/test_SparseMatrix_Double.h5", H5F_ACC_TRUNC);
        EigenHDF5::save_sparse(file, "mat", mat);
    }
    {
        H5::H5File file("/tmp/test_SparseMatrix_Double.h5", H5F_ACC_RDONLY);
        EigenHDF5::load_sparse(file, "mat", mat2);
    }
#ifdef LOGGING
    std::cout << mat2 << std::endl;
#endif
    ASSERT_EQ(Eigen::MatrixXd(mat), Eigen::MatrixXd(mat2));
}

TEST(SparseMatrix, Complex) {
    Eigen::SparseMatrix<std::complex<double> > mat(4, 4), mat2;
    mat.insert(0, 1) = std::complex<double>(2, 4.5);
    mat.insert(1, 2) = std::complex<double>(82, 1);
    {
        H5::H5File file("/tmp/test_SparseMatrix_Complex.h5", H5F_ACC_TRUNC);
        EigenHDF5::save_sparse(file, "mat", mat);
    }
    {
        H5::H5File file("/tmp/test_SparseMatrix_Complex.h5", H5F_ACC_RDONLY);
        EigenHDF5::load_sparse(file, "mat", mat2);
    }
#ifdef LOGGING
    std::cout << mat2 << std::endl;
#endif
    ASSERT_EQ(Eigen::MatrixXcd(mat), Eigen::MatrixXcd(mat2));
}
