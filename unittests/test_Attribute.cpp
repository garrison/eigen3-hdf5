#include <Eigen/Dense>
#include <H5Cpp.h>

#include "eigen3-hdf5.hpp"
#include "gtest-helpers.hpp"

TEST(Attribute, Matrix) {
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> rmat1, rmat2;
    Eigen::Matrix<double, 2, 3, Eigen::ColMajor> cmat1, cmat2;
    rmat1 << 1, 2, 3, 4, 5, 6;
    cmat1 << 1, 2, 3, 4, 5, 6;
    {
        H5::H5File file("test_Attribute_Matrix.h5", H5F_ACC_TRUNC);
        EigenHDF5::save_attribute(file, "rowmat", rmat1);
        EigenHDF5::save_attribute(file, "colmat", cmat1);
    }
    {
        H5::H5File file("test_Attribute_Matrix.h5", H5F_ACC_RDONLY);
        EigenHDF5::load_attribute(file, "rowmat", rmat2);
        EigenHDF5::load_attribute(file, "colmat", cmat2);
    }
    ASSERT_PRED_FORMAT2(assert_same, rmat1, rmat2);
    ASSERT_PRED_FORMAT2(assert_same, cmat1, cmat2);
    ASSERT_PRED_FORMAT2(assert_same, rmat2, cmat2);
}

TEST(Attribute, Integer) {
    H5::H5File file("test_Attribute_Integer.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "integer", 23);
}

TEST(Attribute, Double) {
    H5::H5File file("test_Attribute_Double.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "double", 23.7);
}

TEST(Attribute, String) {
    H5::H5File file("test_Attribute_String.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "str1", std::string("hello"));
    EigenHDF5::save_scalar_attribute(file, "str2", "goodbye");
    const char *s = "again";
    EigenHDF5::save_scalar_attribute(file, "str3", s);
}
