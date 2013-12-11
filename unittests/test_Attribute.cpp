#include <Eigen/Dense>
#include <H5Cpp.h>

#include "eigen3-hdf5.hpp"

#include <gtest/gtest.h>

TEST(Attribute, Matrix) {
}

TEST(Attribute, Integer) {
    H5::H5File file("/tmp/test_Attribute_Integer.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "integer", 23);
}

TEST(Attribute, Double) {
    H5::H5File file("/tmp/test_Attribute_Double.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "double", 23.7);
}

TEST(Attribute, String) {
    H5::H5File file("/tmp/test_Attribute_String.h5", H5F_ACC_TRUNC);
    EigenHDF5::save_scalar_attribute(file, "str1", std::string("hello"));
    EigenHDF5::save_scalar_attribute(file, "str2", "goodbye");
    char *s = "again";
    EigenHDF5::save_scalar_attribute(file, "str3", s);
}
