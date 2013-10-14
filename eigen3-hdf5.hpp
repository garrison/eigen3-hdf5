#ifndef _EIGEN3_HDF5_HPP
#define _EIGEN3_HDF5_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

//#include <hdf5.h>
#include <H5Cpp.h>
#include <Eigen/Dense>

namespace EigenHDF5
{

template <typename T>
H5::PredType get_datatype (void);

template <>
H5::PredType get_datatype<float> (void)
{
    return H5::PredType::NATIVE_FLOAT;
}

template <>
H5::PredType get_datatype<double> (void)
{
    return H5::PredType::NATIVE_DOUBLE;
}

template <>
H5::PredType get_datatype<long double> (void)
{
    return H5::PredType::NATIVE_LDOUBLE;
}

template <>
H5::PredType get_datatype<int> (void)
{
    return H5::PredType::NATIVE_INT;
}

template <>
H5::PredType get_datatype<unsigned int> (void)
{
    return H5::PredType::NATIVE_UINT;
}

// see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html

template <typename Derived>
void save (H5::H5File &file, const std::string &name, const Eigen::EigenBase<Derived> &mat)
{
    typedef typename Derived::Scalar Scalar;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);
    const std::array<hsize_t, 2> dimensions = { {
        static_cast<hsize_t>(mat.rows()),
        static_cast<hsize_t>(mat.cols())
    } };
    H5::DataSpace dataspace(dimensions.size(), dimensions.data());
    H5::PredType datatype = get_datatype<Scalar>();
    H5::DataSet dataset = file.createDataSet(name, datatype, dataspace);
    dataset.write(row_major_mat.data(), datatype);
}

template <typename Derived>
void load (H5::H5File &file, const std::string &name, const Eigen::DenseBase<Derived> &mat)
{
    typedef typename Derived::Scalar Scalar;
    H5::DataSet dataset = file.openDataSet(name);
    H5::DataSpace dataspace = dataset.getSpace();
    const std::size_t ndims = dataspace.getSimpleExtentNdims();
    assert(ndims > 0);
    std::array<hsize_t, 2> dimensions;
    dimensions[1] = 1; // in case it's 1D
    if (ndims > dimensions.size()) {
        throw std::runtime_error("HDF5 array has too many dimensions.");
    }
    dataspace.getSimpleExtentDims(dimensions.data());
    const hsize_t rows = dimensions[0], cols = dimensions[1];
    std::vector<Scalar> data(rows * cols);
    const H5::PredType datatype = get_datatype<Scalar>();
    dataset.read(data.data(), datatype, dataspace);
    // see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
    Eigen::DenseBase<Derived> &mat_ = const_cast<Eigen::DenseBase<Derived> &>(mat);
    mat_.derived().resize(rows, cols);
    mat_ = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(data.data(), rows, cols);
}

} // namespace EigenHDF5

#endif
