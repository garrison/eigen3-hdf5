#ifndef _EIGEN3_HDF5_HPP
#define _EIGEN3_HDF5_HPP

#include <array>
#include <cassert>
#include <complex>
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
struct DatatypeSpecialization;

// floating-point types

template <>
struct DatatypeSpecialization<float>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_FLOAT;
        }
};

template <>
struct DatatypeSpecialization<double>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_DOUBLE;
        }
};

template <>
struct DatatypeSpecialization<long double>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_LDOUBLE;
        }
};

// integer types

template <>
struct DatatypeSpecialization<int>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_INT;
        }
};

template <>
struct DatatypeSpecialization<unsigned int>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_UINT;
        }
};

// complex types
//
// inspired by http://www.mail-archive.com/hdf-forum@hdfgroup.org/msg00759.html

template <typename T>
class ComplexH5Type : public H5::CompType
{
public:
    ComplexH5Type (void)
        : CompType(sizeof(std::complex<T>))
        {
            const H5::DataType * const datatype = DatatypeSpecialization<T>::get();
            assert(datatype->getSize() == sizeof(T));
            // If we call the members "r" and "i", h5py interprets the
            // structure correctly as complex numbers.
            this->insertMember(std::string("r"), 0, *datatype);
            this->insertMember(std::string("i"), sizeof(T), *datatype);
            this->pack();
        }

    static const ComplexH5Type<T> * get_singleton (void)
        {
            // NOTE: constructing this could be a race condition
            static ComplexH5Type<T> singleton;
            return &singleton;
        }
};

template <typename T>
struct DatatypeSpecialization<std::complex<T> >
{
    static inline const H5::DataType * get (void)
        {
            return ComplexH5Type<T>::get_singleton();
        }
};

// see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html

template <typename Derived>
void save (H5::CommonFG &h5group, const std::string &name, const Eigen::EigenBase<Derived> &mat, const H5::DSetCreatPropList &plist=H5::DSetCreatPropList::DEFAULT)
{
    typedef typename Derived::Scalar Scalar;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);
    const std::array<hsize_t, 2> dimensions = { {
        static_cast<hsize_t>(mat.rows()),
        static_cast<hsize_t>(mat.cols())
    } };
    const H5::DataSpace dataspace(dimensions.size(), dimensions.data());
    const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
    H5::DataSet dataset = h5group.createDataSet(name, *datatype, dataspace, plist);
    dataset.write(row_major_mat.data(), *datatype);
}

template <typename Derived>
void load (const H5::CommonFG &h5group, const std::string &name, const Eigen::DenseBase<Derived> &mat)
{
    typedef typename Derived::Scalar Scalar;
    const H5::DataSet dataset = h5group.openDataSet(name);
    const H5::DataSpace dataspace = dataset.getSpace();
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
    const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
    dataset.read(data.data(), *datatype, dataspace);
    // see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
    Eigen::DenseBase<Derived> &mat_ = const_cast<Eigen::DenseBase<Derived> &>(mat);
    mat_.derived().resize(rows, cols);
    mat_ = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(data.data(), rows, cols);
}

} // namespace EigenHDF5

#endif
