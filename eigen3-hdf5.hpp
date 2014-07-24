#ifndef _EIGEN3_HDF5_HPP
#define _EIGEN3_HDF5_HPP

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
struct DatatypeSpecialization<short>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_SHORT;
        }
};

template <>
struct DatatypeSpecialization<unsigned short>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_USHORT;
        }
};

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

template <>
struct DatatypeSpecialization<long>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_LONG;
        }
};

template <>
struct DatatypeSpecialization<unsigned long>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_ULONG;
        }
};

template <>
struct DatatypeSpecialization<long long>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_LLONG;
        }
};

template <>
struct DatatypeSpecialization<unsigned long long>
{
    static inline const H5::DataType * get (void)
        {
            return &H5::PredType::NATIVE_ULLONG;
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

// string types, to be used mainly for attributes

template <>
struct DatatypeSpecialization<const char *>
{
    static inline const H5::DataType * get (void)
        {
            static const H5::StrType strtype(0, H5T_VARIABLE);
            return &strtype;
        }
};

template <>
struct DatatypeSpecialization<char *>
{
    static inline const H5::DataType * get (void)
        {
            static const H5::StrType strtype(0, H5T_VARIABLE);
            return &strtype;
        }
};

// XXX: for some unknown reason the following two functions segfault if
// H5T_VARIABLE is used.  The passed strings should still be null-terminated,
// so this is a bit worrisome.

template <std::size_t N>
struct DatatypeSpecialization<const char [N]>
{
    static inline const H5::DataType * get (void)
        {
            static const H5::StrType strtype(0, N);
            return &strtype;
        }
};

template <std::size_t N>
struct DatatypeSpecialization<char [N]>
{
    static inline const H5::DataType * get (void)
        {
            static const H5::StrType strtype(0, N);
            return &strtype;
        }
};

namespace internal
{
    template <typename Derived>
    H5::DataSpace create_dataspace (const Eigen::EigenBase<Derived> &mat)
    {
        const std::size_t dimensions_size = 2;
        const hsize_t dimensions[dimensions_size] = {
            static_cast<hsize_t>(mat.rows()),
            static_cast<hsize_t>(mat.cols())
        };
        return H5::DataSpace(dimensions_size, dimensions);
    }
}

template <typename T>
void save_scalar_attribute (const H5::H5Location &h5obj, const std::string &name, const T &value)
{
    const H5::DataType * const datatype = DatatypeSpecialization<T>::get();
    H5::DataSpace dataspace(H5S_SCALAR);
    H5::Attribute att = h5obj.createAttribute(name, *datatype, dataspace);
    att.write(*datatype, &value);
}

template <>
inline void save_scalar_attribute (const H5::H5Location &h5obj, const std::string &name, const std::string &value)
{
    save_scalar_attribute(h5obj, name, value.c_str());
}

// see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html

template <typename Derived>
void save (H5::CommonFG &h5group, const std::string &name, const Eigen::EigenBase<Derived> &mat, const H5::DSetCreatPropList &plist=H5::DSetCreatPropList::DEFAULT)
{
    typedef typename Derived::Scalar Scalar;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);
    const H5::DataSpace dataspace = internal::create_dataspace(mat);
    const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
    H5::DataSet dataset = h5group.createDataSet(name, *datatype, dataspace, plist);
    dataset.write(row_major_mat.data(), *datatype);
}

template <typename Derived>
void save_attribute (const H5::H5Location &h5obj, const std::string &name, const Eigen::EigenBase<Derived> &mat)
{
    typedef typename Derived::Scalar Scalar;
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);
    const H5::DataSpace dataspace = internal::create_dataspace(mat);
    const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
    H5::Attribute dataset = h5obj.createAttribute(name, *datatype, dataspace);
    dataset.write(*datatype, row_major_mat.data());
}

namespace internal
{
    // H5::Attribute and H5::DataSet both have similar API's, and although they
    // share a common base class, the relevant methods are not virtual.  Worst
    // of all, they take their arguments in different orders!

    template <typename Scalar>
    inline void read_data (const H5::DataSet &dataset, Scalar *data, const H5::DataType &datatype)
    {
        dataset.read(data, datatype);
    }

    template <typename Scalar>
    inline void read_data (const H5::Attribute &dataset, Scalar *data, const H5::DataType &datatype)
    {
        dataset.read(datatype, data);
    }

    template <typename Derived, typename DataSet>
    void _load (const DataSet &dataset, const Eigen::DenseBase<Derived> &mat)
    {
        typedef typename Derived::Scalar Scalar;
        const H5::DataSpace dataspace = dataset.getSpace();
        const std::size_t ndims = dataspace.getSimpleExtentNdims();
        assert(ndims > 0);
        const std::size_t dimensions_size = 2;
        hsize_t dimensions[dimensions_size];
        dimensions[1] = 1; // in case it's 1D
        if (ndims > dimensions_size) {
            throw std::runtime_error("HDF5 array has too many dimensions.");
        }
        dataspace.getSimpleExtentDims(dimensions);
        const hsize_t rows = dimensions[0], cols = dimensions[1];
        std::vector<Scalar> data(rows * cols);
        const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
        internal::read_data(dataset, data.data(), *datatype);
        // see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
        Eigen::DenseBase<Derived> &mat_ = const_cast<Eigen::DenseBase<Derived> &>(mat);
        mat_.derived().resize(rows, cols);
        mat_ = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(data.data(), rows, cols);
    }

}

template <typename Derived>
void load (const H5::CommonFG &h5group, const std::string &name, const Eigen::DenseBase<Derived> &mat)
{
    const H5::DataSet dataset = h5group.openDataSet(name);
    internal::_load(dataset, mat);
}

template <typename Derived>
void load_attribute (const H5::H5Location &h5obj, const std::string &name, const Eigen::DenseBase<Derived> &mat)
{
    const H5::Attribute dataset = h5obj.openAttribute(name);
    internal::_load(dataset, mat);
}

} // namespace EigenHDF5

#endif
