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

    // I suspect there is a more optimal way to set up the hyperslab that would avoid
    // copying row by row, but I do not know what it is right now. 
    template <typename Derived>
    bool write_mat_by_rows(const Eigen::EigenBase<Derived> &mat, 
        const H5::DataType * const datatype,
        H5::DataSet *dataset,
        const H5::DataSpace* dataspace)
    {
        Derived::Index mstride = mat.derived().outerStride();

        // slab params for the file data
        hsize_t fstride[2] = { 1, mat.cols() };
        hsize_t count[2] = { 1, 1 };
        hsize_t block[2] = { 1, mat.cols() };

        // slab params for the memory data
        hsize_t mdim[2] = { 1, mat.cols() };
        H5::DataSpace mspace(2, mdim);

        // write each row of mat as a slab
        for (int i = 0; i < mat.rows(); i++)
        {
            hsize_t start[2] = { i, 0 };
            dataspace->selectHyperslab(H5S_SELECT_SET, count, start, fstride, block);
            dataset->write(mat.derived().data() + i*mstride, *datatype, mspace, *dataspace);
        }
        return true;
    }

    template <typename Derived>
    bool write_col_mat(const Eigen::EigenBase<Derived> &mat,
        const H5::DataType * const datatype,
        H5::DataSet *dataset,
        const H5::DataSpace* dspace)
    {
        bool written = false;
        Derived::Index rows = mat.rows();
        Derived::Index cols = mat.cols();

        Derived::Index stride = mat.derived().outerStride();
        if (rows == stride)
        {
            // slab params for the file data
            hsize_t fstride[2] = { 1, 1 };
            hsize_t fcount[2] = { 1, 1 };
            hsize_t fblock[2] = { 1, cols };

            // slab params for the memory data
            hsize_t mdim[2] = { cols, rows };
            H5::DataSpace mspace(2, mdim);
            hsize_t mstride[2] = { rows, 1 };
            hsize_t mcount[2] = { 1, 1 };
            hsize_t mblock[2] = { cols, 1 };

            // write each row of mat as a slab
            for (int i = 0; i < rows; i++)
            {
                hsize_t fstart[2] = { i, 0 };
                hsize_t mstart[2] = { 0, i };
                dspace->selectHyperslab(H5S_SELECT_SET, fcount, fstart, fstride, fblock);
                mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);
                dataset->write(mat.derived().data(), *datatype, mspace, *dspace);
            }
            written = true;
        }
        return written;
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
    const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
    const H5::DataSpace dataspace = internal::create_dataspace(mat);
    H5::DataSet dataset = h5group.createDataSet(name, *datatype, dataspace, plist);

    bool written = false;  // flag will be true when the data has been written
    if (mat.derived().Flags & Eigen::RowMajor)
    {
        // the matrix is stored in a row major order; see if it is contiguous
        if (mat.derived().outerStride() == mat.cols())
        {
            // the stride between rows is the number of columns means there is no padding between
            // rows, so the data is contiguous and we can write the data directly from memory
            // without copying. 
            dataset.write(mat.derived().data(), *datatype);
            written = true;
        }
        else if (mat.derived().innerStride() == 1)
        {
            // inner stride == 1 is a sanity check. Only matrices made with unconventional maps
            // would have inner stride != 1, and they will be handled below by first copying the
            // input matrix. 

            // write the matrix by rows to the dataset using the dataspace made above
            written = internal::write_mat_by_rows(mat, datatype, &dataset, &dataspace);
        }
    }
    else
    {
        written = internal::write_col_mat(mat, datatype, &dataset, &dataspace);
    }
    
    if (!written)
    {
        // data has not yet been written, so there is nothing else to try but copy the input
        // matrix to a row major matrix and write it. 
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);
        dataset.write(row_major_mat.data(), *datatype);
    }
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
        const H5::DataType * const datatype = DatatypeSpecialization<Scalar>::get();
        Eigen::DenseBase<Derived> &mat_ = const_cast<Eigen::DenseBase<Derived> &>(mat);
        bool written = false;
        if (mat.Flags & Eigen::RowMajor || dimensions[0] == 1 || dimensions[1] == 1)
        {
            // mat is already row major; resize it and read into it
            mat_.derived().resize(rows, cols);
            Derived::Index stride = mat_.derived().outerStride();
            if (stride == cols || (stride == rows && cols == 1))
            {
                // mat has natural stride, so read directly into its data block
                internal::read_data(dataset, mat_.derived().data(), *datatype);
                written = true;
            }
        }

        if (!written)
        {
            // input is col major or has unnatural stride; read into a temp and copy it
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp(rows, cols);
            internal::read_data(dataset, temp.data(), *datatype);
            mat_ = temp;
            written = true;
        }
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
