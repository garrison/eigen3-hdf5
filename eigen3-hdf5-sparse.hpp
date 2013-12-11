#ifndef _EIGEN3_HDF5_SPARSE_HPP
#define _EIGEN3_HDF5_SPARSE_HPP

#include <vector>

#include <eigen3-hdf5.hpp>
#include <Eigen/Sparse>

namespace EigenHDF5
{

template <typename Scalar>
class MyTriplet : public Eigen::Triplet<Scalar>
{
public:
    MyTriplet (void)
        : Eigen::Triplet<Scalar>()
        {
        }

    MyTriplet (const unsigned int &i, const unsigned int &j, const Scalar &v = Scalar(0))
        : Eigen::Triplet<Scalar>(i, j, v)
        {
        }

    static std::size_t offsetof_row (void) { return offsetof(MyTriplet<Scalar>, m_row); }
    static std::size_t offsetof_col (void) { return offsetof(MyTriplet<Scalar>, m_col); }
    static std::size_t offsetof_value (void) { return offsetof(MyTriplet<Scalar>, m_value); }
};

template <typename T>
class SparseH5Type : public H5::CompType
{
public:
    SparseH5Type (void)
        : CompType(sizeof(MyTriplet<T>))
        {
            const H5::DataType * const datatypei = DatatypeSpecialization<unsigned int>::get();
            const H5::DataType * const datatype = DatatypeSpecialization<T>::get();
            assert(datatype->getSize() == sizeof(T));
            this->insertMember(std::string("r"), MyTriplet<T>::offsetof_row(), *datatypei);
            this->insertMember(std::string("c"), MyTriplet<T>::offsetof_col(), *datatypei);
            this->insertMember(std::string("v"), MyTriplet<T>::offsetof_value(), *datatype);
            this->pack();
        }

    static const SparseH5Type<T> * get_singleton (void)
        {
            // NOTE: constructing this could be a race condition
            static SparseH5Type<T> singleton;
            return &singleton;
        }
};

template <typename SparseMatrixType>
void save_sparse (H5::CommonFG &h5group, const std::string &name, const SparseMatrixType &mat, const H5::DSetCreatPropList &plist=H5::DSetCreatPropList::DEFAULT)
{
    typedef typename SparseMatrixType::Scalar Scalar;
    // save the actual sparse matrix
    std::vector<MyTriplet<Scalar> > data;
    data.reserve(mat.nonZeros());
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (typename SparseMatrixType::InnerIterator it(mat, k); it; ++it) {
            if (it.value() != Scalar(0))
                data.push_back(MyTriplet<Scalar>(it.row(), it.col(), it.value()));
        }
    }
    const hsize_t nnz = data.size();
    const H5::DataSpace dataspace(1, &nnz);
    const H5::DataType * const datatype = SparseH5Type<Scalar>::get_singleton();
    H5::DataSet dataset = h5group.createDataSet(name, *datatype, dataspace, plist);
    dataset.write(data.data(), *datatype);
    // save the matrix's shape as an attribute
    Eigen::Matrix<typename SparseMatrixType::Index, 2, 1> shape;
    shape(0) = mat.rows();
    shape(1) = mat.cols();
    save_attribute(dataset, "shape", shape);
}

template <typename SparseMatrixType>
void load_sparse (const H5::CommonFG &h5group, const std::string &name, SparseMatrixType &mat)
{
    typedef typename SparseMatrixType::Scalar Scalar;
    const H5::DataSet dataset = h5group.openDataSet(name);
    const H5::DataSpace dataspace = dataset.getSpace();
    const std::size_t ndims = dataspace.getSimpleExtentNdims();
    if (ndims != 1) {
        throw std::runtime_error("HDF5 array has incorrect number of dimensions to represent a sparse matrix.");
    }
    Eigen::Matrix<typename SparseMatrixType::Index, 2, 1> shape;
    load_attribute(dataset, "shape", shape);
    hsize_t nnz;
    dataspace.getSimpleExtentDims(&nnz); // assumes ndims == 1 in the data representation
    const H5::DataType * const datatype = SparseH5Type<Scalar>::get_singleton();
    std::vector<MyTriplet<Scalar> > data(nnz);
    dataset.read(data.data(), *datatype, dataspace);
    mat.resize(shape(0), shape(1)); // NOTE: this also clears all existing values
    mat.setFromTriplets(data.begin(), data.end());
}

} // namespace EigenHDF5

#endif
