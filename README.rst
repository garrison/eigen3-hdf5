eigen3-hdf5
===========

Easy serialization of C++ `Eigen <http://eigen.tuxfamily.org/>`_
matrices using `HDF5 <http://www.hdfgroup.org/HDF5/>`_.

The library is meant to be bare-bones.  It gets me 90% of what I
want/need in < 200 lines of code.

Requirements
------------

* Eigen3 (tested on 3.1 and 3.2 branches)
* HDF5 C++ wrapper library
* C++11

Because ``eigen3-hdf5`` is a template library, there is nothing to link
against (besides the HDF5 libraries).

Most of the code is compatible with C++98, but a few C++11 features
(e.g. ``std::array``) are used for additional safety.  It would not be
difficult to drop this requirement in the future.

API
---

Supports saving and restoring Eigen matrices and vectors of ``float``,
``double``, ``long double``, ``int``, ``unsigned int``, and
``std::complex<>``.

.. code:: c++

    #include <eigen3-hdf5.hpp>

    void save_matrix()
    {
        Eigen::Matrix3d mat;
        mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        H5::H5File file("filename1.h5", H5F_ACC_TRUNC);
        EigenHDF5::save(file, "MatrixDataSetName", mat);
    }

    void load_vector()
    {
        Eigen::Vector4i vec;
        H5::H5File file("filename2.h5", H5F_ACC_RDONLY);
        EigenHDF5::load(file, "VectorDataSetName", vec);
    }

See the `unittests <unittests/>`_ directory for more examples.

Unit tests
----------

I am using `premake4 <http://industriousone.com/premake>`_ and
`googletest <https://code.google.com/p/googletest/>`_ because I am
familiar with them.

The unit tests currently write to specific files in /tmp.  This should
change, eventually.

License
-------

MIT license.

Next steps
----------

* Support more fundamental data types

Thoughts/notes
--------------

* Using the HDF5 C++ wrapper library supposedly means it `won't work
  with parallel hdf5
  <http://www.hdfgroup.org/hdf5-quest.html#p5thread>`_.  Is this
  likely to matter?
* Will the HDF5 library always handle endian issues transparently for us?
