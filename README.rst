eigen3-hdf5
===========

Easy serialization of C++ `Eigen <http://eigen.tuxfamily.org/>`_
matrices using `HDF5 <http://www.hdfgroup.org/HDF5/>`_.

The library is meant to be bare-bones (at least for now).  It gets me
90% of what I want/need in a few hundred lines of code.  It may also
get you 90% (or even 100%) of what you need.

Requirements
------------

* Eigen3 (tested on 3.1 and 3.2 branches)
* HDF5 C++ wrapper library >= 1.8.12 (yes, this is a very recent
  version)

Because ``eigen3-hdf5`` is a template library, there is nothing to link
against (besides the HDF5 libraries).

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
  <http://www.hdfgroup.org/hdf5-quest.html#p5thread>`_.  If I were to
  do it again, I would write ``eigen3-hdf5`` using the regular C HDF5
  API, not the C++ wrapper.  Patches are welcome. :)
