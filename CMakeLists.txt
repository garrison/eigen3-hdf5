# Check the cmake version and name the project
cmake_minimum_required(VERSION 3.2.0)

# This is not a conventional top-level cmake file. Rather it is a
# placeholder that has simple constructs for the settings needed to to
# build against external hdf5, eigen, and gtest libraries.  This is
# very minimal and does not have elegant cmake find logic. I use
# find_package Eigen and HDF5 as this seems to work for my windows and
# ubuntu configurations. But gtest is more difficult.
# 
# The recommended gtest convention is to include the gtest fused
# source along with the source of the project being tested. I do not
# want to make such a major addition to eigen3-hdf5, so this is the
# best I could come up with for now.  I require the cmake configure
# invocation to set locations explicitly for gtest include and
# libraries. Here is an example invocation that I used to test on
# linux
#
# cmake -DCMAKE_PREFIX_PATH=/opt/pkg/eigen \
#       -Dgtest_INCLUDE_DIR=/opt/pkg/gtest/include_fused \
#       -Dgtest_LIBRARIES=/opt/pkg/gtest/build/libgtest_main.a;/opt/pkg/gtest/build/libgtest.a
#       ..
#
# In addition to having the required find_package logic, conventional
# top-level cmake files will also have install logic and should
# generate some documentation. Neither of these are part of
# eigen3-hdf5, so they are not here yet.

set(CMAKE_DEBUG_POSTFIX d)

project(eigen3-hdf5)

# Eigen
find_package(Eigen REQUIRED)
message (STATUS "eigen_INCLUDE_DIRS=${eigen_INCLUDE_DIRS}")
include_directories(${eigen_INCLUDE_DIRS})

# gtest use -Dgtest_INCLUDE_DIR=... -Dgtest_LIBRARIES=... when
# configuring cmake

# HDF5

# look for an hdf5 cmake config file
find_package(HDF5 NO_MODULE QUIET COMPONENTS CXX HL)

if (NOT HDF5_FOUND)
  # config file not found, so try to find using a module
  find_package(HDF5 REQUIRED MODULE COMPONENTS CXX HL)
  if ("${HDF5_EXPORT_LIBRARIES}" STREQUAL "")
    # set this config variable using the module variable to simplify
    # the target_link_libraries statement
    set(HDF5_EXPORT_LIBRARIES ${HDF5_LIBRARIES})
  endif()
endif()

enable_testing()

add_subdirectory(unittests)
