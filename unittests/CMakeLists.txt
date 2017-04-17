project(eigen3-hdf5-tests)

include_directories(..)
include_directories(${HDF5_CXX_INCLUDE_DIR})
include_directories(${gtest_INCLUDE_DIR})

set(${PROJECT_NAME}_SOURCES
    test_Attribute.cpp
    test_MatrixRoundTrip.cpp
    test_Sparse.cpp
    test_VectorRoundTrip.cpp
    gtest-helpers.hpp
    )

if (WIN32)
    add_definitions(/W4 /wd4251)
else()
    add_definitions(-Wall -Wextra)
endif()

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES})

if (DEFINED gtest_LIBRARIES)
  target_link_libraries(${PROJECT_NAME} ${gtest_LIBRARIES})
else()
  target_link_libraries(${PROJECT_NAME} GTest GTestMain)
endif()

target_link_libraries(${PROJECT_NAME} ${HDF5_EXPORT_LIBRARIES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
