project(eigen3-hdf5-tests)

include_directories(..)

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

target_link_libraries(${PROJECT_NAME} GTest GTestMain)
target_link_libraries(${PROJECT_NAME} ${HDF5_EXPORT_LIBRARIES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
