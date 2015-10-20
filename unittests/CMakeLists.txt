project(eigen3-hdf5-tests)

include_directories(..)

set(${PROJECT_NAME}_SOURCES
    test_Attribute.cpp
    test_MatrixRoundTrip.cpp
    test_Sparse.cpp
    test_VectorRoundTrip.cpp
    )

if (WIN32)
    add_definitions(/W4 /wd4251)
else()
endif()

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME} hdf5 hdf5_cpp GTest GTestMain)