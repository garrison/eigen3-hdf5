licenses(["notice"])

exports_files(["LICENSE.txt"])

package(default_visibility=["//visibility:public"])

COPTS = [
        "-std=c++17",
        "-msse3",
        "-msse4",
        "-O3",

        "-Iexternal/gtest/include",
        "-I.",             
]

LINK_OPTS = [
        "-lpthread",
        "-lm",   
]

cc_test(
    name = "tests",
    srcs = [
        "test_Sparse.cpp",
    ],
    includes = [ ],
    copts    = COPTS,
    linkopts = LINK_OPTS,
    deps = [
        "//:source_lib",
        "@gtest//:main",
        "@eigen//:main",
        "@hdf5//:main",
    ],
)
