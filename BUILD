licenses(["notice"])

exports_files(["LICENSE.txt"])

package(default_visibility=["//visibility:public"])

COPTS = [
        "-std=c++17",
        "-msse3",
        "-msse4",
        "-O3",
        "-I.",             
]

LINK_OPTS = [
        "-lpthread",
        "-lm",   
]


# util include library
cc_library(
    name = "source_lib",
    srcs = [ ],
    hdrs = [
        "eigen3-hdf5.hpp",
        "eigen3-hdf5-sparse.hpp",
    ],
    deps = [
    ],
    includes = [ ], 
    copts    = COPTS,
    linkopts = LINK_OPTS,
)