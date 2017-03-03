cc_library(
    name = "main",
    srcs = glob(
        ["src/*.cc",],
        exclude = ["src/gtest-all.cc",],
    ),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h",
    ]),
    copts = [
        "-Iexternal/gtest/include",
        "-std=c++17",
    ],
    linkopts = [
        "-pthread", 
        "-lm",
        "-O3",
    ],
    visibility = ["//visibility:public"],
)