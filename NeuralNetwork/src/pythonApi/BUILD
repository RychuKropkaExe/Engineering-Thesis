load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "pythonApi",
    srcs = ["pythonApi.cpp"],
    deps = ["//src/fastMatrix:FastMatrix", "//src/model:Model"],
)