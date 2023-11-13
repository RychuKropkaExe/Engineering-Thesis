load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "test",
    srcs = ["runTests.cpp"],
    deps = ["//test/matrixTest:MatrixTest", "//test/modelTest:ModelTest", "//src/model:Model"],
    visibility = ["//visibility:public"],
)