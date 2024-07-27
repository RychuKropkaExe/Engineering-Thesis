# toolchain/cc_toolchain_config.bzl:
# NEW
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
# NEW
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",    # NEW
    "flag_group", # NEW
    "flag_set",   # NEW
    "tool_path",
)

all_link_actions = [ # NEW
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
    tool_paths = [ # NEW
        tool_path(
            name = "gcc",
            path = "C:/msys64/ucrt64/bin/g++.exe",
        ),
        tool_path(
            name = "ld",
            path = "C:/msys64/ucrt64/bin/ld.exe",
        ),
        tool_path(
            name = "ar",
            path = "C:/msys64/ucrt64/bin/ar.exe",
        ),
        tool_path(
            name = "cpp",
            path = "/bin/false",
        ),
        tool_path(
            name = "gcov",
            path = "/bin/false",
        ),
        tool_path(
            name = "nm",
            path = "/bin/false",
        ),
        tool_path(
            name = "objdump",
            path = "/bin/false",
        ),
        tool_path(
            name = "strip",
            path = "/bin/false",
        ),
    ]

    features = [ # NEW
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-lstdc++",
                                "-O3",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
    ctx = ctx,
    features = features, # NEW
    cxx_builtin_include_directories = [ # NEW
        "C:/msys64/ucrt64/include",
        "C:/msys64/ucrt64/include/c++/13.2.0",
        "C:/msys64/ucrt64/lib/gcc/x86_64-w64-mingw32/13.2.0/include/",
    ],
    toolchain_identifier = "local",
    host_system_name = "local",
    target_system_name = "local",
    target_cpu = "k8",
    target_libc = "unknown",
    compiler = "clang",
    abi_version = "unknown",
    abi_libc_version = "unknown",
    tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)