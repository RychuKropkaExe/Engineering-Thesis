.PHONY: build_test
.PHONY: build_main
build_test:
	bazel build test --verbose_failures; mv "bazel-out/x64_windows-fastbuild/bin/test/test.exe" "test.exe"

build_main:
	bazel build main --verbose_failures
