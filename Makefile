# Simple CMake wrapper.
CMAKE ?= cmake
BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc 2>/dev/null || echo 1)

CMAKE_ARGS ?= -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

.PHONY: all help configure build test bench clean

all: build

help:
	@echo "make build      Configure and build (default)"
	@echo "make configure  Run CMake only"
	@echo "make test       Build and run ctest"
	@echo "make bench      Build and run benchmarks"
	@echo "make clean      Remove $(BUILD_DIR)/"
	@echo ""
	@echo "Variables: BUILD_DIR=$(BUILD_DIR) BUILD_TYPE=$(BUILD_TYPE) JOBS=$(JOBS)"
	@echo "           CMAKE_ARGS=\"...\" for extra CMake flags"

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_ARGS)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure --parallel $(JOBS)

bench: build
	$(BUILD_DIR)/fused_ml_benchmarks

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
