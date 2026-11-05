cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(fused_ml VERSION 0.1.0 LANGUAGES CXX CUDA)

# Options
option(FUSED_ML_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(FUSED_ML_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(FUSED_ML_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
option(FUSED_ML_FETCH_DEPS "Fetch missing dependencies" ON)

# Language standards / global settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED CMAKE_CUDA_SEPARABLE_COMPILATION)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# PyTorch
# Let CMake find the Torch CMake config shipped with the active Python env.
if(NOT DEFINED Torch_DIR)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                    "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE _torch_cmake_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _torch_query_result
            ERROR_QUIET
        )
        if(_torch_query_result EQUAL 0 AND _torch_cmake_prefix)
            list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
        endif()
    endif()
endif()

find_package(Torch REQUIRED)

# Apply torch export flags globally so every C++ and CUDA translation unit uses the same ABI as libtorch.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")

# Dependency helpers
include(FetchContent)

# pybind11
if(FUSED_ML_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        if(NOT FUSED_ML_FETCH_DEPS)
            message(FATAL_ERROR "pybind11 not found and FUSED_ML_FETCH_DEPS=OFF")
        endif()
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v3.0.1
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# GoogleTest
if(FUSED_ML_BUILD_TESTS)
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        if(NOT FUSED_ML_FETCH_DEPS)
            message(FATAL_ERROR "GTest not found and FUSED_ML_FETCH_DEPS=OFF")
        endif()
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# Google Benchmark
if(FUSED_ML_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG QUIET)
    if(NOT benchmark_FOUND)
        if(NOT FUSED_ML_FETCH_DEPS)
            message(FATAL_ERROR "benchmark not found and FUSED_ML_FETCH_DEPS=OFF")
        endif()
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            google_benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.1
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(google_benchmark)
    endif()
endif()

# Core C++/CUDA library
set(FUSED_ML_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(FUSED_ML_INCLUDE_DIR "${FUSED_ML_SRC_DIR}/include")

set(FUSED_ML_CPP_SOURCES
    # "${FUSED_ML_SRC_DIR}/example.cpp"
)

set(FUSED_ML_CUDA_SOURCES
    # "${FUSED_ML_SRC_DIR}/cuda/example.cu"
)

set(FUSED_ML_CORE_SOURCES ${FUSED_ML_CPP_SOURCES} ${FUSED_ML_CUDA_SOURCES})

if(FUSED_ML_CORE_SOURCES)
    add_library(fused_ml_core STATIC ${FUSED_ML_CORE_SOURCES})

    target_include_directories(fused_ml_core
        PUBLIC "${FUSED_ML_INCLUDE_DIR}"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(fused_ml_core PRIVATE ${TORCH_LIBRARIES})
endif()

# Python extension (pybind11)
if(FUSED_ML_BUILD_PYTHON)
    pybind11_add_module(fused_ml_ext "${FUSED_ML_SRC_DIR}/pybind.cpp")

    set_target_properties(fused_ml_ext PROPERTIES
        OUTPUT_NAME "_C"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/fused_ml"
    )

    target_include_directories(fused_ml_ext PRIVATE
        "${FUSED_ML_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(fused_ml_ext PRIVATE
        pybind11::module
        ${TORCH_LIBRARIES}
    )

    if(TARGET fused_ml_core)
        target_link_libraries(fused_ml_ext PRIVATE fused_ml_core)
    endif()
endif()

# Tests
if(FUSED_ML_BUILD_TESTS)
    include(GoogleTest)

    add_executable(fused_ml_tests
        # tests/example_test.cpp
    )

    target_include_directories(fused_ml_tests PRIVATE
        "${FUSED_ML_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(TARGET GTest::gtest_main)
        set(FUSED_ML_GTEST_MAIN GTest::gtest_main)
    elseif(TARGET gtest_main)
        set(FUSED_ML_GTEST_MAIN gtest_main)
    else()
        message(FATAL_ERROR "No GoogleTest main target found")
    endif()

    target_link_libraries(fused_ml_tests PRIVATE
        ${FUSED_ML_GTEST_MAIN}
        ${TORCH_LIBRARIES}
    )

    if(TARGET fused_ml_core)
        target_link_libraries(fused_ml_tests PRIVATE fused_ml_core)
    endif()

    gtest_discover_tests(fused_ml_tests)
endif()

# Benchmarks
if(FUSED_ML_BUILD_BENCHMARKS)

    add_executable(fused_ml_benchmarks
        # benchmarks/example_benchmark.cpp
    )

    target_include_directories(fused_ml_benchmarks PRIVATE
        "${FUSED_ML_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(TARGET benchmark::benchmark_main)
        set(FUSED_ML_BENCHMARK_MAIN benchmark::benchmark_main)
    elseif(TARGET benchmark_main)
        set(FUSED_ML_BENCHMARK_MAIN benchmark_main)
    else()
        message(FATAL_ERROR "No Google Benchmark main target found")
    endif()

    target_link_libraries(fused_ml_benchmarks PRIVATE
        ${FUSED_ML_BENCHMARK_MAIN}
        ${TORCH_LIBRARIES}
    )

    if(TARGET fused_ml_core)
        target_link_libraries(fused_ml_benchmarks PRIVATE fused_ml_core)
    endif()
endif()
