cmake_minimum_required(VERSION 3.28)

project(sparrow-compute LANGUAGES C CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

message(STATUS "🔧 sparrow-compute: C++20 project with sparrow + xtensor compute kernels")

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(
    SPARROW_COMPUTE_ENABLE_CODSPEED
    "Enable CodSpeed integration and allow CODSPEED_ENABLED to be defined"
    OFF
)

set(
    SPARROW_COMPUTE_CODSPEED_MODE
    "off"
    CACHE STRING
    "CodSpeed benchmark mode: off, instrumentation, simulation, memory, or walltime"
)
set_property(
    CACHE SPARROW_COMPUTE_CODSPEED_MODE
    PROPERTY STRINGS off instrumentation simulation memory walltime
)

if(SPARROW_COMPUTE_ENABLE_CODSPEED)
    set(SPARROW_COMPUTE_EFFECTIVE_CODSPEED_MODE "${SPARROW_COMPUTE_CODSPEED_MODE}")
else()
    set(SPARROW_COMPUTE_EFFECTIVE_CODSPEED_MODE "off")
    if(NOT SPARROW_COMPUTE_CODSPEED_MODE STREQUAL "off")
        message(
            WARNING
            "SPARROW_COMPUTE_ENABLE_CODSPEED=OFF forces CodSpeed off; ignoring "
            "SPARROW_COMPUTE_CODSPEED_MODE=${SPARROW_COMPUTE_CODSPEED_MODE}."
        )
    endif()
endif()

message(STATUS "🔧 CodSpeed enabled: ${SPARROW_COMPUTE_ENABLE_CODSPEED}")
message(STATUS "🔧 CodSpeed mode: ${SPARROW_COMPUTE_CODSPEED_MODE}")

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build sparrow-compute test suite" ON)
message(STATUS "🔧 Build tests: ${BUILD_TESTS}")
option(BUILD_BENCHMARKS "Build sparrow-compute benchmark suite" OFF)
message(STATUS "🔧 Build benchmarks: ${BUILD_BENCHMARKS}")

# ---------------------------------------------------------------------------
# Dependencies — see cmake/external_dependencies.cmake
# ---------------------------------------------------------------------------
include(external_dependencies)

# ---------------------------------------------------------------------------
# sparrow-compute library (header-only for now)
# ---------------------------------------------------------------------------
add_library(sparrow-compute INTERFACE)
target_include_directories(sparrow-compute INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# Enable xtensor SIMD acceleration for consumers of sparrow-compute.
target_compile_definitions(sparrow-compute INTERFACE XTENSOR_USE_XSIMD)
target_link_libraries(sparrow-compute INTERFACE xtensor xsimd sparrow::sparrow)

# ---------------------------------------------------------------------------
# Demo executable
# ---------------------------------------------------------------------------
add_executable(sparrow-compute_demo examples/demo_kernels.cpp)
target_link_libraries(sparrow-compute_demo PRIVATE sparrow-compute)
set_target_properties(sparrow-compute_demo PROPERTIES
    CXX_EXTENSIONS OFF
    FOLDER "sparrow-compute"
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if (BUILD_TESTS)
    enable_testing()

    add_executable(sparrow-compute_tests tests/test_kernels.cpp)
    target_link_libraries(sparrow-compute_tests PRIVATE
        sparrow-compute
        GTest::gtest_main
    )
    set_target_properties(sparrow-compute_tests PROPERTIES
        CXX_EXTENSIONS OFF
        FOLDER "sparrow-compute"
    )

    include(GoogleTest)
    gtest_discover_tests(sparrow-compute_tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if (BUILD_BENCHMARKS)
    add_executable(sparrow-compute_bench benchmarks/bench_kernels.cpp)
    target_link_libraries(sparrow-compute_bench PRIVATE
        sparrow-compute
        benchmark::benchmark
    )
    set_target_properties(sparrow-compute_bench PROPERTIES
        CXX_EXTENSIONS OFF
        FOLDER "sparrow-compute"
    )
endif()
