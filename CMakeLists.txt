cmake_minimum_required(VERSION 3.28)

project(spacrow LANGUAGES C CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

message(STATUS "🔧 spacrow: C++20 project with sparrow + xtensor compute kernels")

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
    SPACROW_ENABLE_CODSPEED
    "Enable CodSpeed integration and allow CODSPEED_ENABLED to be defined"
    ON
)

set(
    SPACROW_CODSPEED_MODE
    "off"
    CACHE STRING
    "CodSpeed benchmark mode: off, instrumentation, simulation, memory, or walltime"
)
set_property(
    CACHE SPACROW_CODSPEED_MODE
    PROPERTY STRINGS off instrumentation simulation memory walltime
)

if(SPACROW_ENABLE_CODSPEED)
    set(SPACROW_EFFECTIVE_CODSPEED_MODE "${SPACROW_CODSPEED_MODE}")
else()
    set(SPACROW_EFFECTIVE_CODSPEED_MODE "off")
    if(NOT SPACROW_CODSPEED_MODE STREQUAL "off")
        message(
            WARNING
            "SPACROW_ENABLE_CODSPEED=OFF forces CodSpeed off; ignoring "
            "SPACROW_CODSPEED_MODE=${SPACROW_CODSPEED_MODE}."
        )
    endif()
endif()

message(STATUS "🔧 CodSpeed enabled: ${SPACROW_ENABLE_CODSPEED}")
message(STATUS "🔧 CodSpeed mode: ${SPACROW_CODSPEED_MODE}")

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build spacrow test suite" ON)
message(STATUS "🔧 Build tests: ${BUILD_TESTS}")
option(BUILD_BENCHMARKS "Build spacrow benchmark suite" ON)
message(STATUS "🔧 Build benchmarks: ${BUILD_BENCHMARKS}")

# ---------------------------------------------------------------------------
# Dependencies — see cmake/external_dependencies.cmake
# ---------------------------------------------------------------------------
include(external_dependencies)

# ---------------------------------------------------------------------------
# spacrow library (header-only for now)
# ---------------------------------------------------------------------------
add_library(spacrow INTERFACE)
target_include_directories(spacrow INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(spacrow INTERFACE sparrow)
# Enable xtensor SIMD acceleration for consumers of spacrow.
target_compile_definitions(spacrow INTERFACE XTENSOR_USE_XSIMD)
target_link_libraries(spacrow INTERFACE xtensor xsimd)

# ---------------------------------------------------------------------------
# Demo executable
# ---------------------------------------------------------------------------
add_executable(spacrow_demo examples/demo_kernels.cpp)
target_link_libraries(spacrow_demo PRIVATE spacrow)
set_target_properties(spacrow_demo PROPERTIES
    CXX_EXTENSIONS OFF
    FOLDER "spacrow"
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(spacrow_tests tests/test_kernels.cpp)
target_link_libraries(spacrow_tests PRIVATE
    spacrow
    GTest::gtest_main
)
set_target_properties(spacrow_tests PROPERTIES
    CXX_EXTENSIONS OFF
    FOLDER "spacrow"
)

include(GoogleTest)
gtest_discover_tests(spacrow_tests)

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
add_executable(spacrow_bench benchmarks/bench_kernels.cpp)
target_link_libraries(spacrow_bench PRIVATE
    spacrow
    benchmark::benchmark
)
set_target_properties(spacrow_bench PROPERTIES
    CXX_EXTENSIONS OFF
    FOLDER "spacrow"
)
