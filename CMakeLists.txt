cmake_minimum_required(VERSION 3.28)

project(spacrow LANGUAGES C CXX)

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
# Dependencies (fetched via FetchContent)
# ---------------------------------------------------------------------------
include(FetchContent)

# Pinned versions for reproducibility.
set(SPACROW_XTL_VERSION    "0.8.0")
set(SPACROW_XSIMD_VERSION  "13.2.0")
set(SPACROW_XTENSOR_VERSION "0.27.1")
set(SPACROW_SPARROW_GIT_TAG "main")
set(SPACROW_GTEST_VERSION  "v1.15.2")

# xtl (required by xtensor) -------------------------------------------------
FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        ${SPACROW_XTL_VERSION}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

# xsimd (optional SIMD backend for xtensor) --------------------------------
FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        ${SPACROW_XSIMD_VERSION}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

# xtensor (header-only) ----------------------------------------------------
FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        ${SPACROW_XTENSOR_VERSION}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

# sparrow (Apache Arrow columnar format in C++20) -------------------------
FetchContent_Declare(
    sparrow
    GIT_REPOSITORY https://github.com/sparrow-org/sparrow.git
    GIT_TAG        ${SPACROW_SPARROW_GIT_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

# GoogleTest ----------------------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        ${SPACROW_GTEST_VERSION}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

# CodSpeed Google Benchmark fork (benchmarks only) -------------------------
set(SPACROW_CODSPEED_CPP_VERSION "v2.4.0")
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp.git
    GIT_TAG        ${SPACROW_CODSPEED_CPP_VERSION}
    SOURCE_SUBDIR  google_benchmark
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
)

# Make xtl/xsimd available first so xtensor's find_package(xtl) resolves
# to the in-tree targets.
FetchContent_MakeAvailable(xtl xsimd)

# xtensor is header-only; building it just exports the target.
set(XTENSOR_ENABLE_ASSERT OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xtensor)

# Disable sparrow extras we don't need (must be set before MakeAvailable).
set(BUILD_TESTS              OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS        OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS               OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES           OFF CACHE BOOL "" FORCE)
set(USE_DATE_POLYFILL        OFF CACHE BOOL "" FORCE)
set(CREATE_JSON_READER_TARGET OFF CACHE BOOL "" FORCE)
set(ENABLE_INTEGRATION_TEST  OFF CACHE BOOL "" FORCE)
set(SPARROW_BUILD_SHARED     OFF CACHE BOOL "" FORCE)
set(SPARROW_CONTRACTS_THROW_ON_FAILURE ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sparrow)

# GoogleTest (tests only)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark (benchmarks only)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(CODSPEED_MODE "${SPACROW_EFFECTIVE_CODSPEED_MODE}" CACHE STRING "" FORCE)
FetchContent_MakeAvailable(benchmark)

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
