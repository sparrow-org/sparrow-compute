cmake_minimum_required(VERSION 3.28)

set(SPARROW_COMPUTE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(
    SPARROW_COMPUTE_VERSION_HEADER
    "${SPARROW_COMPUTE_INCLUDE_DIR}/sparrow-compute/config/sparrow_compute_version.hpp"
)

# Versioning
# ==========
file(
    STRINGS "${SPARROW_COMPUTE_VERSION_HEADER}" sparrow_compute_version_defines
    REGEX "constexpr int SPARROW_COMPUTE_VERSION_(MAJOR|MINOR|PATCH)"
)
foreach(ver ${sparrow_compute_version_defines})
    if(ver MATCHES "constexpr int SPARROW_COMPUTE_VERSION_(MAJOR|MINOR|PATCH) = ([0-9]+);$")
        set(SPARROW_COMPUTE_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    endif()
endforeach()

if(
    NOT DEFINED SPARROW_COMPUTE_VERSION_MAJOR
    OR NOT DEFINED SPARROW_COMPUTE_VERSION_MINOR
    OR NOT DEFINED SPARROW_COMPUTE_VERSION_PATCH
)
    message(FATAL_ERROR "Could not parse the version from ${SPARROW_COMPUTE_VERSION_HEADER}")
endif()

project(
    sparrow-compute
    VERSION
        "${SPARROW_COMPUTE_VERSION_MAJOR}.${SPARROW_COMPUTE_VERSION_MINOR}.${SPARROW_COMPUTE_VERSION_PATCH}"
    LANGUAGES C CXX
)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

message(
    STATUS
    "🔧 sparrow-compute v${PROJECT_VERSION}: C++20 project with sparrow + xtensor compute kernels"
)

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
    $<BUILD_INTERFACE:${SPARROW_COMPUTE_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# Enable xtensor SIMD acceleration for consumers of sparrow-compute.
target_compile_definitions(sparrow-compute INTERFACE XTENSOR_USE_XSIMD)
target_link_libraries(sparrow-compute INTERFACE xtensor xsimd sparrow::sparrow)

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
set(SPARROW_COMPUTE_CMAKECONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/cmake/sparrow-compute"
    CACHE STRING "install path for sparrow-computeConfig.cmake"
)

install(
    DIRECTORY "${SPARROW_COMPUTE_INCLUDE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    TARGETS sparrow-compute
    EXPORT sparrow-computeTargets
)

install(
    EXPORT sparrow-computeTargets
    FILE sparrow-computeTargets.cmake
    NAMESPACE sparrow-compute::
    DESTINATION "${SPARROW_COMPUTE_CMAKECONFIG_INSTALL_DIR}"
)

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/sparrow-computeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/sparrow-computeConfig.cmake"
    INSTALL_DESTINATION "${SPARROW_COMPUTE_CMAKECONFIG_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sparrow-computeConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/sparrow-computeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/sparrow-computeConfigVersion.cmake"
    DESTINATION "${SPARROW_COMPUTE_CMAKECONFIG_INSTALL_DIR}"
)

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
