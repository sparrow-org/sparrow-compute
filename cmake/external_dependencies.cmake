include(FetchContent)

# ---------------------------------------------------------------------------
# Fetch strategy
#   OFF     = require pre-installed packages (find_package REQUIRED)
#   ON      = always fetch via FetchContent
#   MISSING = try find_package first, fall back to FetchContent
# ---------------------------------------------------------------------------
option(FETCH_DEPENDENCIES_WITH_CMAKE
    "Fetch dependencies with CMake: OFF, ON, or MISSING"
    MISSING
)
message(STATUS "🔧 FETCH_DEPENDENCIES_WITH_CMAKE: ${FETCH_DEPENDENCIES_WITH_CMAKE}")

if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "OFF")
    set(FIND_PACKAGE_OPTIONS REQUIRED)
else()
    set(FIND_PACKAGE_OPTIONS QUIET)
endif()

# Remember the user's intent before we override the cache.
# Dependencies must NOT build their own test/benchmark suites, so we force
# BUILD_TESTS=OFF and BUILD_BENCHMARKS=OFF globally. The saved values gate
# whether sparrow-compute fetches googletest / benchmark.
set(_sparrow-compute_build_tests ${BUILD_TESTS})
set(_sparrow-compute_build_benchmarks ${BUILD_BENCHMARKS})
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

# Pinned versions for reproducibility.
set(SPARROW_COMPUTE_XTL_VERSION         "0.8.0")
set(SPARROW_COMPUTE_XSIMD_VERSION       "13.2.0")
set(SPARROW_COMPUTE_XTENSOR_VERSION     "0.27.1")
set(SPARROW_COMPUTE_SPARROW_VERSION     "2.4.0")
set(SPARROW_COMPUTE_GTEST_VERSION       "v1.15.2")
set(SPARROW_COMPUTE_CODSPEED_CPP_VERSION "v2.4.0")

# ---------------------------------------------------------------------------
# xtl — required by xtensor
# ---------------------------------------------------------------------------
if(NOT FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON")
    find_package(xtl CONFIG ${FIND_PACKAGE_OPTIONS})
endif()
if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON"
   OR (FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "MISSING" AND NOT xtl_FOUND))
    message(STATUS "📦 Fetching xtl ${SPARROW_COMPUTE_XTL_VERSION}")
    FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG        ${SPARROW_COMPUTE_XTL_VERSION}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(xtl)
    message(STATUS "   ✅ Fetched xtl ${SPARROW_COMPUTE_XTL_VERSION}")
elseif(xtl_FOUND)
    message(STATUS "📦 xtl found: ${xtl_DIR}")
endif()

# ---------------------------------------------------------------------------
# xsimd — optional SIMD backend for xtensor
# ---------------------------------------------------------------------------
if(NOT FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON")
    find_package(xsimd CONFIG ${FIND_PACKAGE_OPTIONS})
endif()
if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON"
   OR (FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "MISSING" AND NOT xsimd_FOUND))
    message(STATUS "📦 Fetching xsimd ${SPARROW_COMPUTE_XSIMD_VERSION}")
    FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG        ${SPARROW_COMPUTE_XSIMD_VERSION}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(xsimd)
    message(STATUS "   ✅ Fetched xsimd ${SPARROW_COMPUTE_XSIMD_VERSION}")
elseif(xsimd_FOUND)
    message(STATUS "📦 xsimd found: ${xsimd_DIR}")
endif()

# ---------------------------------------------------------------------------
# xtensor — header-only
# ---------------------------------------------------------------------------
if(NOT FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON")
    find_package(xtensor CONFIG ${FIND_PACKAGE_OPTIONS})
endif()
if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON"
   OR (FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "MISSING" AND NOT xtensor_FOUND))
    message(STATUS "📦 Fetching xtensor ${SPARROW_COMPUTE_XTENSOR_VERSION}")
    set(XTENSOR_ENABLE_ASSERT OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG        ${SPARROW_COMPUTE_XTENSOR_VERSION}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(xtensor)
    unset(XTENSOR_ENABLE_ASSERT CACHE)
    message(STATUS "   ✅ Fetched xtensor ${SPARROW_COMPUTE_XTENSOR_VERSION}")
elseif(xtensor_FOUND)
    message(STATUS "📦 xtensor found: ${xtensor_DIR}")
endif()

# ---------------------------------------------------------------------------
# GoogleTest — only fetched when the user requested tests
# ---------------------------------------------------------------------------
if(_sparrow-compute_build_tests)
    if(NOT FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON")
        find_package(GTest CONFIG ${FIND_PACKAGE_OPTIONS})
    endif()
    if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON"
       OR (FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "MISSING" AND NOT GTest_FOUND))
        message(STATUS "📦 Fetching GoogleTest ${SPARROW_COMPUTE_GTEST_VERSION}")
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        ${SPARROW_COMPUTE_GTEST_VERSION}
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
            SYSTEM
            EXCLUDE_FROM_ALL
        )
        FetchContent_MakeAvailable(googletest)
        unset(BUILD_GMOCK CACHE)
        unset(INSTALL_GTEST CACHE)
        message(STATUS "   ✅ Fetched GoogleTest ${SPARROW_COMPUTE_GTEST_VERSION}")
    elseif(GTest_FOUND)
        message(STATUS "📦 GTest found: ${GTest_DIR}")
    endif()
endif()

# ---------------------------------------------------------------------------
# CodSpeed Google Benchmark fork — only fetched when the user requested benchmarks
# ---------------------------------------------------------------------------
if(_sparrow-compute_build_benchmarks)
    if(NOT FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON")
        find_package(benchmark CONFIG ${FIND_PACKAGE_OPTIONS})
    endif()
    if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON"
       OR (FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "MISSING" AND NOT benchmark_FOUND))
        message(STATUS "📦 Fetching CodSpeed benchmark ${SPARROW_COMPUTE_CODSPEED_CPP_VERSION}")
        set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        set(CODSPEED_MODE "${SPARROW_COMPUTE_EFFECTIVE_CODSPEED_MODE}" CACHE STRING "" FORCE)
        FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp.git
            GIT_TAG        ${SPARROW_COMPUTE_CODSPEED_CPP_VERSION}
            SOURCE_SUBDIR  google_benchmark
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
            SYSTEM
            EXCLUDE_FROM_ALL
        )
        FetchContent_MakeAvailable(benchmark)
        unset(BENCHMARK_DOWNLOAD_DEPENDENCIES CACHE)
        unset(BENCHMARK_ENABLE_TESTING CACHE)
        unset(BENCHMARK_ENABLE_INSTALL CACHE)
        unset(BENCHMARK_ENABLE_GTEST_TESTS CACHE)
        unset(CODSPEED_MODE CACHE)
        message(STATUS "   ✅ Fetched CodSpeed benchmark ${SPARROW_COMPUTE_CODSPEED_CPP_VERSION}")
    elseif(benchmark_FOUND)
        message(STATUS "📦 benchmark found: ${benchmark_DIR}")
    endif()
endif()

# ---------------------------------------------------------------------------
# sparrow — Apache Arrow columnar format in C++20
# ---------------------------------------------------------------------------
if(NOT FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON")
    find_package(sparrow CONFIG ${FIND_PACKAGE_OPTIONS})
endif()
if(FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "ON"
   OR (FETCH_DEPENDENCIES_WITH_CMAKE STREQUAL "MISSING" AND NOT sparrow_FOUND))
    message(STATUS "📦 Fetching sparrow ${SPARROW_COMPUTE_SPARROW_VERSION}")
    set(BUILD_DOCS                 OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES             OFF CACHE BOOL "" FORCE)
    set(USE_DATE_POLYFILL          OFF CACHE BOOL "" FORCE)
    set(CREATE_JSON_READER_TARGET  OFF CACHE BOOL "" FORCE)
    set(ENABLE_INTEGRATION_TEST    OFF CACHE BOOL "" FORCE)
    set(SPARROW_BUILD_SHARED       OFF CACHE BOOL "" FORCE)
    set(SPARROW_CONTRACTS_THROW_ON_FAILURE ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        sparrow
        GIT_REPOSITORY https://github.com/sparrow-org/sparrow.git
        GIT_TAG        ${SPARROW_COMPUTE_SPARROW_VERSION}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(sparrow)
    unset(BUILD_DOCS CACHE)
    unset(BUILD_EXAMPLES CACHE)
    unset(USE_DATE_POLYFILL CACHE)
    unset(CREATE_JSON_READER_TARGET CACHE)
    unset(ENABLE_INTEGRATION_TEST CACHE)
    unset(SPARROW_BUILD_SHARED CACHE)
    unset(SPARROW_CONTRACTS_THROW_ON_FAILURE CACHE)
    message(STATUS "   ✅ Fetched sparrow ${SPARROW_COMPUTE_SPARROW_VERSION}")
elseif(sparrow_FOUND)
    message(STATUS "📦 sparrow found: ${sparrow_DIR}")
endif()

# Restore the user's original values so downstream targets (tests, benchmarks)
# see the correct values.
set(BUILD_TESTS ${_sparrow-compute_build_tests} CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS ${_sparrow-compute_build_benchmarks} CACHE BOOL "" FORCE)
