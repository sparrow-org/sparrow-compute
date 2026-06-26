/// @file test_kernels.cpp
/// @brief GoogleTest cases for spacrow compute kernels.

#include <gtest/gtest.h>

#include <sparrow/primitive_array.hpp>

#include "spacrow/kernels.hpp"

namespace
{
    template <class T>
    std::vector<T> to_vector(const sparrow::primitive_array<T>& arr)
    {
        std::vector<T> v;
        v.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i)
        {
            v.push_back(arr[i].value());
        }
        return v;
    }
}  // namespace

// ---------------------------------------------------------------------------
// Functional kernels
// ---------------------------------------------------------------------------

TEST(Kernels, AddDoubles)
{
    sparrow::primitive_array<double> a{1.0, 2.0, 3.0, 4.0};
    sparrow::primitive_array<double> b{10.0, 20.0, 30.0, 40.0};

    auto r = spacrow::add(a, b);
    ASSERT_EQ(r.size(), 4u);
    EXPECT_EQ(to_vector(r), (std::vector<double>{11.0, 22.0, 33.0, 44.0}));
}

TEST(Kernels, SubtractInts)
{
    sparrow::primitive_array<int32_t> a{10, 20, 30, 40};
    sparrow::primitive_array<int32_t> b{1, 2, 3, 4};

    auto r = spacrow::subtract(a, b);
    ASSERT_EQ(r.size(), 4u);
    EXPECT_EQ(to_vector(r), (std::vector<int32_t>{9, 18, 27, 36}));
}

TEST(Kernels, MultiplyDoubles)
{
    sparrow::primitive_array<double> a{1.0, 2.0, 3.0};
    sparrow::primitive_array<double> b{2.0, 3.0, 4.0};

    auto r = spacrow::multiply(a, b);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(to_vector(r), (std::vector<double>{2.0, 6.0, 12.0}));
}

TEST(Kernels, DivideDoubles)
{
    sparrow::primitive_array<double> a{10.0, 20.0, 30.0};
    sparrow::primitive_array<double> b{2.0, 5.0, 10.0};

    auto r = spacrow::divide(a, b);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(to_vector(r), (std::vector<double>{5.0, 4.0, 3.0}));
}

TEST(Kernels, DivideIntsTruncates)
{
    sparrow::primitive_array<int32_t> a{7, 8, 9};
    sparrow::primitive_array<int32_t> b{2, 3, 4};

    auto r = spacrow::divide(a, b);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(to_vector(r), (std::vector<int32_t>{3, 2, 2}));
}

// ---------------------------------------------------------------------------
// In-place kernels
// ---------------------------------------------------------------------------

TEST(Kernels, InPlaceAddResizesAndWrites)
{
    sparrow::primitive_array<double> a{1.0, 2.0, 3.0};
    sparrow::primitive_array<double> b{10.0, 20.0, 30.0};
    // sparrow primitive arrays have no default constructor; init from empty.
    sparrow::primitive_array<double> out{std::vector<double>{}};

    spacrow::add(a, b, out);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(to_vector(out), (std::vector<double>{11.0, 22.0, 33.0}));
}

TEST(Kernels, InPlaceMultiplyOverwritesExisting)
{
    sparrow::primitive_array<int32_t> a{1, 2, 3, 4};
    sparrow::primitive_array<int32_t> b{2, 3, 4, 5};
    sparrow::primitive_array<int32_t> out{0, 0, 0, 0};

    spacrow::multiply(a, b, out);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(to_vector(out), (std::vector<int32_t>{2, 6, 12, 20}));
}

TEST(Kernels, InPlaceSubtractMatchesFunctional)
{
    sparrow::primitive_array<double> a{5.0, 6.0, 7.0};
    sparrow::primitive_array<double> b{1.0, 2.0, 3.0};

    auto functional = spacrow::subtract(a, b);
    sparrow::primitive_array<double> in_place{std::vector<double>{}};
    spacrow::subtract(a, b, in_place);

    ASSERT_EQ(functional.size(), in_place.size());
    EXPECT_EQ(to_vector(functional), to_vector(in_place));
}