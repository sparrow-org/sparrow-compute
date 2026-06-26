#pragma once

#include <stdexcept>

#include <xtensor/containers/xtensor.hpp>

#include <sparrow/primitive_array.hpp>
#include <sparrow/u8_buffer.hpp>

#include "spacrow/xtensor_adaptor.hpp"

namespace spacrow
{
    namespace detail
    {
        template <typename T>
        inline void ensure_same_size(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
        {
            if (a.size() != b.size())
            {
                throw std::invalid_argument("spacrow kernels require equal-length inputs");
            }
        }
    }  // namespace detail

    // -------------------------------------------------------------------------
    // Functional kernels — return a new sparrow array
    // -------------------------------------------------------------------------

    /// Element wise addition: returns `a + b` as a new sparrow array.
    template <typename T>
    [[nodiscard]] sparrow::primitive_array<T>
    add(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view + b_view);
    }

    /// Element wise subtraction: returns `a - b` as a new sparrow array.
    template <typename T>
    [[nodiscard]] sparrow::primitive_array<T>
    subtract(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view - b_view);
    }

    /// Element wise multiplication: returns `a * b` as a new sparrow array.
    template <typename T>
    [[nodiscard]] sparrow::primitive_array<T>
    multiply(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view * b_view);
    }

    /// Element wise division: returns `a / b` as a new sparrow array.
    ///
    /// @note No divide-by-zero check; behavior follows the C++ operator for `T`.
    template <typename T>
    [[nodiscard]] sparrow::primitive_array<T>
    divide(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view / b_view);
    }

}  // namespace spacrow
