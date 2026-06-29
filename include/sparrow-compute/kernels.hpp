#pragma once

#include <stdexcept>

#include <xtensor/containers/xtensor.hpp>

#include <sparrow/primitive_array.hpp>
#include <sparrow/u8_buffer.hpp>

#include "sparrow-compute/xtensor_adaptor.hpp"

namespace sparrow::compute
{
    namespace detail
    {
        // Throws an exception if the two arrays do not have the same size.
        template <typename T>
        inline void ensure_same_size(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
        {
            if (a.size() != b.size())
            {
                throw std::invalid_argument("sparrow-compute kernels require equal-length inputs");
            }
        }
    }  // namespace detail

    // -------------------------------------------------------------------------
    // Functional kernels — return a new sparrow array
    // -------------------------------------------------------------------------

    /**
     * Element wise addition: returns `a + b` as a new sparrow array.
     * @tparam T the type of the values in the input arrays.
     * @param a the first input array.
     * @param b the second input array.
     * @return a new sparrow array containing the element wise sum of `a` and `b`.
     * @throws std::invalid_argument if the input arrays do not have the same size.
     */
    template <sparrow::primitive_type T>
    [[nodiscard]] sparrow::primitive_array<T>
    add(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view + b_view);
    }

    /**
     * Element wise subtraction: returns `a - b` as a new sparrow array.
     * @tparam T the type of the values in the input arrays.
     * @param a the first input array.
     * @param b the second input array.
     * @return a new sparrow array containing the element wise difference of `a` and `b`.
     * @throws std::invalid_argument if the input arrays do not have the same size.
     */
    template <sparrow::primitive_type T>
    [[nodiscard]] sparrow::primitive_array<T>
    subtract(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view - b_view);
    }

    /**
     * Element wise multiplication: returns `a * b` as a new sparrow array.
     * @tparam T the type of the values in the input arrays.
     * @param a the first input array.
     * @param b the second input array.
     * @return a new sparrow array containing the element wise product of `a` and `b`.
     * @throws std::invalid_argument if the input arrays do not have the same size.
     */
    template <sparrow::primitive_type T>
    [[nodiscard]] sparrow::primitive_array<T>
    multiply(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view * b_view);
    }

    /**
     * Element wise division: returns `a / b` as a new sparrow array.
     * @tparam T the type of the values in the input arrays.
     * @param a the first input array.
     * @param b the second input array.
     * @return a new sparrow array containing the element wise quotient of `a` and `b`.
     * @throws std::invalid_argument if the input arrays do not have the same size.
     * @note No divide-by-zero check; behavior follows the C++ operator for `T`.
     */
    template <sparrow::primitive_type T>
    [[nodiscard]] sparrow::primitive_array<T>
    divide(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        const auto a_view = as_xtensor_view(a);
        const auto b_view = as_xtensor_view(b);
        return to_sparrow<T>(a_view / b_view);
    }

}  // namespace sparrow::compute
