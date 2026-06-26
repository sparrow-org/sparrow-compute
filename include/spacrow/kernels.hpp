#pragma once

/// @file
/// @brief Basic elementwise compute kernels over sparrow primitive arrays,
/// implemented by adapting the arrays' contiguous value buffers into xtensor
/// expressions and evaluating the result.
///
/// @note v1 ignores Arrow validity bitmaps: kernels operate on the raw value
/// buffers and assume all elements are non-null. Null-mask propagation is left
/// as future work. Inputs must have equal length (no broadcasting across
/// arrays of different sizes — sparrow primitive arrays are 1-D).

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <sparrow/primitive_array.hpp>
#include <sparrow/u8_buffer.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xnoalias.hpp>

#include "spacrow/xtensor_adaptor.hpp"

namespace spacrow
{
    namespace detail
    {
        template <class T>
        inline void ensure_same_size(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
        {
            if (a.size() != b.size())
            {
                throw std::invalid_argument("spacrow kernels require equal-length inputs");
            }
        }

        /// @brief Build a sparrow primitive array from an xtensor expression
        /// with zero-copy evaluation.
        ///
        /// Allocates a buffer compatible with sparrow's allocator, evaluates
        /// the xtensor expression directly into it via `xt::noalias` (no
        /// temporary variable, no intermediate heap allocation), then
        /// transfers ownership of the buffer to sparrow via
        /// `u8_buffer<T>(T*, count, alloc)`.
        ///
        /// This is the single-allocation path: the only heap allocation is
        /// the sparrow Arrow buffer itself, matching pure xtensor performance.
        template <class T, class E>
        [[nodiscard]] sparrow::primitive_array<T> to_sparrow(E&& expr)
        {
            using alloc_type = typename sparrow::u8_buffer<T>::default_allocator;
            alloc_type alloc;

            const auto n = static_cast<std::size_t>(expr.shape()[0]);

            auto* raw_bytes = alloc.allocate(n * sizeof(T));
            auto* typed = reinterpret_cast<T*>(raw_bytes);

            auto view = xt::adapt(typed, n, xt::no_ownership());
            xt::noalias(view) = std::forward<E>(expr);

            sparrow::u8_buffer<T> buf(typed, n, alloc);
            return sparrow::primitive_array<T>(std::move(buf), n, /*nullable=*/false);
        }

    }  // namespace detail

    // -------------------------------------------------------------------------
    // Functional kernels — return a new sparrow array
    // -------------------------------------------------------------------------

    /// @brief Elementwise addition: returns `a + b` as a new sparrow array.
    template <class T>
    [[nodiscard]] sparrow::primitive_array<T>
    add(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        auto a_view = as_xtensor_view(a);
        auto b_view = as_xtensor_view(b);
        return detail::to_sparrow<T>(a_view + b_view);
    }

    /// @brief Elementwise subtraction: returns `a - b` as a new sparrow array.
    template <class T>
    [[nodiscard]] sparrow::primitive_array<T>
    subtract(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        auto a_view = as_xtensor_view(a);
        auto b_view = as_xtensor_view(b);
        return detail::to_sparrow<T>(a_view - b_view);
    }

    /// @brief Elementwise multiplication: returns `a * b` as a new sparrow array.
    template <class T>
    [[nodiscard]] sparrow::primitive_array<T>
    multiply(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        auto a_view = as_xtensor_view(a);
        auto b_view = as_xtensor_view(b);
        return detail::to_sparrow<T>(a_view * b_view);
    }

    /// @brief Elementwise division: returns `a / b` as a new sparrow array.
    ///
    /// @note No divide-by-zero check; behavior follows the C++ operator for `T`.
    template <class T>
    [[nodiscard]] sparrow::primitive_array<T>
    divide(const sparrow::primitive_array<T>& a, const sparrow::primitive_array<T>& b)
    {
        detail::ensure_same_size(a, b);
        auto a_view = as_xtensor_view(a);
        auto b_view = as_xtensor_view(b);
        return detail::to_sparrow<T>(a_view / b_view);
    }

}  // namespace spacrow
