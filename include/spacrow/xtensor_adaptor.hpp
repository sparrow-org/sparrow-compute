#pragma once

/// @file
/// @brief Bridge between sparrow primitive arrays and xtensor expressions.
///
/// A sparrow `primitive_array<T>` owns a contiguous Arrow buffer of values.
/// xtensor can operate on that memory in place by wrapping it in an
/// `xt::xtensor_adaptor` with `xt::no_ownership()`. The resulting adaptor
/// is a 1-D xtensor expression that reads/writes the sparrow buffer directly,
/// enabling lazy broadcasting and SIMD-accelerated kernels without copies.
///
/// @warning The returned adaptor does **not** own the underlying memory.
/// It must not outlive the sparrow array it wraps. Bind it to a local
/// variable whose scope is nested inside the array's lifetime.

#include <cstddef>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xnoalias.hpp>

#include <sparrow/arrow_interface/arrow_array_schema_proxy.hpp>
#include <sparrow/buffer/buffer_view.hpp>
#include <sparrow/layout/array_access.hpp>
#include <sparrow/primitive_array.hpp>

namespace spacrow
{
    namespace detail
    {
        /// @brief Get a raw pointer to the contiguous value buffer of a
        /// sparrow primitive array.
        ///
        /// Uses `sparrow::detail::array_access` to reach the arrow_proxy,
        /// then reads the data buffer (index 1 for primitive arrays: index 0
        /// is the validity bitmap, index 1 is the values buffer) and applies
        /// the array's logical offset.
        template <typename T>
        [[nodiscard]] std::span<const T> view_data(const sparrow::primitive_array<T>& arr)
        {
            const auto& proxy = sparrow::detail::array_access::get_arrow_proxy(arr);
            return {
                proxy.buffers()[1].template data<T>() + static_cast<std::size_t>(proxy.offset()),
                static_cast<std::size_t>(proxy.length())
            };
        }
    }  // namespace detail

    /// @brief Wrap a const sparrow primitive array as a 1-D xtensor adaptor.
    /// Const overload: the adaptor exposes the buffer read-only.
    template <typename T>
    [[nodiscard]] auto as_xtensor_view(const sparrow::primitive_array<T>& arr)
    {
        const auto data = detail::view_data(arr);
        return xt::adapt(data.data(), data.size(), xt::no_ownership());
    }

    /// Build a sparrow primitive array from an xtensor expression
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
}  // namespace spacrow
