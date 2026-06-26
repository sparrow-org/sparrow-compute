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

#include <sparrow/arrow_interface/arrow_array_schema_proxy.hpp>
#include <sparrow/buffer/buffer_view.hpp>
#include <sparrow/layout/array_access.hpp>
#include <sparrow/primitive_array.hpp>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace spacrow
{
    namespace detail
    {
        template <typename T>
        struct array_view_data
        {
            T* ptr;
            std::size_t size;
        };

        /// @brief Get a raw pointer to the contiguous value buffer of a
        /// sparrow primitive array.
        ///
        /// Uses `sparrow::detail::array_access` to reach the arrow_proxy,
        /// then reads the data buffer (index 1 for primitive arrays: index 0
        /// is the validity bitmap, index 1 is the values buffer) and applies
        /// the array's logical offset.
        template <typename T>
        [[nodiscard]] array_view_data<T> view_data(sparrow::primitive_array<T>& arr)
        {
            auto& proxy = sparrow::detail::array_access::get_arrow_proxy(arr);
            return {
                proxy.buffers()[1].template data<T>() + static_cast<std::size_t>(proxy.offset()),
                static_cast<std::size_t>(proxy.length())
            };
        }

        template <typename T>
        [[nodiscard]] array_view_data<const T> view_data(const sparrow::primitive_array<T>& arr)
        {
            const auto& proxy = sparrow::detail::array_access::get_arrow_proxy(arr);
            return {
                proxy.buffers()[1].template data<T>() + static_cast<std::size_t>(proxy.offset()),
                static_cast<std::size_t>(proxy.length())
            };
        }

        template <typename T>
        [[nodiscard]] T* raw_data(sparrow::primitive_array<T>& arr)
        {
            return view_data(arr).ptr;
        }

        template <typename T>
        [[nodiscard]] const T* raw_data(const sparrow::primitive_array<T>& arr)
        {
            return view_data(arr).ptr;
        }

        /// @brief Number of logical elements in a sparrow primitive array.
        template <typename T>
        [[nodiscard]] std::size_t size_of(const sparrow::primitive_array<T>& arr)
        {
            return view_data(arr).size;
        }
    }  // namespace detail

    /// @brief Wrap a mutable sparrow primitive array as a 1-D xtensor adaptor.
    ///
    /// The adaptor references the array's value buffer with `no_ownership`:
    /// writes to the adaptor mutate the array in place, and the array must
    /// outlive the adaptor.
    template <typename T>
    [[nodiscard]] auto as_xtensor_view(sparrow::primitive_array<T>& arr)
    {
        const auto data = detail::view_data(arr);
        return xt::adapt(data.ptr, data.size, xt::no_ownership());
    }

    /// @brief Wrap a const sparrow primitive array as a 1-D xtensor adaptor.
    /// Const overload: the adaptor exposes the buffer read-only.
    template <typename T>
    [[nodiscard]] auto as_xtensor_view(const sparrow::primitive_array<T>& arr)
    {
        const auto data = detail::view_data(arr);
        return xt::adapt(data.ptr, data.size, xt::no_ownership());
    }
}  // namespace spacrow
