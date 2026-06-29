#pragma once

#include <cstddef>

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xbuffer_adaptor.hpp>
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
        /// Get a raw pointer to the contiguous value buffer of a
        /// sparrow primitive array.
        template <sparrow::primitive_type T>
        [[nodiscard]] std::span<const T> view_data(const sparrow::primitive_array<T>& arr)
        {
            const auto& proxy = sparrow::detail::array_access::get_arrow_proxy(arr);
            return {
                proxy.buffers()[1].template data<T>() + static_cast<std::size_t>(proxy.offset()),
                static_cast<std::size_t>(proxy.length())
            };
        }
    }  // namespace detail

    /// Wrap a const sparrow primitive array as a 1-D xtensor adaptor.
    /// Const overload: the adaptor exposes the buffer read-only.
    template <sparrow::primitive_type T>
    [[nodiscard]] auto as_xtensor_view(const sparrow::primitive_array<T>& arr)
    {
        const auto data = detail::view_data(arr);
        return xt::adapt(xt::xbuffer_adaptor<const T*, xt::no_ownership>(data.data(), data.size()));
    }

    /// Build a sparrow primitive array from an xtensor expression
    /// with zero-copy evaluation.
    template <sparrow::primitive_type T, class E>
    [[nodiscard]] sparrow::primitive_array<T> to_sparrow(E&& expr)
    {
        using alloc_type = typename sparrow::u8_buffer<T>::default_allocator;
        alloc_type alloc;

        const auto n = static_cast<std::size_t>(expr.shape()[0]);

        auto* raw_bytes = alloc.allocate(n * sizeof(T));
        T* typed = reinterpret_cast<T*>(raw_bytes);

        auto view = xt::adapt(xt::xbuffer_adaptor<T*, xt::no_ownership>(typed, n));
        xt::noalias(view) = std::forward<E>(expr);

        sparrow::u8_buffer<T> buf(typed, n, alloc);
        return sparrow::primitive_array<T>(std::move(buf), n, /*nullable=*/false);
    }
}  // namespace spacrow
