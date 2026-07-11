#pragma once

#include <functional>
#include <stdexcept>

#include <xtensor/containers/xtensor.hpp>

#include <sparrow/primitive_array.hpp>
#include <sparrow/u8_buffer.hpp>

#include "sparrow-compute/xtensor_adaptor.hpp"

namespace sparrow::compute
{
    namespace detail
    {
        template <typename T>
        concept primitive_array_c = sparrow::is_primitive_array_v<std::decay_t<T>>;

        // Throws an exception if the two arrays do not have the same size.
        template <primitive_array_c A, primitive_array_c... Args>
        inline void ensure_same_size(const A& a, const Args&... args)
        {
            if (!((a.size() == args.size()) && ...))
            {
                throw std::invalid_argument("sparrow-compute kernels require equal-length inputs");
            }
        }
    
        template <typename Func, primitive_array_c... Args>
        using sparrow_op_return_type =
            decltype(std::declval<Func>()(typename std::decay_t<Args>::inner_value_type()...));

        template <typename Func, primitive_array_c... Args>
        [[nodiscard]] sparrow::primitive_array<sparrow_op_return_type<Func, Args...>>
        sparrow_op(Func&& func, Args&&... args)
        {
            ensure_same_size(args...);
            using return_type = sparrow_op_return_type<Func, Args...>;
            return to_sparrow<return_type>(func(as_xtensor_view(std::forward<Args>(args))...));
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
        return detail::sparrow_op(std::plus{}, a, b);
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
        return detail::sparrow_op(std::minus{}, a, b);
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
        return detail::sparrow_op(std::multiplies<>{}, a, b);
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
        return detail::sparrow_op(std::divides<>{}, a, b);
    }

}  // namespace sparrow::compute
