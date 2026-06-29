/// @file demo_kernels.cpp
/// @brief Demonstrates spacrow compute kernels over sparrow primitive arrays.

#include <iostream>
#include <vector>

#include <sparrow/primitive_array.hpp>

#include "spacrow/kernels.hpp"

namespace
{
    template <sparrow::primitive_type T>
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

    template <sparrow::primitive_type T>
    void print(const char* label, const sparrow::primitive_array<T>& arr)
    {
        std::cout << label << ": [ ";
        for (std::size_t i = 0; i < arr.size(); ++i)
        {
            std::cout << arr[i].value();
            if (i + 1 < arr.size()) std::cout << ", ";
        }
        std::cout << " ]\n";
    }

    void demo_doubles()
    {
        std::cout << "=== double kernels ===\n";
        sparrow::primitive_array<double> a{1.0, 2.0, 3.0, 4.0};
        sparrow::primitive_array<double> b{10.0, 20.0, 30.0, 40.0};

        print("a        ", a);
        print("b        ", b);

        auto sum = spacrow::add(a, b);
        auto diff = spacrow::subtract(a, b);
        auto prod = spacrow::multiply(a, b);
        auto quot = spacrow::divide(a, b);

        print("a + b    ", sum);
        print("a - b    ", diff);
        print("a * b    ", prod);
        print("a / b    ", quot);
    }

    void demo_ints()
    {
        std::cout << "\n=== int32 kernels ===\n";
        sparrow::primitive_array<int32_t> a{5, 6, 7, 8};
        sparrow::primitive_array<int32_t> b{1, 2, 3, 4};

        print("a        ", a);
        print("b        ", b);
        print("a + b    ", spacrow::add(a, b));
        print("a - b    ", spacrow::subtract(a, b));
        print("a * b    ", spacrow::multiply(a, b));
        print("a / b    ", spacrow::divide(a, b));
    }
}  // namespace

int main()
{
    demo_doubles();
    demo_ints();
    return 0;
}