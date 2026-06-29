#include <benchmark/benchmark.h>

#include <vector>

#include <sparrow/primitive_array.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xeval.hpp>
#include <xtensor/core/xnoalias.hpp>

#include "spacrow/kernels.hpp"

// -------------------------------------------------------------------------
// Test data generation
// -------------------------------------------------------------------------

template <typename T>
std::vector<T> make_input(std::size_t n, T start)
{
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        v[i] = static_cast<T>(start + static_cast<T>(i));
    }
    return v;
}

// -------------------------------------------------------------------------
// Sparrow-array kernels (functional: return a new sparrow array)
// -------------------------------------------------------------------------

template <sparrow::primitive_type T>
void bench_sparrow_add(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));
    auto va = make_input<T>(n, T{1});
    auto vb = make_input<T>(n, T{2});
    sparrow::primitive_array<T> a(va);
    sparrow::primitive_array<T> b(vb);

    for (auto _ : state)
    {
        auto r = spacrow::add(a, b);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
}

    template <sparrow::primitive_type T>
    void bench_sparrow_subtract(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        auto va = make_input<T>(n, T{1});
        auto vb = make_input<T>(n, T{2});
        sparrow::primitive_array<T> a(va);
        sparrow::primitive_array<T> b(vb);

        for (auto _ : state)
        {
            auto r = spacrow::subtract(a, b);
            benchmark::DoNotOptimize(r);
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
    }

    template <sparrow::primitive_type T>
    void bench_sparrow_multiply(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        auto va = make_input<T>(n, T{1});
        auto vb = make_input<T>(n, T{2});
        sparrow::primitive_array<T> a(va);
        sparrow::primitive_array<T> b(vb);

        for (auto _ : state)
        {
            auto r = spacrow::multiply(a, b);
            benchmark::DoNotOptimize(r);
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
    }

    template <sparrow::primitive_type T>
    void bench_sparrow_divide(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        auto va = make_input<T>(n, T{1});
        auto vb = make_input<T>(n, T{2});
        sparrow::primitive_array<T> a(va);
        sparrow::primitive_array<T> b(vb);

        for (auto _ : state)
        {
            auto r = spacrow::divide(a, b);
            benchmark::DoNotOptimize(r);
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
    }

    // -------------------------------------------------------------------------
    // xtensor-vector kernels (functional: return a new xt::xtensor<T, 1>)
    // -------------------------------------------------------------------------

    template <sparrow::primitive_type T>
    void bench_xtensor_add(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        auto va = make_input<T>(n, T{1});
        auto vb = make_input<T>(n, T{2});
        xt::xtensor<T, 1> a = xt::adapt(va, std::array<std::size_t, 1>{n});
        xt::xtensor<T, 1> b = xt::adapt(vb, std::array<std::size_t, 1>{n});

        for (auto _ : state)
        {
            auto r = xt::eval(a + b);
            benchmark::DoNotOptimize(r);
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
    }

    template <sparrow::primitive_type T>
    void bench_xtensor_subtract(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        auto va = make_input<T>(n, T{1});
        auto vb = make_input<T>(n, T{2});
        xt::xtensor<T, 1> a = xt::adapt(va, std::array<std::size_t, 1>{n});
        xt::xtensor<T, 1> b = xt::adapt(vb, std::array<std::size_t, 1>{n});

        for (auto _ : state)
        {
            auto r = xt::eval(a - b);
            benchmark::DoNotOptimize(r);
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
    }

    template <sparrow::primitive_type T>
    void bench_xtensor_multiply(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        auto va = make_input<T>(n, T{1});
        auto vb = make_input<T>(n, T{2});
        xt::xtensor<T, 1> a = xt::adapt(va, std::array<std::size_t, 1>{n});
        xt::xtensor<T, 1> b = xt::adapt(vb, std::array<std::size_t, 1>{n});

        for (auto _ : state)
        {
            auto r = xt::eval(a * b);
            benchmark::DoNotOptimize(r);
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
    }

    template <sparrow::primitive_type T>
    void bench_xtensor_divide(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        auto va = make_input<T>(n, T{1});
        auto vb = make_input<T>(n, T{2});
        xt::xtensor<T, 1> a = xt::adapt(va, std::array<std::size_t, 1>{n});
        xt::xtensor<T, 1> b = xt::adapt(vb, std::array<std::size_t, 1>{n});

        for (auto _ : state)
        {
            auto r = xt::eval(a / b);
            benchmark::DoNotOptimize(r);
        }
        state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * sizeof(T)));
    }
// Register benchmarks for double and int32_t across a range of sizes.
#define REGISTER_BENCHMARKS(T)                                                \
    BENCHMARK_TEMPLATE(bench_sparrow_add, T)->RangeMultiplier(10)->Range(100, 1000000); \
    BENCHMARK_TEMPLATE(bench_sparrow_subtract, T)->RangeMultiplier(10)->Range(100, 1000000); \
    BENCHMARK_TEMPLATE(bench_sparrow_multiply, T)->RangeMultiplier(10)->Range(100, 1000000); \
    BENCHMARK_TEMPLATE(bench_sparrow_divide, T)->RangeMultiplier(10)->Range(100, 1000000); \
    BENCHMARK_TEMPLATE(bench_xtensor_add, T)->RangeMultiplier(10)->Range(100, 1000000); \
    BENCHMARK_TEMPLATE(bench_xtensor_subtract, T)->RangeMultiplier(10)->Range(100, 1000000); \
    BENCHMARK_TEMPLATE(bench_xtensor_multiply, T)->RangeMultiplier(10)->Range(100, 1000000); \
    BENCHMARK_TEMPLATE(bench_xtensor_divide, T)->RangeMultiplier(10)->Range(100, 1000000);

REGISTER_BENCHMARKS(double)
REGISTER_BENCHMARKS(int32_t)

// =========================================================================
// Step-by-step breakdown (flamegraph substitute — times each sub-step of
// to_sparrow individually at 1M elements to pinpoint remaining overhead).
// Benchmarks are in milliseconds so the per-call costs are visible.
// =========================================================================

template <sparrow::primitive_type T>
void bench_breakdown_allocate(benchmark::State& state)
{
    const auto n = 1000000;
    using alloc_type = typename sparrow::u8_buffer<T>::default_allocator;
    for (auto _ : state) {
        alloc_type alloc;
        auto* raw = alloc.allocate(n * sizeof(T));
        benchmark::DoNotOptimize(reinterpret_cast<T*>(raw));
        alloc.deallocate(raw, n * sizeof(T));
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <sparrow::primitive_type T>
void bench_breakdown_eval(benchmark::State& state)
{
    const auto n = 1000000;
    auto va = make_input<T>(n, T{1});
    auto vb = make_input<T>(n, T{2});
    sparrow::primitive_array<T> a(va);
    sparrow::primitive_array<T> b(vb);
    auto expr = spacrow::as_xtensor_view(a) + spacrow::as_xtensor_view(b);

    // Pre-allocate once before the loop.
    using alloc_type = typename sparrow::u8_buffer<T>::default_allocator;
    alloc_type alloc;
    auto* raw = alloc.allocate(n * sizeof(T));
    auto* ptr = reinterpret_cast<T*>(raw);
    auto view = xt::adapt(ptr, n, xt::no_ownership());

    for (auto _ : state) {
        xt::noalias(view) = expr;
        benchmark::ClobberMemory();
    }
    alloc.deallocate(raw, n * sizeof(T));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <sparrow::primitive_type T>
void bench_breakdown_arrow_ctor(benchmark::State& state)
{
    const auto n = 1000000;
    using alloc_type = typename sparrow::u8_buffer<T>::default_allocator;

    for (auto _ : state) {
        alloc_type alloc;
        auto* raw = alloc.allocate(n * sizeof(T));
        auto* ptr = reinterpret_cast<T*>(raw);
        // u8_buffer takes ownership → destroys via 'alloc'.
        sparrow::u8_buffer<T> buf(ptr, n, alloc);
        auto result = sparrow::primitive_array<T>(std::move(buf), n, /*nullable=*/false);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// Register at 1M elements, output in ms for per-call visibility.
#define REGISTER_BREAKDOWN(T)                                                 \
    BENCHMARK_TEMPLATE(bench_breakdown_allocate, T)->Unit(benchmark::kMillisecond); \
    BENCHMARK_TEMPLATE(bench_breakdown_eval, T)->Unit(benchmark::kMillisecond);    \
    BENCHMARK_TEMPLATE(bench_breakdown_arrow_ctor, T)->Unit(benchmark::kMillisecond);

REGISTER_BREAKDOWN(double)
REGISTER_BREAKDOWN(int32_t)

BENCHMARK_MAIN();
