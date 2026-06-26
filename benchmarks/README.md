# Benchmarks

This directory contains the microbenchmarks used to compare `spacrow`
sparrow-backed kernels against equivalent pure xtensor kernels.

## Build

Default local timing build:

```bash
cmake -S . -B build/default -DSPACROW_CODSPEED_MODE=off
cmake --build build/default --target spacrow_bench
```

CodSpeed wall-time mode:

```bash
cmake -S . -B build/default -DSPACROW_CODSPEED_MODE=walltime
cmake --build build/default --target spacrow_bench
```

CodSpeed instrumentation mode:

```bash
cmake -S . -B build/default \
  -DSPACROW_CODSPEED_MODE=instrumentation \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/default --target spacrow_bench
```

Allowed `SPACROW_CODSPEED_MODE` values are `off`, `instrumentation`,
`simulation`, `memory`, and `walltime`.

## Run

Run all benchmarks:

```bash
./build/default/spacrow_bench
```

Run a filtered benchmark and emit JSON:

```bash
./build/default/spacrow_bench \
  --benchmark_filter='(bench_sparrow_add<double>/1000000|bench_xtensor_add<double>/1000000)$' \
  --benchmark_format=json
```

The benchmark source is [bench_kernels.cpp](/home/alexisp/Dev/spacrow/benchmarks/bench_kernels.cpp).

## Compare Kernels

Use the helper script to compare sparrow and xtensor variants side by side:

```bash
python3 benchmarks/compare_kernels.py \
  --ops add \
  --types float \
  --sizes 1000000 \
  --benchmark-min-time 0.02s
```

Build before running the comparison:

```bash
python3 benchmarks/compare_kernels.py --build
```

## Notes

- CodSpeed's Google Benchmark compatibility layer does not support benchmark
  declarations inside anonymous namespaces.
- Local profiling with `perf` is typically most useful with
  `SPACROW_CODSPEED_MODE=off`.
