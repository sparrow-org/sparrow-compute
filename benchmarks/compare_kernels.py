#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


BENCH_RE = re.compile(
    r"^bench_(?P<impl>sparrow|xtensor)_(?P<op>add|subtract|multiply|divide)"
    r"<(?P<dtype>[^>]+)>/(?P<size>\d+)$"
)

TYPE_ALIASES = {
    "int": "int32_t",
    "int32": "int32_t",
    "std::int32_t": "int32_t",
    "float64": "double",
    "float": "double",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sparrow-compute sparrow-backed kernels against pure xtensor kernels."
    )
    parser.add_argument(
        "--benchmark",
        default="build/default/sparrow-compute_bench",
        help="Path to the benchmark binary.",
    )
    parser.add_argument(
        "--ops",
        default="add,subtract,multiply,divide",
        help="Comma-separated operators to include.",
    )
    parser.add_argument(
        "--types",
        default="double,int32_t",
        help="Comma-separated benchmark types to include.",
    )
    parser.add_argument(
        "--sizes",
        default="100,1000,10000,100000,1000000",
        help="Comma-separated input sizes to include.",
    )
    parser.add_argument(
        "--benchmark-min-time",
        default="0.05s",
        help="Passed through to Google Benchmark.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the benchmark target before running.",
    )
    parser.add_argument(
        "--build-dir",
        default="build/default",
        help="Build directory used when --build is set.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_types(dtypes: list[str]) -> list[str]:
    return [TYPE_ALIASES.get(dtype, dtype) for dtype in dtypes]


def build_benchmark(build_dir: str) -> None:
    cmd = ["cmake", "--build", build_dir, "--target", "sparrow-compute_bench"]
    subprocess.run(cmd, check=True)


def benchmark_filter(ops: list[str], dtypes: list[str], sizes: list[str]) -> str:
    ops_group = "|".join(re.escape(op) for op in ops)
    type_group = "|".join(re.escape(dtype) for dtype in dtypes)
    size_group = "|".join(re.escape(size) for size in sizes)
    return (
        rf"^bench_(sparrow|xtensor)_({ops_group})<({type_group})>/({size_group})$"
    )


def run_benchmark(binary: str, bench_filter: str, min_time: str) -> dict:
    cmd = [
        binary,
        f"--benchmark_filter={bench_filter}",
        f"--benchmark_min_time={min_time}",
        "--benchmark_format=json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        message = (result.stdout or result.stderr).strip()
        if not message:
            message = "benchmark runner produced no output"
        raise RuntimeError(
            "benchmark runner did not return JSON.\n"
            f"filter: {bench_filter}\n"
            f"output: {message}"
        ) from None


def time_in_ns(entry: dict) -> float:
    unit_scale = {
        "ns": 1.0,
        "us": 1_000.0,
        "ms": 1_000_000.0,
        "s": 1_000_000_000.0,
    }
    return float(entry["cpu_time"]) * unit_scale[entry["time_unit"]]


def format_ns(ns: float) -> str:
    if ns >= 1_000_000_000.0:
        return f"{ns / 1_000_000_000.0:.3f}s"
    if ns >= 1_000_000.0:
        return f"{ns / 1_000_000.0:.3f}ms"
    if ns >= 1_000.0:
        return f"{ns / 1_000.0:.3f}us"
    return f"{ns:.1f}ns"


def format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return "-"
    return f"{numerator / denominator:.2f}x"


def collect_rows(payload: dict) -> dict[tuple[str, str, str], dict[str, float]]:
    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    for bench in payload.get("benchmarks", []):
        name = bench.get("name", "")
        match = BENCH_RE.match(name)
        if not match:
            continue
        key = (match.group("op"), match.group("dtype"), match.group("size"))
        variant = match.group("impl")
        rows.setdefault(key, {})[variant] = time_in_ns(bench)
    return rows


def print_table(rows: dict[tuple[str, str, str], dict[str, float]]) -> None:
    header = (
        "op",
        "type",
        "size",
        "sparrow",
        "xtensor",
        "xtensor/sparrow",
    )
    print(" | ".join(header))
    print(" | ".join("-" * len(col) for col in header))

    def sort_key(item: tuple[tuple[str, str, str], dict[str, float]]) -> tuple[str, str, int]:
        (op, dtype, size), _ = item
        return (dtype, op, int(size))

    for (op, dtype, size), variants in sorted(rows.items(), key=sort_key):
        sparrow = variants.get("sparrow")
        xtensor = variants.get("xtensor")
        row = (
            op,
            dtype,
            size,
            format_ns(sparrow) if sparrow is not None else "-",
            format_ns(xtensor) if xtensor is not None else "-",
            format_ratio(xtensor, sparrow),
        )
        print(" | ".join(row))


def main() -> int:
    args = parse_args()
    benchmark = Path(args.benchmark)
    if args.build:
        build_benchmark(args.build_dir)
    if not benchmark.exists():
        print(f"benchmark binary not found: {benchmark}", file=sys.stderr)
        return 1

    ops = split_csv(args.ops)
    dtypes = normalize_types(split_csv(args.types))
    sizes = split_csv(args.sizes)

    try:
        payload = run_benchmark(
            str(benchmark),
            benchmark_filter(ops, dtypes, sizes),
            args.benchmark_min_time,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    rows = collect_rows(payload)
    if not rows:
        print("no matching benchmarks found", file=sys.stderr)
        return 1
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
