#!/usr/bin/env python3
"""Simple load test utility for the /embeddings endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import math
import time
from pathlib import Path
from typing import Iterable

import httpx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the embeddings endpoint with sample images."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/embeddings",
        help="Target embeddings endpoint URL.",
    )
    parser.add_argument(
        "--images-dir",
        default="face_v3/images_test",
        help="Directory containing sample images.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Total number of requests to issue.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images per request.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of in-flight requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--include-aligned",
        action="store_true",
        help="Request aligned face crops in responses.",
    )
    return parser.parse_args()


def _collect_images(directory: Path) -> list[dict[str, str]]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    encoded: list[dict[str, str]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        encoded.append(
            {
                "id": path.stem,
                "data": base64.b64encode(path.read_bytes()).decode("utf-8"),
            }
        )

    if not encoded:
        raise RuntimeError(f"No images found under {directory}")
    return encoded


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]

    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


async def _issue_request(
    client: httpx.AsyncClient,
    url: str,
    images: Iterable[dict[str, str]],
    include_aligned: bool,
) -> tuple[int, float]:
    payload = {
        "images": list(images),
        "return_aligned_faces": include_aligned,
    }

    start = time.perf_counter()
    response = await client.post(url, json=payload)
    latency = time.perf_counter() - start
    return response.status_code, latency


async def _run_benchmark(args: argparse.Namespace) -> None:
    encoded_images = _collect_images(Path(args.images_dir))
    batch_size = max(1, args.batch_size)
    total_requests = max(1, args.requests)
    concurrency = max(1, args.concurrency)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        semaphore = asyncio.Semaphore(concurrency)
        latencies: list[float] = []
        failures: list[int] = []

        start_time = time.perf_counter()

        async def run_single(index: int) -> None:
            async with semaphore:
                batch = [
                    encoded_images[(index * batch_size + offset) % len(encoded_images)]
                    for offset in range(batch_size)
                ]
                status, latency = await _issue_request(
                    client, args.url, batch, args.include_aligned
                )
                if status != httpx.codes.OK:
                    failures.append(status)
                else:
                    latencies.append(latency)

        await asyncio.gather(*(run_single(i) for i in range(total_requests)))

        total_time = time.perf_counter() - start_time

    if not latencies:
        print("No successful responses were recorded.")
        if failures:
            print(f"Failures: {len(failures)} responses, statuses={set(failures)}")
        return

    latencies.sort()
    avg_ms = (sum(latencies) / len(latencies)) * 1000.0
    p95_ms = _percentile(latencies, 95.0) * 1000.0
    throughput = len(latencies) / total_time if total_time else float("inf")

    print(f"Completed {len(latencies)} successful requests ({total_requests} attempted)")
    print(f"Average latency: {avg_ms:.2f} ms")
    print(f"P95 latency: {p95_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} req/s")
    if failures:
        print(f"{len(failures)} request(s) failed with statuses: {sorted(set(failures))}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_benchmark(args))


if __name__ == "__main__":
    main()
