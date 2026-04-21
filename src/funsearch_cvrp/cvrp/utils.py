from funsearch_cvrp.cvrp.core import CVRPInstance


import random


def generate_synthetic_benchmarks(seed: int = 2026, sizes: list[int] = [20, 50, 100]) -> list[CVRPInstance]:
    """
    Generate synthetic CVRP instances with random coordinates and demands.
    """

    rng = random.Random(seed)
    instances: list[CVRPInstance] = []

    for idx, n in enumerate(sizes, start=1):
        coords = [(50.0, 50.0)]
        demands = [0]
        for _ in range(n):
            coords.append((rng.uniform(0, 100), rng.uniform(0, 100)))
            demands.append(rng.randint(1, 10))

        capacity = max(25, int(sum(demands) / (n / 4.0)))
        instances.append(
            CVRPInstance(
                name=f"SYN-{idx:02d}-N{n}",
                capacity=capacity,
                demands=demands,
                coords=coords,
            )
        )

    return instances