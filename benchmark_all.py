"""
Benchmark runner for s347289 solution.

Runs all combinations requested and writes a markdown table to results.md:
n in [20, 50, 100, 500]
alpha in [0, 1, 2]
beta in [0.5, 1, 2]
density in [0.25, 0.5, 1]

Columns:
n, alpha, beta, density, seed, baseline_cost, my_cost, difference, improvement_pct, status
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import math

import networkx as nx

from Problem import Problem
from s347289 import solution


def evaluate_route_cost(p: Problem, route: List[Tuple[int, float]]) -> float:
    """
    Compute true travel cost for the route using edge-by-edge cost on shortest paths.
    Uses caching of single-source shortest paths for speed.
    """
    G = p.graph
    gold = {i: float(G.nodes[i]["gold"]) for i in G.nodes}

    # cache shortest paths from each source node
    sp_cache: Dict[int, Dict[int, List[int]]] = {}

    def shortest_path(u: int, v: int) -> List[int]:
        if u not in sp_cache:
            sp_cache[u] = nx.single_source_dijkstra_path(G, u, weight="dist")
        return sp_cache[u][v]

    carried = 0.0
    total = 0.0
    prev_city = 0

    for city, g in route:
        # travel prev -> city with current carried
        if city != prev_city:
            path = shortest_path(prev_city, city)
            for a, b in zip(path, path[1:]):
                total += p.cost([a, b], carried)

        # arrive
        if city == 0:
            carried = 0.0
        else:
            # ensure full gold is collected (use graph gold)
            carried += gold[city]

        prev_city = city

    return total


def run():
    ns = [20, 50, 100, 500]
    alphas = [0, 1, 2]
    betas = [0.5, 1, 2]
    densities = [0.25, 0.5, 1]
    seed = 42

    rows = []
    start_all = time.time()

    for n in ns:
        for a in alphas:
            for b in betas:
                for d in densities:
                    p = Problem(num_cities=n, alpha=float(a), beta=float(b), density=float(d), seed=seed)
                    baseline = p.baseline()

                    t0 = time.time()
                    route = solution(p)
                    my_cost = evaluate_route_cost(p, route)
                    t1 = time.time()

                    diff = baseline - my_cost
                    imp = (diff / baseline * 100.0) if baseline != 0 else 0.0
                    status = "BEATS" if my_cost < baseline - 1e-9 else "NOT"

                    rows.append((n, a, b, d, seed, baseline, my_cost, diff, imp, status, (t1 - t0)))

    elapsed = time.time() - start_all

    # write markdown
    lines = []
    lines.append("# Results\n")
    lines.append(f"Total runtime: **{elapsed:.3f}s**\n")
    lines.append("| n | alpha | beta | density | seed | baseline_cost | my_cost | difference | improvement_% | status | solve_time_s |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for (n, a, b, d, seed, baseline, my_cost, diff, imp, status, st) in rows:
        lines.append(
            f"| {n} | {a} | {b} | {d} | {seed} | {baseline:.6f} | {my_cost:.6f} | {diff:.6f} | {imp:.3f} | {status} | {st:.4f} |"
        )

    with open("results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote results.md")


if __name__ == "__main__":
    run()
