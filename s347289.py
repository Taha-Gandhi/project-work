"""
Gold Thief / Traveling Thief-like Problem Solution
Student ID: s347289

Requirements satisfied:
- Provides a `solution(p)` function returning:
  [(0, 0), (c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
- Prints (0,0) whenever the traveler returns to base to unload gold.
- Fast runtime (<< 1 second for typical sizes).
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from Problem import Problem
from src.fast_ga import FastGASolver


def solution(p: Problem) -> List[Tuple[int, float]]:
    """
    Compute a route that visits all cities, collects all gold,
    and unloads at the base (0) whenever needed.

    Returned format:
    [(0,0), (c1,g1), ..., (0,0), (cK,gK), ..., (0,0)]
    """
    solver = FastGASolver(p)
    route = solver.solve()

    # ---- STRICT FORMAT ENFORCEMENT ----

    # ensure start at base
    if not route or route[0] != (0, 0):
        route = [(0, 0)] + route

    # ensure end at base
    if route[-1] != (0, 0):
        route.append((0, 0))

    # remove duplicate consecutive (0,0)
    cleaned = [route[0]]
    for item in route[1:]:
        if item == (0, 0) and cleaned[-1] == (0, 0):
            continue
        cleaned.append(item)

    return cleaned


def _main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    p = Problem(
        num_cities=args.n,
        alpha=args.alpha,
        beta=args.beta,
        density=args.density,
        seed=args.seed,
    )

    route = solution(p)

    print(f"num_cities={p.n}, alpha={p.alpha}, beta={p.beta}, density={p.density}")
    print("Solution path :")
    print(route)


if __name__ == "__main__":
    _main()
