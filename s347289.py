"""
Gold Thief / Traveling Thief-like Problem Solution
Student ID: s347289

Requirements satisfied:
- Provides a `solution(p)` function returning:
  [(0, 0), (c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
- When executed as a script, prints parameters and the solution path.
- Fast runtime (designed to be << 1s for typical sizes).
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from Problem import Problem
from src.fast_ga import FastGASolver


def solution(p: Problem) -> List[Tuple[int, float]]:
    """
    Compute a route that visits all cities, collects all their gold, and returns to base (0)
    one or more times.

    Returned route format:
        [(0, 0), (c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]

    Notes:
    - City 0 is the base.
    - Whenever (0, 0) appears, the carried gold is unloaded.
    - For each city i>0, the returned gold amount equals the full gold available at that city.
    """
    solver = FastGASolver(p)
    route = solver.solve()

    # --- STRICT FORMAT ENFORCEMENT ---

    # ensure start at base
    if not route or route[0] != (0, 0):
        route = [(0, 0)] + route

    # ensure end at base
    if route[-1] != (0, 0):
        route.append((0, 0))

    return route


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

    # REQUIRED OUTPUT FORMAT
    print(f"num_cities={p.num_cities} \nalpha={p.alpha} \nbeta={p.beta} \ndensity={p.density} \n")
    print("Solution path :")
    print(route)


if __name__ == "__main__":
    _main() 
