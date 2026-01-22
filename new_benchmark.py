import time
import itertools
from pathlib import Path

from Problem import Problem
import s347289  # uses solution(p)

# ---- PARAMETERS ----
N = 50
ALPHAS = [0, 1, 2]
BETAS = [0.5, 1, 2]
DENSITIES = [0.25, 0.5, 1]
SEED = 42

OUTFILE = Path("results.md")


def safe_eval_cost(p: Problem, route):
    if hasattr(p, "evaluate_solution"):
        return p.evaluate_solution(route)
    if hasattr(p, "cost"):
        return p.cost(route)
    raise AttributeError("Problem has no evaluate_solution() or cost() method")


def baseline_route(p: Problem):
    r = [(0, 0)]
    for i in range(1, p.n):
        r.append((i, float(p.gold[i])))
        r.append((0, 0))
    return r


def main():
    combos = list(itertools.product(ALPHAS, BETAS, DENSITIES))
    total = len(combos)

    header = (
        "# Benchmark Results (n = 50)\n\n"
        "| n | alpha | beta | density | seed | baseline_cost | my_cost | difference | improvement_% | status | solve_time_s |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|\n"
    )
    OUTFILE.write_text(header, encoding="utf-8")

    start_time = time.time()

    for idx, (alpha, beta, density) in enumerate(combos, start=1):
        percent = (idx / total) * 100
        elapsed = time.time() - start_time
        print(
            f"[{idx}/{total}] {percent:.1f}% | "
            f"alpha={alpha}, beta={beta}, density={density} | "
            f"elapsed={elapsed:.1f}s"
        )

        p = Problem(
            num_cities=N,
            alpha=alpha,
            beta=beta,
            density=density,
            seed=SEED,
        )

        # baseline
        base_route = baseline_route(p)
        baseline_cost = safe_eval_cost(p, base_route)

        # your solution
        t0 = time.time()
        my_route = s347289.solution(p)
        solve_time = time.time() - t0

        my_cost = safe_eval_cost(p, my_route)

        diff = baseline_cost - my_cost
        improvement = (diff / baseline_cost) * 100 if baseline_cost != 0 else 0.0
        status = "BEATS" if my_cost < baseline_cost else "NOT"

        line = (
            f"| {N} | {alpha} | {beta} | {density} | {SEED} | "
            f"{baseline_cost:.6f} | {my_cost:.6f} | "
            f"{diff:.6f} | {improvement:.3f} | {status} | {solve_time:.3f} |\n"
        )

        with OUTFILE.open("a", encoding="utf-8") as f:
            f.write(line)

    print("Benchmark completed. Results saved to results.md")


if __name__ == "__main__":
    main()
