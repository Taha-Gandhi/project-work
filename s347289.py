# Rename this file to s<YOUR_STUDENT_ID>.py before submitting.

from __future__ import annotations

from typing import List, Tuple

from Problem import Problem
from src.solver import OptimizedGoldSolver


def solution(p: Problem) -> List[Tuple[int, float]]:
    """Entry point required by the grader."""
    solver = OptimizedGoldSolver(p, time_limit_s=1.8, seed=42)
    return solver.solution()


if __name__ == '__main__':
    # Simple local smoke test (not used by the grader)
    p = Problem(60, density=0.2, alpha=1.0, beta=1.0, seed=42)
    solver = OptimizedGoldSolver(p, time_limit_s=2.0, seed=42)
    sol = solver.solution()
    print('SOLUTION:')
    print(sol)
    baseline_cost = solver._trips_cost(solver.baseline_nearest_first_trips())
    print(f'Baseline nearest-first cost: {baseline_cost:.6f}')
    print(f'Our best cost:              {solver.best_cost:.6f}')
    if baseline_cost > 0:
        print(f'Improvement: {(baseline_cost - solver.best_cost)/baseline_cost*100:.2f}%')
    solver.write_step_report(sol, 'solution_steps.txt')
    solver.plot_solution(sol, savepath='solution_plot.png', show=False)
    print('Wrote solution_steps.txt and solution_plot.png')
