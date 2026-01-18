# Computational Intelligence – Project Work (Gold Routing)

This repository contains a **baseline** and an **improved** solver for the project
based on Giovanni Squillero's `Problem` class.

## Problem recap
- Cities are nodes in a connected undirected graph.
- Node `0` is the **base**.
- Each city `i > 0` has a positive amount of gold `gold[i]`.
- Each edge `(u,v)` has a Euclidean distance `dist`.
- If you traverse an edge of length `d` while carrying weight `W` (total gold currently in the bag),
  the cost is:

```
C = d + (alpha * d * W)^beta
```

with **alpha ≥ 0** and **beta ≥ 0** (in the course and in your tests typically `0..1`,
very often `alpha = 1`, `beta = 1`).

The goal is to collect all gold and return to base while **minimizing total cost**.

## Baseline required in this assignment
Your requirement for the baseline is:
1. Starting at base `0`, pick the **nearest not-yet-collected** city (nearest by shortest-path distance in the graph).
2. Travel `0 -> city -> 0`.
3. Collect **all** gold in that city.
4. Repeat until all cities are collected.

This baseline is implemented inside `src/solver.py` as `baseline_nearest_one_city_trips(...)`.

## Improved solver (what we submit)
The submission entry point is `s<student_id>.py` with:

```python
def solution(p: Problem) -> list[tuple[int, float]]:
    ...
```

### Key idea
The baseline pays a lot because it **returns to base after every single city**.
When `beta > 0`, carrying weight matters, but returning after every city wastes distance.

The improved algorithm uses **multi-city trips**:
- Start at base (weight 0)
- Visit several cities in a good order
- Collect ALL gold the first time we visit each city
- Return to base to unload

### How we search for a good multi-trip plan (course topics)
We model a solution as a **state**:
- A state is a list of trips
- Each trip is a list of city ids (without 0)

Example state:
```
[[12, 3, 9], [7, 5], [1, 8, 2]]
```
meaning:
- Trip 1: 0 -> 12 -> 3 -> 9 -> 0
- Trip 2: 0 -> 7 -> 5 -> 0
- Trip 3: 0 -> 1 -> 8 -> 2 -> 0

We use two steps:

1) **Constructive greedy initialization**
- Sort cities by distance from base (farthest-first) so we travel far while still light.
- Build trips by inserting cities where they increase cost the least.
- Close a trip when adding another city would be worse than starting a new trip.

2) **Local Search / Simulated Annealing**
We improve the state by applying small neighbourhood moves:
- swap two cities (possibly across trips)
- relocate a city to another position or another trip
- split a trip into two trips
- merge two trips

Acceptance rule (Simulated Annealing):
- Always accept improvements
- Sometimes accept a worse move with probability `exp(-Δ / T)` to escape local minima
- Gradually decrease `T` (cooling)

This approach is directly aligned with the slide topics:
- state representation
- neighbourhood operators
- local search
- simulated annealing

### Guarantee: never worse than baseline
At the end we **compute**:
- baseline cost (nearest-first, one city per trip)
- improved cost (multi-trip)

We return the better one.
So the returned solution is always **>= 0% improvement** over the baseline on that instance.

> Note: If costs are exactly equal (rare), improvement is 0%. In practice the multi-trip
> heuristic and SA almost always produce a strictly lower cost.

## Output format
The solver returns a list:

```
[(node_0, gold_taken_0), (node_1, gold_taken_1), ..., (0, 0.0)]
```
- Nodes include intermediate nodes from shortest paths, so every step is along a real edge.
- `gold_taken_here` is non-zero only at the first arrival to a city (we take all gold there).

## Local run (plot + report)
Run:

```bash
python s347289.py
```

It will generate:
- `solution_plot.png`: the graph with the chosen path highlighted
- `solution_steps.txt`: a step-by-step trace (tab-separated), including:
  step number, from->to, edge distance, carry before/after, edge cost, cumulative cost

## Requirements
Install dependencies from:

```bash
pip install -r base_requirements.txt
```

## File overview
- `Problem.py` – professor-provided class (unchanged)
- `s347289.py` – submission entry point (`solution(p)`)
- `src/solver.py` – baseline + improved solver + plotting/report
- `base_requirements.txt` – python dependencies

