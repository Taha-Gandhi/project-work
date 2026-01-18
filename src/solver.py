"""src/solver.py

Computational Intelligence – Project Work

This implementation is intentionally *well-commented* so you can explain it.

Course connections (from the slides you uploaded):
- Problem classification: this is a hard combinatorial optimization problem.
- Searching for a state: we represent a candidate solution as a *state* (a sequence
  of trips, each trip is an ordered list of cities to visit before returning to base).
- Local search / Simulated Annealing: we improve the state using neighbourhood moves.
- Diversity: we use random neighbourhood moves + occasional perturbations to escape
  local minima.

Problem recap (informal):
- City 0 is the base.
- Each city i>0 has a fixed amount of gold gold[i].
- We move along graph edges; each edge has attribute 'dist'.
- If we traverse an edge of length d while carrying weight W, the move cost is:

    cost = d + (alpha * d * W) ** beta

  where alpha >= 0 and beta >= 0.
- When we visit a city for the *first* time, we collect ALL its gold.
  (No partial pickup in this version, per your latest requirement.)
- When we return to base 0, the carried weight resets to 0 (we unload).

Baseline required by you (NOT the professor's baseline() implementation):
- Repeatedly choose the nearest not-yet-collected city (by shortest-path distance from base).
- Go 0 -> that city -> 0.
- Collect all gold from that city.

Our algorithm improves over that baseline because:
- We can visit multiple cities in a single trip, while still collecting ALL gold from each.
- The ordering within a trip tries to visit farther cities earlier (when we are light),
  and closer ones later (when we are heavier).
- We run a small simulated annealing to improve the trip partitioning and ordering.

Outputs:
- For the grader: `OptimizedGoldSolver.solution()` returns a list of tuples:

    [(city_id, gold_taken_here), ... , (0, 0.0)]

  The path is expanded node-by-node so every move follows a real graph edge.
  Gold is non-zero only the first time we reach each city i>0.

- For local runs: you can also call plot_solution() and write_step_report().
"""

from dataclasses import dataclass
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# All-Pairs Shortest Paths (APSP)
# -----------------------------------------------------------------------------

@dataclass
class APSP:
    """Precomputed shortest-path data.

    dist[u][v]     shortest distance from u to v
    next_hop[u][v] first node to go to when moving from u to v along a shortest path
    """

    dist: List[List[float]]
    next_hop: List[List[Optional[int]]]


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

class OptimizedGoldSolver:
    """Solve a given Problem instance.

    Key idea: evaluate many candidate solutions quickly using precomputed shortest paths.

    State representation (what SA optimizes):
      trips = [ [c1, c2, ...], [d1, d2, ...], ... ]
    Each inner list is an ordered list of *cities* (nodes != 0) to visit in that trip.
    Each trip always starts at base 0 and ends at base 0.

    Constraints:
    - Every city 1..N-1 must appear exactly once across all trips (collect all gold once).
    """

    def __init__(self, problem, time_limit_s: float = 2.0, seed: int = 42):
        self.p = problem
        self.G: nx.Graph = problem.graph
        self.n = self.G.number_of_nodes()

        # Enforce alpha,beta >= 0
        if self.p.alpha < 0 or self.p.beta < 0:
            raise ValueError("alpha and beta must be >= 0")

        # City gold (0 has gold=0 by construction)
        self.gold: List[float] = [float(self.G.nodes[i]["gold"]) for i in range(self.n)]

        self.time_limit_s = float(time_limit_s)
        self.rng = random.Random(seed)

        # Precompute shortest paths once
        self.apsp = self._build_apsp()

        # Best solution found
        self.best_trips: List[List[int]] = [[1]] if self.n > 1 else [[]]
        self.best_cost: float = float("inf")

        # Run the search immediately
        self._search()

    # ------------------------------------------------------------------
    # APSP builder (robust across NetworkX versions)
    # ------------------------------------------------------------------

    def _build_apsp(self) -> APSP:
        dist = [[math.inf] * self.n for _ in range(self.n)]
        next_hop: List[List[Optional[int]]] = [[None] * self.n for _ in range(self.n)]

        # For each source, run Dijkstra
        for s in range(self.n):
            d, paths = nx.single_source_dijkstra(self.G, source=s, weight="dist")

            for t, dt in d.items():
                dist[s][t] = float(dt)

            # paths[t] is a list like [s, ..., t]
            for t, path in paths.items():
                if t == s:
                    continue
                if len(path) >= 2:
                    next_hop[s][t] = path[1]

        # Graph must be connected
        for v in range(self.n):
            if dist[0][v] == math.inf:
                raise ValueError("Graph appears disconnected (unexpected).")

        return APSP(dist=dist, next_hop=next_hop)

    def _expand_shortest_path(self, u: int, v: int) -> List[int]:
        """Reconstruct node-by-node shortest path u -> v using next_hop."""
        if u == v:
            return [u]
        if self.apsp.next_hop[u][v] is None:
            # Should not happen for connected graph
            return [u, v]

        path = [u]
        cur = u
        while cur != v:
            cur = self.apsp.next_hop[cur][v]
            if cur is None:
                return [u, v]
            path.append(cur)
        return path

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    def _move_cost(self, d: float, carry: float) -> float:
        """Edge traversal cost under current carried weight."""
        return d + (self.p.alpha * d * carry) ** self.p.beta

    def _trips_cost(self, trips: List[List[int]]) -> float:
        """Compute total cost of a trip plan using shortest-path distances (fast).

        We simulate each trip:
          carry starts at 0 at base
          visit cities in the trip order (collect all gold)
          return to base (carry resets)

        NOTE: This does NOT expand node-by-node; it uses APSP distances.
        """
        total = 0.0

        for trip in trips:
            carry = 0.0
            cur = 0

            for city in trip:
                # move cur -> city
                d = self.apsp.dist[cur][city]
                total += self._move_cost(d, carry)

                # arrive, collect ALL gold
                carry += self.gold[city]
                cur = city

            # return to base
            if cur != 0:
                d = self.apsp.dist[cur][0]
                total += self._move_cost(d, carry)

        return float(total)

    # ------------------------------------------------------------------
    # Baseline (as you requested)
    # ------------------------------------------------------------------

    def baseline_nearest_first_trips(self) -> List[List[int]]:
        """Baseline: each trip visits exactly 1 city.

        Cities are ordered by shortest-path distance from base (nearest first).
        """
        if self.n <= 1:
            return [[]]
        cities = list(range(1, self.n))
        cities.sort(key=lambda c: self.apsp.dist[0][c])
        return [[c] for c in cities]

    # ------------------------------------------------------------------
    # Constructive heuristic (better starting point)
    # ------------------------------------------------------------------

    def _greedy_multi_trip(self) -> List[List[int]]:
        """Build an initial solution that usually beats the baseline.

        Heuristic intuition (works well especially when alpha,beta in [0,1]):
        - Going far while light is good.
        - Therefore start each trip from one of the farthest remaining cities.
        - Then add a few more nearby cities to the same trip if that seems cheaper
          than doing them as separate trips.

        This produces a reasonable trip partition quickly.
        """
        if self.n <= 1:
            return [[]]

        remaining = set(range(1, self.n))
        # Consider farthest cities first (distance from base)
        far_order = sorted(list(remaining), key=lambda c: self.apsp.dist[0][c], reverse=True)

        trips: List[List[int]] = []

        # Hyper-parameter: max cities per trip (keeps solution stable and fast)
        # If alpha is small or beta < 1, penalty grows slowly -> allow larger trips.
        if self.p.alpha == 0 or self.p.beta == 0:
            max_trip = min(25, len(far_order))
        else:
            # typical setting for alpha,beta ~ 1
            max_trip = 6 if self.p.beta >= 0.8 else 10

        for seed_city in far_order:
            if seed_city not in remaining:
                continue

            # Start a new trip with a far city
            trip = [seed_city]
            remaining.remove(seed_city)

            # Keep adding cities greedily while beneficial
            while remaining and len(trip) < max_trip:
                cur = trip[-1]

                # candidate: pick next city that is close to current
                # (we also slightly prefer cities that are not too far from base)
                best = None
                best_score = float("inf")

                for c in remaining:
                    score = self.apsp.dist[cur][c] + 0.25 * self.apsp.dist[c][0]
                    if score < best_score:
                        best_score = score
                        best = c

                if best is None:
                    break

                # Decide if adding this city is beneficial vs making it a separate trip
                # We compare incremental costs approximately.
                #
                # Current trip cost (using APSP) vs if we end trip now and do best alone.
                cur_cost = self._trips_cost([trip])
                alone_cost = self._trips_cost([[best]])

                new_trip = trip + [best]
                new_cost = self._trips_cost([new_trip])

                if new_cost <= cur_cost + alone_cost:
                    trip.append(best)
                    remaining.remove(best)
                else:
                    break

            trips.append(trip)

        return trips

    # ------------------------------------------------------------------
    # Neighbourhood moves (for SA)
    # ------------------------------------------------------------------

    def _copy_trips(self, trips: List[List[int]]) -> List[List[int]]:
        return [t[:] for t in trips]

    def _neighbor(self, trips: List[List[int]]) -> List[List[int]]:
        """Create a neighbour solution by applying one random move.

        Moves:
        - swap two cities (possibly across trips)
        - move one city to another position/trip
        - reverse a subsequence inside a trip (2-opt-ish)
        - merge two small trips or split a large trip
        """
        T = self._copy_trips(trips)

        # Remove empty trips (keep representation clean)
        T = [t for t in T if t]
        if not T:
            return [[]]

        move_type = self.rng.random()

        # Helper: get random trip indices safely
        def rand_trip_index() -> int:
            return self.rng.randrange(len(T))

        if move_type < 0.35:
            # Swap two cities
            i = rand_trip_index()
            j = rand_trip_index()
            if not T[i] or not T[j]:
                return T
            a = self.rng.randrange(len(T[i]))
            b = self.rng.randrange(len(T[j]))
            T[i][a], T[j][b] = T[j][b], T[i][a]

        elif move_type < 0.70:
            # Relocate one city
            i = rand_trip_index()
            if not T[i]:
                return T
            a = self.rng.randrange(len(T[i]))
            city = T[i].pop(a)
            if not T[i]:
                T.pop(i)

            # Insert into another trip or create a new trip
            if T and self.rng.random() < 0.85:
                j = self.rng.randrange(len(T))
                pos = self.rng.randrange(len(T[j]) + 1)
                T[j].insert(pos, city)
            else:
                T.append([city])

        elif move_type < 0.85:
            # Reverse a subsequence inside a trip (small 2-opt)
            i = rand_trip_index()
            if len(T[i]) >= 3:
                a, b = sorted(self.rng.sample(range(len(T[i])), 2))
                T[i][a : b + 1] = reversed(T[i][a : b + 1])

        else:
            # Merge or split
            if len(T) >= 2 and self.rng.random() < 0.5:
                # Merge two trips
                i, j = self.rng.sample(range(len(T)), 2)
                T[i].extend(T[j])
                T.pop(j)
            else:
                # Split one trip
                i = rand_trip_index()
                if len(T[i]) >= 4:
                    cut = self.rng.randrange(1, len(T[i]) - 1)
                    left = T[i][:cut]
                    right = T[i][cut:]
                    T[i] = left
                    T.append(right)

        # Remove empty trips again
        T = [t for t in T if t]
        return T

    # ------------------------------------------------------------------
    # Simulated Annealing loop
    # ------------------------------------------------------------------

    def _simulated_annealing(self, start: List[List[int]], deadline: float) -> Tuple[List[List[int]], float]:
        """Simulated Annealing optimizing trip partition + ordering."""
        cur = self._copy_trips(start)
        cur_cost = self._trips_cost(cur)

        best = self._copy_trips(cur)
        best_cost = cur_cost

        # SA parameters
        T = 1.0
        cooling = 0.9992

        while time.time() < deadline:
            nxt = self._neighbor(cur)
            nxt_cost = self._trips_cost(nxt)

            delta = nxt_cost - cur_cost
            if delta <= 0:
                cur = nxt
                cur_cost = nxt_cost
                if cur_cost < best_cost:
                    best = self._copy_trips(cur)
                    best_cost = cur_cost
            else:
                # accept worse with probability
                p = math.exp(-delta / max(T, 1e-12))
                if self.rng.random() < p:
                    cur = nxt
                    cur_cost = nxt_cost

            T *= cooling
            if T < 1e-6:
                T = 1e-6

        return best, float(best_cost)

    # ------------------------------------------------------------------
    # Search driver: guarantee improvement if found, otherwise baseline fallback
    # ------------------------------------------------------------------

    def _search(self) -> None:
        end_time = time.time() + self.time_limit_s

        # Baseline (nearest-first, 1 city per trip)
        baseline_trips = self.baseline_nearest_first_trips()
        baseline_cost = self._trips_cost(baseline_trips)

        # Stronger starting point
        greedy_trips = self._greedy_multi_trip()
        greedy_cost = self._trips_cost(greedy_trips)

        # Keep the best seen so far
        self.best_trips = greedy_trips if greedy_cost < baseline_cost else baseline_trips
        self.best_cost = min(greedy_cost, baseline_cost)

        # Run SA in small chunks (multi-start feel without heavy runtime)
        # We also occasionally restart from the current best.
        while time.time() < end_time:
            chunk_deadline = min(end_time, time.time() + 0.25)
            start = self.best_trips

            cand_trips, cand_cost = self._simulated_annealing(start, chunk_deadline)

            if cand_cost < self.best_cost:
                self.best_trips = cand_trips
                self.best_cost = cand_cost

        # If (rarely) we could not improve, we still return the best between greedy and baseline.

    # ------------------------------------------------------------------
    # Convert trips -> expanded required output list
    # ------------------------------------------------------------------

    def solution(self) -> List[Tuple[int, float]]:
        """Return the final solution in required list-of-tuples format.

        We expand each APSP segment into actual node steps.
        Gold is taken only once at first arrival to each city.
        """
        if self.n <= 1:
            return [(0, 0.0)]

        visited = set([0])

        expanded_nodes: List[int] = [0]
        expanded_take: List[float] = [0.0]

        for trip in self.best_trips:
            cur = 0

            for city in trip:
                path = self._expand_shortest_path(cur, city)
                # add nodes (skip first because it's cur)
                for node in path[1:]:
                    expanded_nodes.append(node)
                    expanded_take.append(0.0)

                # collect ALL gold if first time
                if city not in visited:
                    expanded_take[-1] = float(self.gold[city])
                    visited.add(city)

                cur = city

            # return to base
            path = self._expand_shortest_path(cur, 0)
            for node in path[1:]:
                expanded_nodes.append(node)
                expanded_take.append(0.0)

        # ensure ends at base
        if expanded_nodes[-1] != 0:
            expanded_nodes.append(0)
            expanded_take.append(0.0)

        return list(zip(expanded_nodes, expanded_take))

    # ------------------------------------------------------------------
    # Local-run utilities: step report + plotting
    # ------------------------------------------------------------------

    def evaluate_expanded_solution_cost(self, sol: List[Tuple[int, float]]) -> float:
        """Compute cost from the expanded (node-by-node) output list."""
        nodes = [c for c, _ in sol]
        take = [g for _, g in sol]

        carry = 0.0
        total = 0.0

        for a, b, take_b in zip(nodes, nodes[1:], take[1:]):
            d = self.apsp.dist[a][b]
            total += self._move_cost(d, carry)

            if b == 0:
                carry = 0.0
            else:
                carry += float(take_b)

        return float(total)

    def write_step_report(self, sol: List[Tuple[int, float]], filepath: str = "solution_steps.txt") -> None:
        """Write a detailed, line-by-line report (tab-separated) for debugging/explaining."""
        nodes = [c for c, _ in sol]
        take = [g for _, g in sol]

        carried = 0.0
        cumulative = 0.0
        delivered_total = 0.0

        headers = [
            "step",
            "from",
            "to",
            "edge_dist",
            "city_gold_total",
            "taken_here",
            "carry_before",
            "carry_after",
            "move_cost",
            "cumulative_cost",
            "delivered_total_so_far",
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Detailed solution trace (one move per line)\n")
            f.write(f"alpha={self.p.alpha}, beta={self.p.beta}\n")
            f.write("\t".join(headers) + "\n")

            for step, (a, b, take_b) in enumerate(zip(nodes, nodes[1:], take[1:]), start=1):
                d = self.apsp.dist[a][b]
                move_cost = self._move_cost(d, carried)
                carry_before = carried

                # arrival updates
                if b == 0:
                    delivered_total += carried
                    carried = 0.0
                    city_total = 0.0
                else:
                    city_total = self.gold[b]
                    carried += float(take_b)

                carry_after = carried
                cumulative += move_cost

                row = [
                    str(step),
                    str(a),
                    str(b),
                    f"{d:.6f}",
                    f"{city_total:.6f}",
                    f"{float(take_b):.6f}",
                    f"{carry_before:.6f}",
                    f"{carry_after:.6f}",
                    f"{move_cost:.6f}",
                    f"{cumulative:.6f}",
                    f"{delivered_total:.6f}",
                ]
                f.write("\t".join(row) + "\n")

    def plot_solution(self, sol: List[Tuple[int, float]], show: bool = True, savepath: Optional[str] = "solution_plot.png") -> None:
        """Plot the graph and highlight the expanded path."""
        nodes = [c for c, _ in sol]
        edges = list(zip(nodes, nodes[1:]))

        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self.G, "pos")
        if not pos:
            pos = nx.spring_layout(self.G, seed=1)

        # Size proportional to gold (base fixed)
        sizes = [200] + [max(40.0, self.gold[i]) for i in range(1, self.n)]
        colors = ["red"] + ["lightblue"] * (self.n - 1)

        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_color=colors,
            node_size=sizes,
            edge_color="lightgray",
            width=1.0,
        )

        nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=edges,
            width=3.0,
            edge_color="black",
            alpha=0.85,
        )

        plt.title("Graph + highlighted solution path")

        if savepath:
            plt.savefig(savepath, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()
