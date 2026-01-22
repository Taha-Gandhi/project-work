"""
Fast heuristic solver for the "Gold Thief" / Traveling Thief-style problem.

NOTE:
- Kept filename/class name (FastGASolver) to minimize changes to the project structure.
- Replaced the expensive all-pairs shortest paths + GA with a fast, deterministic heuristic
  designed to run in << 1s even for n up to ~500.

Core idea:
- Start from the baseline (one city per trip).
- Greedily merge small groups of nearby cities into the same trip only when it strictly
  reduces total cost under the true edge-by-edge cost function.
- Special-case alpha == 0 (no weight penalty): then do a single tour (TSP-like) using a
  fast nearest-neighbor heuristic, because carrying gold is free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import networkx as nx


@dataclass
class _Trip:
    cities: List[int]  # excludes base (0)


class FastGASolver:
    """
    Drop-in replacement for the previous GA solver.
    Exposes the same API: FastGASolver(problem).solve() -> route list.
    """

    def __init__(self, problem):
        self.problem = problem
        self.G = problem.graph
        self.alpha = float(problem.alpha)
        self.beta = float(problem.beta)
        self.n = len(self.G.nodes)

        # Precompute base shortest paths once (cheap: one Dijkstra)
        self._base_paths: Dict[int, List[int]] = nx.single_source_dijkstra_path(self.G, 0, weight="dist")
        self._base_dists: Dict[int, float] = nx.single_source_dijkstra_path_length(self.G, 0, weight="dist")

        # Node positions for quick "nearest neighbor" candidate generation
        self._pos = nx.get_node_attributes(self.G, "pos")
        self._gold = {i: float(self.G.nodes[i]["gold"]) for i in self.G.nodes}

        # Cache for a few Dijkstra results to avoid repeating (bounded by demand)
        self._sp_path_cache: Dict[Tuple[int, int], List[int]] = {}

    # --------------------------- Public API ---------------------------

    def solve(self) -> List[Tuple[int, float]]:
        if self.n <= 1:
            return [(0, 0)]

        # alpha == 0 => cost is purely distance; best is a single tour
        if abs(self.alpha) < 1e-12:
            tour = self._nearest_neighbor_tour()
            return self._tour_to_route_single_trip(tour)

        # Otherwise: greedy merge improvements over baseline
        trips = self._baseline_trips()
        trips = self._improve_by_pair_merges(trips, k_candidates=12)
        trips = self._improve_by_triple_extension(trips, k_candidates=10)

        return self._trips_to_route(trips)

    # --------------------------- Baseline ---------------------------

    def _baseline_trips(self) -> List[_Trip]:
        # one city per trip
        return [_Trip([i]) for i in range(1, self.n)]

    # --------------------------- Cost helpers (true edge-by-edge) ---------------------------

    def _shortest_path(self, u: int, v: int) -> List[int]:
        key = (u, v)
        if key in self._sp_path_cache:
            return self._sp_path_cache[key]
        # Use Dijkstra path (sparse graph, n<=500 ok for limited calls)
        p = nx.dijkstra_path(self.G, u, v, weight="dist")
        self._sp_path_cache[key] = p
        return p

    def _path_cost(self, path: List[int], carried_weight: float) -> float:
        # Sum per edge using problem.cost([a,b], carried_weight)
        c = 0.0
        for a, b in zip(path, path[1:]):
            c += self.problem.cost([a, b], carried_weight)
        return c

    def _trip_cost(self, trip: _Trip) -> float:
        """
        Cost of: base -> city1 -> city2 -> ... -> base,
        collecting all gold in each city upon arrival.
        """
        w = 0.0
        total = 0.0
        prev = 0
        for city in trip.cities:
            path = self._base_paths[city] if prev == 0 else self._shortest_path(prev, city)
            total += self._path_cost(path, w)
            w += self._gold[city]
            prev = city
        # return to base
        if prev != 0:
            back_path = list(reversed(self._base_paths[prev]))  # undirected graph
            total += self._path_cost(back_path, w)
        return total

    # --------------------------- Improvements ---------------------------

    def _euclid(self, a: int, b: int) -> float:
        ax, ay = self._pos[a]
        bx, by = self._pos[b]
        dx = ax - bx
        dy = ay - by
        return math.hypot(dx, dy)

    def _nearest_candidates(self, city: int, remaining: set, k: int) -> List[int]:
        # pick k nearest by euclidean among remaining
        # (fast enough for n<=500)
        cand = [(self._euclid(city, j), j) for j in remaining if j != city]
        cand.sort(key=lambda x: x[0])
        return [j for _, j in cand[:k]]

    def _improve_by_pair_merges(self, trips: List[_Trip], k_candidates: int = 12) -> List[_Trip]:
        """
        Try to merge single-city trips into 2-city trips when beneficial.
        """
        remaining = set(range(1, self.n))
        trips_out: List[_Trip] = []
        used = set()

        # For reproducibility & speed: process cities in ascending base distance (closer first)
        order = sorted(list(remaining), key=lambda c: self._base_dists.get(c, 1e9))

        for a in order:
            if a in used:
                continue

            used.add(a)
            best_b = None
            best_delta = 0.0

            # candidates among not-used
            pool = remaining - used
            for b in self._nearest_candidates(a, pool, k_candidates):
                if b in used:
                    continue
                # baseline: two separate trips
                cost_sep = self._trip_cost(_Trip([a])) + self._trip_cost(_Trip([b]))
                # merged: try both orders and pick best
                cost_ab = min(self._trip_cost(_Trip([a, b])), self._trip_cost(_Trip([b, a])))
                delta = cost_sep - cost_ab
                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_b = b

            if best_b is not None:
                used.add(best_b)
                trips_out.append(_Trip([a, best_b]))
            else:
                trips_out.append(_Trip([a]))

        return trips_out

    def _improve_by_triple_extension(self, trips: List[_Trip], k_candidates: int = 10) -> List[_Trip]:
        """
        Try to extend some 2-city trips to 3 cities if beneficial.
        Keep this conservative for speed and robustness.
        """
        remaining = set(range(1, self.n))
        in_trip = set()
        for t in trips:
            for c in t.cities:
                in_trip.add(c)
        # already covers all, but we use it to avoid mistakes
        assert in_trip == remaining

        trips_out: List[_Trip] = []
        unused = set(remaining)

        for t in trips:
            for c in t.cities:
                if c in unused:
                    unused.remove(c)

        # Actually unused should be empty; extension works by taking from other trips.
        # We'll only attempt "steal one city" from a single-city trip into a 2-city trip if good.

        singles = [tr for tr in trips if len(tr.cities) == 1]
        single_set = set(tr.cities[0] for tr in singles)

        for t in trips:
            if len(t.cities) != 2:
                trips_out.append(t)
                continue

            a, b = t.cities
            # try taking a nearby single-city 'c' and make a 3-city trip, leaving that single removed
            best_c = None
            best_gain = 0.0

            # candidate singles near either a or b
            candidates = set(self._nearest_candidates(a, single_set, k_candidates) + self._nearest_candidates(b, single_set, k_candidates))
            for c in candidates:
                if c in (a, b):
                    continue
                # old cost: trip(a,b) + trip(c)
                old = self._trip_cost(t) + self._trip_cost(_Trip([c]))
                # new: best permutation of (a,b,c)
                perms = ([a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a])
                new = min(self._trip_cost(_Trip(list(p))) for p in perms)
                gain = old - new
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_c = c

            if best_c is not None and best_gain > 1e-6:
                trips_out.append(_Trip([a, b, best_c]))
                # Remove best_c by filtering it out later
                single_set.discard(best_c)
            else:
                trips_out.append(t)

        # Remove singles that were absorbed
        final_trips: List[_Trip] = []
        absorbed = set()
        for t in trips_out:
            if len(t.cities) == 3:
                absorbed.add(t.cities[2])  # the one we added (not perfect but ok)
        for t in trips_out:
            if len(t.cities) == 1 and t.cities[0] in absorbed:
                continue
            final_trips.append(t)
        return final_trips

    # --------------------------- Route formatting ---------------------------

    def _tour_to_route_single_trip(self, tour: List[int]) -> List[Tuple[int, float]]:
        route: List[Tuple[int, float]] = []
        for city in tour:
            route.append((city, self._gold[city]))
        route.append((0, 0))
        return route

    def _trips_to_route(self, trips: List[_Trip]) -> List[Tuple[int, float]]:
        route: List[Tuple[int, float]] = []
        for t in trips:
            for city in t.cities:
                route.append((city, self._gold[city]))
            route.append((0, 0))
        # Ensure ends at base
        if not route or route[-1] != (0, 0):
            route.append((0, 0))
        return route

    # --------------------------- alpha==0 tour builder ---------------------------

    def _nearest_neighbor_tour(self) -> List[int]:
        """
        Very fast nearest-neighbor tour based on Euclidean distance (not shortest-path),
        used only when alpha==0 (weight irrelevant). We still return a *feasible*
        route because feasibility is handled by evaluator via graph shortest paths.
        """
        unvisited = set(range(1, self.n))
        current = 0
        tour: List[int] = []
        while unvisited:
            # choose nearest euclidean
            nxt = min(unvisited, key=lambda j: self._euclid(current if current != 0 else 0, j))
            unvisited.remove(nxt)
            tour.append(nxt)
            current = nxt
        return tour
