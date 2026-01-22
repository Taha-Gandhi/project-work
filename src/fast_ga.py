"""
Fast Genetic Algorithm Solver for Traveling Thief Problem
Based on proven strategies from successful implementations
"""

import random
import numpy as np
import networkx as nx
from typing import List, Tuple


class FastGASolver:
    def __init__(self, problem):
        self.problem = problem
        self.graph = problem.graph
        self.alpha = problem.alpha
        self.beta = problem.beta
        self.n_cities = len(self.graph.nodes)
        
        # Precompute distances
        self.dist_matrix = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='dist'))
        self.paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight='dist'))
        
        # Cache gold values
        self.gold = np.array([self.graph.nodes[i]['gold'] for i in range(self.n_cities)])
        
        # Get cities list (excluding base)
        self.cities = [i for i in range(1, self.n_cities)]
        
    def evaluate_tour(self, tour: List[int]) -> float:
        """
        Evaluate total cost of a tour with optimal split strategy
        """
        if not tour:
            return float('inf')
            
        # For beta > 1: hub-spoke is better (individual trips)
        if self.beta > 1.0:
            return self._eval_hub_spoke(tour)
        
        # For beta <= 1: accumulation strategy with smart splitting
        return self._eval_with_split(tour)
    
    def _eval_hub_spoke(self, tour: List[int]) -> float:
        """Hub-spoke: go to each city and return immediately"""
        total_cost = 0
        for city in tour:
            gold = self.gold[city]
            # Go there (no weight)
            try:
                path_there = self.paths[0][city]
                cost_there = sum(self.problem.cost([path_there[i], path_there[i+1]], 0) 
                               for i in range(len(path_there)-1))
                
                # Return (with gold)
                path_back = self.paths[city][0]
                cost_back = sum(self.problem.cost([path_back[i], path_back[i+1]], gold) 
                              for i in range(len(path_back)-1))
                
                total_cost += cost_there + cost_back
            except:
                return float('inf')
        
        return total_cost
    
    def _eval_with_split(self, tour: List[int]) -> float:
        """
        Evaluate tour with dynamic splitting for beta <= 1
        Uses dynamic programming to find optimal split points
        """
        n = len(tour)
        if n == 0:
            return 0
        
        # DP array: dp[i] = min cost to collect gold from cities tour[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Try all possible last trip lengths
        max_trip_len = min(n, 15)  # Limit trip length for efficiency
        
        for i in range(1, n + 1):
            for trip_len in range(1, min(i, max_trip_len) + 1):
                start_idx = i - trip_len
                trip_cities = tour[start_idx:i]
                
                trip_cost = self._calc_trip_cost(trip_cities)
                dp[i] = min(dp[i], dp[start_idx] + trip_cost)
        
        return dp[n]
    
    def _calc_trip_cost(self, cities: List[int]) -> float:
        """Calculate cost of a single trip: 0 -> cities -> 0"""
        if not cities:
            return 0
        
        total_cost = 0
        current_weight = 0
        current_pos = 0
        
        # Visit each city
        for city in cities:
            try:
                path = self.paths[current_pos][city]
                for i in range(len(path) - 1):
                    total_cost += self.problem.cost([path[i], path[i+1]], current_weight)
                
                current_weight += self.gold[city]
                current_pos = city
            except:
                return float('inf')
        
        # Return to base
        try:
            path = self.paths[current_pos][0]
            for i in range(len(path) - 1):
                total_cost += self.problem.cost([path[i], path[i+1]], current_weight)
        except:
            return float('inf')
        
        return total_cost
    
    def nearest_neighbor(self, start=None) -> List[int]:
        """Greedy nearest neighbor heuristic"""
        if start is None:
            start = random.choice(self.cities)
        
        tour = [start]
        remaining = set(self.cities) - {start}
        current = start
        
        while remaining:
            nearest = min(remaining, key=lambda c: self.dist_matrix[current][c])
            tour.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return tour
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        pos = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                if pos >= size:
                    pos = 0
                child[pos] = city
                pos += 1
        
        return child
    
    def mutate(self, tour: List[int]) -> List[int]:
        """Mutation: swap or reverse segment"""
        tour = tour.copy()
        
        if random.random() < 0.5:
            # Swap mutation
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        else:
            # Reverse segment
            i, j = sorted(random.sample(range(len(tour)), 2))
            tour[i:j+1] = reversed(tour[i:j+1])
        
        return tour
    
    def local_search_2opt(self, tour: List[int], max_iter=50) -> List[int]:
        """2-opt local search"""
        improved = True
        iterations = 0
        best_tour = tour.copy()
        best_cost = self.evaluate_tour(best_tour)
        
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            for i in range(len(best_tour) - 1):
                for j in range(i + 2, len(best_tour)):
                    new_tour = best_tour.copy()
                    new_tour[i+1:j+1] = reversed(new_tour[i+1:j+1])
                    
                    new_cost = self.evaluate_tour(new_tour)
                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_tour
    
    def solve(self) -> List[Tuple[int, float]]:
        """Main GA solver"""
        print(f"Solving with GA (n={self.n_cities}, α={self.alpha}, β={self.beta})...")
        
        # Parameters
        pop_size = min(100, self.n_cities * 2)
        generations = min(200, self.n_cities * 3)
        elite_size = max(5, pop_size // 10)
        mutation_rate = 0.3
        
        # Initialize population
        population = []
        
        # 50% nearest neighbor from different starts
        for _ in range(pop_size // 2):
            population.append(self.nearest_neighbor())
        
        # 50% random
        for _ in range(pop_size - len(population)):
            tour = self.cities.copy()
            random.shuffle(tour)
            population.append(tour)
        
        # Evaluate
        fitness = [self.evaluate_tour(ind) for ind in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_cost = fitness[best_idx]
        
        print(f"Initial best cost: {best_cost:.2f}")
        
        # Evolution
        for gen in range(generations):
            new_population = []
            
            # Elitism
            sorted_idx = np.argsort(fitness)
            for i in range(elite_size):
                new_population.append(population[sorted_idx[i]])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                p1 = population[min(random.sample(range(pop_size), 3), key=lambda i: fitness[i])]
                p2 = population[min(random.sample(range(pop_size), 3), key=lambda i: fitness[i])]
                
                # Crossover
                child = self.order_crossover(p1, p2)
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            # Local search on best
            if gen % 10 == 0:
                new_population[0] = self.local_search_2opt(new_population[0], max_iter=30)
            
            population = new_population
            fitness = [self.evaluate_tour(ind) for ind in population]
            
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_cost:
                best_cost = fitness[curr_best_idx]
                best_solution = population[curr_best_idx]
                print(f"Gen {gen}: New best = {best_cost:.2f}")
        
        # Final 2-opt
        print("Final refinement...")
        best_solution = self.local_search_2opt(best_solution, max_iter=100)
        
        # Convert to required format
        return self._convert_to_path(best_solution)
    
    def _convert_to_path(self, tour: List[int]) -> List[Tuple[int, float]]:
        """Convert tour to required path format"""
        if self.beta > 1.0:
            # Hub-spoke: each city individually
            path = []
            for city in tour:
                path.append((city, self.gold[city]))
                path.append((0, 0))
            return path
        
        # For beta <= 1: use optimal splitting
        n = len(tour)
        dp = [float('inf')] * (n + 1)
        parent = [-1] * (n + 1)
        dp[0] = 0
        
        max_trip_len = min(n, 15)
        
        for i in range(1, n + 1):
            for trip_len in range(1, min(i, max_trip_len) + 1):
                start_idx = i - trip_len
                trip_cities = tour[start_idx:i]
                trip_cost = self._calc_trip_cost(trip_cities)
                
                if dp[start_idx] + trip_cost < dp[i]:
                    dp[i] = dp[start_idx] + trip_cost
                    parent[i] = start_idx
        
        # Reconstruct trips
        trips = []
        i = n
        while i > 0:
            start = parent[i]
            trips.append(tour[start:i])
            i = start
        
        trips.reverse()
        
        # Build path
        path = []
        for trip in trips:
            for city in trip:
                path.append((city, self.gold[city]))
            path.append((0, 0))
        
        return path