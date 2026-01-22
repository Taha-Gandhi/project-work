"""
Gold Thief / Traveling Thief-like Problem Solution
Student ID: s123456

Requirements satisfied:
- Provides a `solution(p)` function returning:
  [(0, 0), (c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
- When executed as a script, prints parameters and the solution path.
- Fast runtime (designed to be << 1s for typical sizes).
"""

from __future__ import annotations

import argparse
from typing import List, Tuple
import networkx as nx

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


def calculate_route_cost(problem: Problem, route: List[Tuple[int, float]]) -> float:
    """Calculate the total cost of a route"""
    total_cost = 0
    current_weight = 0
    prev_city = 0
    
    for city, gold in route:
        # Calculate cost from prev to current (before resetting weight)
        if prev_city != city:
            try:
                path = nx.shortest_path(problem.graph, prev_city, city, weight='dist')
                for i in range(len(path) - 1):
                    total_cost += problem.cost([path[i], path[i+1]], current_weight)
            except:
                pass
        
        if city == 0:
            current_weight = 0
        else:
            current_weight += gold
        
        prev_city = city
    
    return total_cost


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

    # Calculate costs
    baseline_cost = p.baseline()
    my_cost = calculate_route_cost(p, route)
    
    # Calculate improvement
    if baseline_cost > 0:
        improvement = ((baseline_cost - my_cost) / baseline_cost) * 100
    else:
        improvement = 0

    # Generate visualization
    generate_route_visualization(p, route, my_cost, baseline_cost, improvement)

    # REQUIRED OUTPUT FORMAT
    print(f"num_cities={args.n}")
    print(f"alpha={p.alpha}")
    print(f"beta={p.beta}")
    print(f"density={args.density}")
    print()
    print(f"baseline_cost = {baseline_cost:.2f}")
    print(f"my_cost = {my_cost:.2f}")
    print(f"improvement% = {improvement:.2f}%")
    print()
    print("Solution path:")
    # Clean up np.float64 from output
    clean_route = [(int(c), float(g)) for c, g in route]
    print(clean_route)


def generate_route_visualization(problem: Problem, route: List[Tuple[int, float]], 
                                 my_cost: float, baseline_cost: float, improvement: float):
    """Generate and save route visualization graph"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Get positions
        pos = nx.get_node_attributes(problem.graph, 'pos')
        
        # Draw all edges lightly
        nx.draw_networkx_edges(problem.graph, pos, edge_color='lightgray',
                               width=0.5, alpha=0.2, ax=ax)
        
        # Parse route into trips
        trips = []
        current_trip = []
        for city, gold in route:
            if city == 0:
                if current_trip:
                    trips.append(current_trip)
                    current_trip = []
            else:
                current_trip.append(city)
        
        # Draw each trip with different color
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(trips), 1)))
        
        for trip_idx, trip in enumerate(trips):
            color = colors[trip_idx % len(colors)]
            
            # Full path: 0 -> cities -> 0
            full_path = [0] + trip + [0]
            
            for i in range(len(full_path) - 1):
                start = full_path[i]
                end = full_path[i + 1]
                
                try:
                    path = nx.shortest_path(problem.graph, start, end, weight='dist')
                    edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
                    nx.draw_networkx_edges(problem.graph, pos, edgelist=edges,
                                         edge_color=[color], width=4, alpha=0.7,
                                         arrows=True, arrowsize=15, ax=ax,
                                         connectionstyle='arc3,rad=0.05')
                except:
                    pass
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in problem.graph.nodes():
            if node == 0:
                node_colors.append('#FF0000')
                node_sizes.append(1000)
            else:
                node_colors.append('#87CEEB')
                gold = problem.graph.nodes[node]['gold']
                node_sizes.append(max(300, gold * 0.4))
        
        nx.draw_networkx_nodes(problem.graph, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.95, ax=ax,
                              edgecolors='black', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(problem.graph, pos, font_size=11,
                               font_weight='bold', font_color='black', ax=ax)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#FF0000', label='Base City (0)'),
            mpatches.Patch(color='#87CEEB', label='Cities with Gold'),
            mpatches.Patch(color='gray', label=f'Total Trips: {len(trips)}')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
                 framealpha=0.9)
        
        # Title
        title = f'Traveling Thief Solution\n'
        title += f'Baseline: {baseline_cost:.2f} | Our Cost: {my_cost:.2f} | Improvement: {improvement:.2f}%'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('route_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n✓ Graph saved as 'route_visualization.png'")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠ Graph generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    _main()