# Traveling Thief Problem - Optimized Solution

## Strategy

This solution uses an **Adaptive Genetic Algorithm** that selects the optimal strategy based on problem parameters:

### For β > 1 (High Weight Penalty):
- **Hub-Spoke Strategy**: Visit each city individually and return immediately
- Minimizes weight carried at any time
- Best when weight penalties are severe

### For β ≤ 1 (Low Weight Penalty):
- **Accumulation with Dynamic Splitting**: Collect from multiple cities per trip
- Uses dynamic programming to find optimal split points
- Balances distance savings vs weight penalties

## Algorithm Components

1. **Population Initialization**:
   - 50% Nearest Neighbor heuristic (different starting points)
   - 50% Random permutations
   
2. **Genetic Operators**:
   - Order Crossover (OX) for recombination
   - Swap and reverse-segment mutations
   - Tournament selection (size 3)

3. **Local Search**:
   - 2-opt optimization on elite solutions
   - Applied every 10 generations + final refinement

4. **Fitness Evaluation**:
   - Hub-spoke cost calculation for β > 1
   - DP-based optimal splitting for β ≤ 1

## Main Files

- `s123456.py` - Main solution file
- `Problem.py` - Problem definition
- `src/fast_ga.py` - Genetic algorithm implementation
- `README.md` - This file
- `base_requirements.txt` - Dependencies

## Usage

```python
python s347289.py --n 50 --alpha 1 --beta 0.5 --density 0.5 --seed 42
```

Author : Taha Yusuf Gandhi