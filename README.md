# Sokoban Solver — make.fun bounty `sokoban-solver-01-run-20261121`

[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Biose-dot/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Biose-dot/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver for the [make.fun bounty](https://make.fun/bounty/sokoban-solver-01-run-20261121): one program answers 5 instances in a single run, writing the player's moves that push every box onto a goal in as few pushes as possible.

## How it works

The generated levels are a one-square-wide hallway with independent rooms off it — one door each, boxes never leave their room, and a level's push-optimum P\* is the sum of its rooms' optima. So the solver:

1. **Decomposes** each level into hallway + rooms (connected components of floor off the hallway row).
2. **Solves each room push-optimally** with A\* over pushes: state = sorted box cells + canonical player cell (the smallest cell of the player's reachable region), heuristic = minimum-cost assignment of boxes to goals over precomputed grid distances (admissible, so the first goal popped is push-optimal). Pruning: static dead squares, corner and wall-run deadlocks, and 2×2 freeze deadlocks.
3. **Stitches** the room solutions together with shortest player walks through the hallway, tracking the boxes' current positions so a walk never pushes a box by accident.
4. **Replays** every move string on the level before writing it; a replay that fails writes an empty solution for that instance.

All search budgets are counts of expansions, never the wall clock, so two runs write the same bytes. The search runs on flat typed arrays with numerically packed state keys — a whole 5-instance set solves in about half a second.

## Results

| Set | Score |
| --- | --- |
| Practice set (kit) | **1,000,000 / 1,000,000** — all 5 instances solved in exactly P\* pushes (13, 16, 29, 47, 60) |
| 10 extra sets drawn from the kit's own generator | **1,000,000** on every set |
| House baseline plain-astar 1.0.0 (practice set) | 200,000 (solves only i001) |

## Running

Node 22, no dependencies, no install step:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

In the grader it runs as `node /bot/main.ts`, reading `/input/<id>/level.txt` and writing `/output/<id>/solution.txt` for each of the 5 instances.

## License

MIT — see [LICENSE](LICENSE).
