#!/usr/bin/env python3
"""
AStarMaze_HW3.py — HW3-ready maze search visualizer

USAGE (examples)
---------------
# Problem 1 (Greedy vs A*, Manhattan, 4-way)
python AStarMaze_HW3.py --mode greedy --heuristic manhattan --neighbors 4 \
  --maze mazes/my_maze.txt --out images/greedy_p1.png

python AStarMaze_HW3.py --mode astar  --heuristic manhattan --neighbors 4 \
  --maze mazes/my_maze.txt --out images/astar_p1.png

# Problem 2 (Euclidean, 8-way diagonals, random neighbor order)
python AStarMaze_HW3.py --mode greedy --heuristic euclidean --neighbors 8 \
  --randomize --seed 7 --maze mazes/my_maze_diagonal.txt --out images/greedy_p2.png

python AStarMaze_HW3.py --mode astar  --heuristic euclidean --neighbors 8 \
  --randomize --seed 7 --maze mazes/my_maze_diagonal.txt --out images/astar_p2.png

# Problem 3 (Weighted A*, vary alpha/beta)
python AStarMaze_HW3.py --mode wastar --alpha 1 --beta 4 --heuristic euclidean --neighbors 8 \
  --randomize --seed 7 --maze mazes/my_maze_diagonal.txt --out images/wastar_a1_b4.png
"""

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
import tkinter as tk
from heapq import heappush, heappop

# ========= Heuristics =========
def h_manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def h_euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ========= Directions =========
ORTHO = [(1,0),(-1,0),(0,1),(0,-1)]
DIAGS = [(1,1),(1,-1),(-1,1),(-1,-1)]

def neighbor_dirs(mode="4", randomize=False):
    """Return a list of direction offsets; optionally shuffled each expansion."""
    dirs = list(ORTHO)
    if mode == "8":
        dirs += DIAGS
    if randomize:
        random.shuffle(dirs)
    return dirs

# ========= Maze load / helpers =========
@dataclass(frozen=True)
class Point:
    r: int
    c: int

def load_maze(path):
    """Reads a text file of 0/1 chars (spaces allowed). Returns (grid, rows, cols)."""
    grid = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = line.replace(" ", "")
            grid.append([1 if ch == "1" else 0 for ch in line])
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return grid, rows, cols

def default_start_goal(grid):
    """Start = first open in top row; Goal = last open in bottom row."""
    rows, cols = len(grid), len(grid[0])
    start = None
    goal = None
    for c in range(cols):             # top row left->right
        if grid[0][c] == 0:
            start = Point(0, c)
            break
    for c in range(cols-1, -1, -1):   # bottom row right->left
        if grid[rows-1][c] == 0:
            goal = Point(rows-1, c)
            break
    if not start or not goal:
        raise ValueError("Could not infer start/goal from maze; ensure top and bottom rows have at least one 0.")
    return start, goal

def in_bounds(p, rows, cols):
    return 0 <= p.r < rows and 0 <= p.c < cols

def passable(grid, p):
    return grid[p.r][p.c] == 0

def step_cost(a: Point, b: Point) -> float:
    """Cost: 1 for orthogonal, √2 for diagonal."""
    return math.hypot(b.r - a.r, b.c - a.c)

def diagonal_move(a: Point, b: Point) -> bool:
    return (a.r != b.r) and (a.c != b.c)

def blocks_corner(grid, a: Point, b: Point) -> bool:
    """Disallow cutting corners through walls for diagonal moves."""
    if not diagonal_move(a, b):
        return False
    mid1 = Point(a.r, b.c)
    mid2 = Point(b.r, a.c)
    return not passable(grid, mid1) or not passable(grid, mid2)

# ========= Priority function =========
def f_score(mode, g, h, alpha=1.0, beta=1.0):
    if mode == "greedy":
        return h
    elif mode == "wastar":
        return alpha * g + beta * h
    else:  # "astar"
        return g + h

# ========= Search =========
def search(grid, start: Point, goal: Point, mode: str, heuristic, dirs_fn, alpha=1.0, beta=1.0):
    rows, cols = len(grid), len(grid[0])

    g = defaultdict(lambda: float("inf"))
    g[start] = 0.0
    came_from = {}
    expanded = 0

    open_heap = []
    counter = 0  # tie-breaker to avoid comparing Point objects

    # initial push
    h0 = heuristic((start.r, start.c), (goal.r, goal.c))
    heappush(open_heap, (f_score(mode, 0.0, h0, alpha, beta), 0.0, counter, start))
    counter += 1
    closed = set()

    while open_heap:
        _, gcur, _, cur = heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1

        if cur == goal:
            # reconstruct path
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path, g[goal], expanded

        # neighbors (order may be randomized each expansion)
        for dr, dc in dirs_fn():
            nxt = Point(cur.r + dr, cur.c + dc)
            if not in_bounds(nxt, rows, cols) or not passable(grid, nxt):
                continue
            if blocks_corner(grid, cur, nxt):
                continue
            tentative = g[cur] + step_cost(cur, nxt)
            if tentative < g[nxt]:
                g[nxt] = tentative
                came_from[nxt] = cur
                hval = heuristic((nxt.r, nxt.c), (goal.r, goal.c))
                heappush(open_heap, (f_score(mode, tentative, hval, alpha, beta), tentative, counter, nxt))
                counter += 1

    return None, float("inf"), expanded  # no path

# ========= Visualization (ONLY final path) =========
def draw_and_optionally_save(grid, start, goal, path, cell=30, out_png=""):
    rows, cols = len(grid), len(grid[0])
    w = cols * cell
    h = rows * cell

    root = tk.Tk()
    root.title("A* Maze")
    canvas = tk.Canvas(root, width=w, height=h, bg="#f7f7f7", highlightthickness=0)
    canvas.pack()

    # Walls
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                x0, y0 = c*cell, r*cell
                x1, y1 = x0+cell, y0+cell
                canvas.create_rectangle(x0, y0, x1, y1, fill="#6b0f0f", outline="")

    # Path (draw as light-blue cells)
    if path:
        for p in path:
            x0, y0 = p.c*cell, p.r*cell
            x1, y1 = x0+cell, y0+cell
            canvas.create_rectangle(x0, y0, x1, y1, fill="#bfe4ff", outline="white")

    # Start & Goal overlays
    sx0, sy0 = start.c*cell, start.r*cell
    gx0, gy0 = goal.c*cell, goal.r*cell
    canvas.create_rectangle(sx0, sy0, sx0+cell, sy0+cell, fill="#0b2a8f", outline="")
    canvas.create_rectangle(gx0, gy0, gx0+cell, gy0+cell, fill="#2f8f0b", outline="")

    canvas.update()

    # Optional save to PNG (requires pillow)
    if out_png:
        try:
            from PIL import ImageGrab  # pillow
            x = canvas.winfo_rootx()
            y = canvas.winfo_rooty()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(out_png)
            print(f"[saved] {out_png}")
        except Exception as e:
            print(f"[warn] Could not save PNG automatically ({e}). Take a macOS screenshot instead (Shift+Cmd+4).")

    # Auto-close if saving; otherwise leave window up
    if out_png:
        root.after(800, root.destroy)
    root.mainloop()

# ========= CLI =========
def parse_point(s: str) -> Point:
    # format: "r,c"
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Point must be 'r,c'")
    return Point(int(parts[0]), int(parts[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["greedy","astar","wastar"], default="astar",
                    help="greedy: f=h; astar: f=g+h; wastar: f=alpha*g+beta*h")
    ap.add_argument("--heuristic", choices=["manhattan","euclidean"], default="manhattan")
    ap.add_argument("--neighbors", choices=["4","8"], default="4", help="4-way or 8-way moves")
    ap.add_argument("--randomize", action="store_true", help="randomize neighbor expansion order")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maze", required=True, help="path to 0/1 text maze file")
    ap.add_argument("--out", default="", help="optional PNG path to save canvas (requires pillow)")
    ap.add_argument("--start", type=parse_point, default=None, help="override start as 'r,c'")
    ap.add_argument("--goal",  type=parse_point, default=None, help="override goal as 'r,c'")
    ap.add_argument("--cell", type=int, default=30, help="cell size in pixels")
    args = ap.parse_args()

    if args.seed:
        random.seed(args.seed)

    grid, rows, cols = load_maze(args.maze)
    start, goal = (args.start, args.goal)
    if start is None or goal is None:
        s, g = default_start_goal(grid)
        start = start or s
        goal  = goal  or g

    heuristic = h_manhattan if args.heuristic == "manhattan" else h_euclidean

    # Build a callable returning (possibly randomized) directions for each expansion
    def dirs_fn():
        return neighbor_dirs(args.neighbors, args.randomize)

    t0 = time.time()
    path, cost, expanded = search(
        grid, start, goal, args.mode, heuristic, dirs_fn, alpha=args.alpha, beta=args.beta
    )
    dt = time.time() - t0

    if path is None:
        print("No path found.")
    else:
        print(f"PATH_LEN={len(path)}  PATH_COST={cost:.3f}  NODES_EXPANDED={expanded}  ELAPSED={dt*1000:.1f}ms")

    # Draw ONLY the shortest path, as required
    draw_and_optionally_save(grid, start, goal, path, cell=args.cell, out_png=args.out)

if __name__ == "__main__":
    main()
