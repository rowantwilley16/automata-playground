"""
Conway's Game of Life — minimal, interactive (Tkinter only)

Controls:
  • Left click        : toggle a cell (alive/dead)
  • Space             : start / pause
  • S (or s)          : step one generation
  • R (or r)          : randomize board
  • C (or c)          : clear board
  • Q (or q) / Escape : quit

No external dependencies.
"""

import random
import tkinter as tk

# ======= Config =======
CELL_SIZE    = 14       # pixels per cell
GRID_ROWS    = 40
GRID_COLS    = 60
TICK_MS      = 80       # ms between generations while running
ALIVE_COLOR  = "#1f77b4"
DEAD_COLOR   = "#1b1b1b"
GRID_COLOR   = "#2b2b2b"
BG_COLOR     = "#101010"
# ======================

class GameOfLife:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.grid = [[0]*cols for _ in range(rows)]

    def clear(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = 0

    def randomize(self, p=0.25):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = 1 if random.random() < p else 0

    def toggle(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r][c] ^= 1

    def _neighbors(self, r, c):
        n = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    n += self.grid[rr][cc]
        return n

    def step(self):
        nxt = [[0]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                alive = self.grid[r][c] == 1
                k = self._neighbors(r, c)
                # Conway's rules
                if alive and (k == 2 or k == 3):
                    nxt[r][c] = 1
                elif not alive and k == 3:
                    nxt[r][c] = 1
                else:
                    nxt[r][c] = 0
        self.grid = nxt


class LifeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conway's Game of Life")
        self.root.configure(bg=BG_COLOR)

        self.life = GameOfLife(GRID_ROWS, GRID_COLS)
        w = GRID_COLS * CELL_SIZE
        h = GRID_ROWS * CELL_SIZE

        self.canvas = tk.Canvas(
            root, width=w, height=h, bg=BG_COLOR,
            highlightthickness=0
        )
        self.canvas.pack()

        # Pre-create rectangles for fast redraw
        self.rects = [
            [
                self.canvas.create_rectangle(
                    c*CELL_SIZE, r*CELL_SIZE,
                    (c+1)*CELL_SIZE, (r+1)*CELL_SIZE,
                    outline=GRID_COLOR, fill=DEAD_COLOR
                )
                for c in range(GRID_COLS)
            ]
            for r in range(GRID_ROWS)
        ]

        # State
        self.running = False
        self._after_id = None

        # Bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<space>", self.on_space)
        self.root.bind("s", self.on_step); self.root.bind("S", self.on_step)
        self.root.bind("r", self.on_random); self.root.bind("R", self.on_random)
        self.root.bind("c", self.on_clear); self.root.bind("C", self.on_clear)
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("q", lambda e: self.root.quit())
        self.root.bind("Q", lambda e: self.root.quit())

        # First draw
        self.redraw()

    def on_click(self, event):
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE
        self.life.toggle(r, c)
        self._color_cell(r, c)

    def on_space(self, _):
        if self.running:
            self.pause()
        else:
            self.run()

    def on_step(self, _):
        if not self.running:
            self.life.step()
            self.redraw()

    def on_random(self, _):
        if not self.running:
            self.life.randomize()
            self.redraw()

    def on_clear(self, _):
        if not self.running:
            self.life.clear()
            self.redraw()

    def _tick(self):
        self.life.step()
        self.redraw()
        if self.running:
            self._after_id = self.root.after(TICK_MS, self._tick)

    def run(self):
        if not self.running:
            self.running = True
            self._tick()

    def pause(self):
        self.running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _color_cell(self, r, c):
        fill = ALIVE_COLOR if self.life.grid[r][c] else DEAD_COLOR
        self.canvas.itemconfig(self.rects[r][c], fill=fill)

    def redraw(self):
        # Minimal redraw: recolor all; still fast enough for moderate grids
        for r in range(GRID_ROWS):
            row = self.life.grid[r]
            for c in range(GRID_COLS):
                self.canvas.itemconfig(
                    self.rects[r][c],
                    fill=(ALIVE_COLOR if row[c] else DEAD_COLOR)
                )

if __name__ == "__main__":
    root = tk.Tk()
    app = LifeApp(root)
    root.mainloop()
