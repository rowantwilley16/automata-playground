import sys, time, random, math
import numpy as np
import pygame as pg

# ========= Config =========
GRID_ROWS   = 120
GRID_COLS   = 120
CELL_SIZE   = 8            # window pixels per cell (scaled)
TARGET_FPS  = 60
WRAP_WORLD  = True         # toroidal world (fastest)
RANDOM_P    = 0.25
ALIVE_RGB   = (31, 119, 180)  # "#1f77b4"
DEAD_RGB    = (17, 17, 17)    # "#111111"
BG_RGB      = (0, 0, 0)
SHOW_FPS    = True
GRID_COLOR  = (80, 80, 80)    # grid line color
GRID_ALPHA  = 90               # 0-255 transparency
# =========================

# --- Vectorized step (wrap/no-wrap) ---
def step_wrap(g: np.ndarray, out: np.ndarray):
    n = (
        np.roll(g,  1, 0) + np.roll(g, -1, 0) +
        np.roll(g,  1, 1) + np.roll(g, -1, 1) +
        np.roll(np.roll(g, 1, 0),  1, 1) +
        np.roll(np.roll(g, 1, 0), -1, 1) +
        np.roll(np.roll(g,-1, 0),  1, 1) +
        np.roll(np.roll(g,-1, 0), -1, 1)
    )
    out[:] = ((n == 3) | ((g == 1) & ((n == 2) | (n == 3)))).astype(np.uint8)

def step_nowrap(g: np.ndarray, out: np.ndarray):
    rows, cols = g.shape
    n = np.zeros_like(g, dtype=np.uint8)
    n[1:,   1:  ] += g[:-1, :-1]
    n[1:,   :   ] += g[:-1, :  ]
    n[1:,   :-1 ] += g[:-1, 1: ]
    n[:,    1:  ] += g[:,   :-1]
    n[:,    :-1 ] += g[:,   1: ]
    n[:-1,  1:  ] += g[1:,  :-1]
    n[:-1,  :   ] += g[1:,  :  ]
    n[:-1,  :-1 ] += g[1:,  1: ]
    out[:] = ((n == 3) | ((g == 1) & (n == 2)) | ((g == 1) & (n == 3))).astype(np.uint8)

class Life:
    def __init__(self, rows, cols, wrap=True):
        self.rows, self.cols = rows, cols
        self.wrap = wrap
        self.g      = np.zeros((rows, cols), dtype=np.uint8)
        self.next_g = np.zeros_like(self.g)
        self.step_fn = step_wrap if wrap else step_nowrap

    def randomize(self, p=RANDOM_P):
        self.g[:] = (np.random.random(self.g.shape) < p).astype(np.uint8)

    def clear(self):
        self.g.fill(0)

    def set_cell(self, r, c, val):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.g[r, c] = 1 if val else 0

    def get_cell(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(self.g[r, c])
        return 0

    def step(self):
        self.step_fn(self.g, self.next_g)
        self.g, self.next_g = self.next_g, self.g

# -------- Disco helpers --------
def hsv_to_rgb_np(h, s, v):
    i = np.floor(h*6).astype(np.int32)
    f = h*6 - i
    p = (v*(1-s))
    q = (v*(1-f*s))
    t = (v*(1-(1-f)*s))
    i_mod = (i % 6)

    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return (np.clip(r*255,0,255).astype(np.uint8),
            np.clip(g*255,0,255).astype(np.uint8),
            np.clip(b*255,0,255).astype(np.uint8))

class Disco:
    OFF=0; HUE=1; RAINBOW=2

class Renderer:
    def __init__(self, cols, rows, cell_size, alive_rgb, dead_rgb):
        self.cols, self.rows = cols, rows
        self.cell_size = cell_size
        self.window_size = (cols*cell_size, rows*cell_size)
        self.dead = np.array(dead_rgb, dtype=np.uint8)
        self.alive_rgb = np.array(alive_rgb, dtype=np.uint8)

        xv, yv = np.meshgrid(np.linspace(0,1,cols,endpoint=False),
                             np.linspace(0,1,rows,endpoint=False))
        base_h = (xv*2 + yv*3) % 1.0
        r,g,b = hsv_to_rgb_np(base_h, 1.0, 1.0)
        self.rainbow_base = np.dstack([r,g,b])  # (H,W,3) uint8

        self.small_surf = pg.Surface((cols, rows))
        self.colors = np.array([dead_rgb, alive_rgb], dtype=np.uint8)

        self.mode = Disco.OFF
        self._last_alive = self.alive_rgb.copy()

        # --- pre-rendered GRID SURFACE (fast) ---
        self.show_grid = True
        self.grid_surf = self._make_grid_surface(self.window_size, cell_size, GRID_COLOR, GRID_ALPHA)

    def _make_grid_surface(self, window_size, cell_size, color, alpha):
        w, h = window_size
        surf = pg.Surface((w, h), flags=pg.SRCALPHA)  # per-pixel alpha
        surf.fill((0,0,0,0))
        rgba = (*color, alpha)
        # vertical lines
        for x in range(0, w+1, cell_size):
            pg.draw.line(surf, rgba, (x, 0), (x, h))
        # horizontal lines
        for y in range(0, h+1, cell_size):
            pg.draw.line(surf, rgba, (0, y), (w, y))
        return surf

    def toggle_mode(self):
        self.mode = (self.mode + 1) % 3

    def toggle_grid(self):
        self.show_grid = not self.show_grid

    def _alive_color_for_frame(self):
        if self.mode == Disco.OFF:
            return self.alive_rgb
        elif self.mode == Disco.HUE:
            t = pg.time.get_ticks() * 0.00025
            h = (t % 1.0)
            r,g,b = hsv_to_rgb_np(np.array(h), 1.0, 1.0)
            return np.array([int(r), int(g), int(b)], dtype=np.uint8)
        else:
            return None

    def draw(self, screen, grid_u8):
        if self.mode in (Disco.OFF, Disco.HUE):
            alive = self._alive_color_for_frame()
            if alive is None:
                alive = self.alive_rgb
            if not np.array_equal(alive, self._last_alive):
                self.colors[1] = alive
                self._last_alive = alive
            rgb = self.colors[grid_u8]  # (H,W,3)
        else:
            t = pg.time.get_ticks() * 0.001
            shift = int((t*60) % self.cols)
            shifted = np.roll(self.rainbow_base, shift, axis=1)
            rgb = np.where(grid_u8[...,None] == 1, shifted, self.dead)

        pg.surfarray.blit_array(self.small_surf, rgb.swapaxes(0,1))
        frame = pg.transform.scale(self.small_surf, self.window_size)
        screen.blit(frame, (0,0))

        # draw the grid overlay (cheap blit)
        if self.show_grid:
            screen.blit(self.grid_surf, (0,0))

# ---------- Patterns (stamps) ----------
# Each pattern is a list of (dr, dc) offsets relative to (r, c)
GLIDER = [(0,1),(1,2),(2,0),(2,1),(2,2)]
# Pulsar (period-3 oscillator), 13x13 footprint centered around (r,c)
PULSAR = [
    (-6,-4),(-6,-3),(-6,-2),(-6,2),(-6,3),(-6,4),
    (-1,-4),(-1,-3),(-1,-2),(-1,2),(-1,3),(-1,4),
    (1,-4),(1,-3),(1,-2),(1,2),(1,3),(1,4),
    (6,-4),(6,-3),(6,-2),(6,2),(6,3),(6,4),
    (-4,-6),(-3,-6),(-2,-6),(2,-6),(3,-6),(4,-6),
    (-4,-1),(-3,-1),(-2,-1),(2,-1),(3,-1),(4,-1),
    (-4,1),(-3,1),(-2,1),(2,1),(3,1),(4,1),
    (-4,6),(-3,6),(-2,6),(2,6),(3,6),(4,6),
]
# Gosper Glider Gun (top-left anchor). Use at comfortable coords.
GOSPER_GUN = [
    (0,24),(1,22),(1,24),(2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
    (3,11),(3,15),(3,20),(3,21),(3,34),(3,35),
    (4,0),(4,1),(4,10),(4,16),(4,20),(4,21),
    (5,0),(5,1),(5,10),(5,14),(5,16),(5,17),(5,22),(5,24),
    (6,10),(6,16),(6,24),
    (7,11),(7,15),
    (8,12),(8,13)
]

def stamp(grid: np.ndarray, r: int, c: int, pattern, wrap: bool):
    rows, cols = grid.shape
    for dr, dc in pattern:
        rr = r + dr
        cc = c + dc
        if wrap:
            rr %= rows
            cc %= cols
            grid[rr, cc] = 1
        else:
            if 0 <= rr < rows and 0 <= cc < cols:
                grid[rr, cc] = 1

def mouse_to_rc(pos):
    x, y = pos
    c = x // CELL_SIZE
    r = y // CELL_SIZE
    r = max(0, min(GRID_ROWS-1, r))
    c = max(0, min(GRID_COLS-1, c))
    return r, c

def main():
    pg.init()
    clock = pg.time.Clock()
    font  = pg.font.SysFont(None, 22)

    life = Life(GRID_ROWS, GRID_COLS, wrap=WRAP_WORLD)
    renderer = Renderer(GRID_COLS, GRID_ROWS, CELL_SIZE, ALIVE_RGB, DEAD_RGB)

    screen = pg.display.set_mode(renderer.window_size)
    pg.display.set_caption("Conway's Game of Life — Pygame (Disco + Grid) | L-drag draw, R-drag erase, 1/2/3 stamps")

    running = False
    dragging = False
    erasing = False   # right-click or Shift = erase
    paint_val = 1
    last_rc = (-1, -1)
    fps_accum_time = 0.0
    fps_accum_frames = 0

    while True:
        # --- Input ---
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit(); sys.exit(0)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE or event.key == pg.K_q:
                    pg.quit(); sys.exit(0)
                elif event.key == pg.K_SPACE:
                    running = not running
                elif event.key == pg.K_s:
                    if not running: life.step()
                elif event.key == pg.K_r:
                    if not running: life.randomize()
                elif event.key == pg.K_c:
                    if not running: life.clear()
                elif event.key == pg.K_w:
                    life.wrap = not life.wrap
                    life.step_fn = step_wrap if life.wrap else step_nowrap
                elif event.key == pg.K_d:
                    renderer.toggle_mode()   # disco 🪩
                elif event.key == pg.K_g:
                    renderer.toggle_grid()   # grid toggle
                # --- Pattern stamps (paused): 1=glider, 2=pulsar, 3=gosper gun ---
                elif not running and event.key in (pg.K_1, pg.K_2, pg.K_3):
                    r, c = mouse_to_rc(pg.mouse.get_pos())
                    if event.key == pg.K_1:
                        stamp(life.g, r, c, GLIDER, life.wrap)
                    elif event.key == pg.K_2:
                        stamp(life.g, r, c, PULSAR, life.wrap)
                    elif event.key == pg.K_3:
                        stamp(life.g, r, c, GOSPER_GUN, life.wrap)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:  # left: draw
                    dragging = True
                    erasing = pg.key.get_mods() & pg.KMOD_SHIFT  # Shift+drag = erase
                    r, c = mouse_to_rc(pg.mouse.get_pos())
                    # if shift held, force erase; else toggle mode by initial cell
                    paint_val = 0 if (erasing or life.get_cell(r, c)) else 1
                    last_rc = (-1, -1)
                    life.set_cell(r, c, paint_val)
                elif event.button == 3:  # right: erase drag
                    dragging = True
                    erasing = True
                    r, c = mouse_to_rc(pg.mouse.get_pos())
                    paint_val = 0
                    last_rc = (-1, -1)
                    life.set_cell(r, c, paint_val)
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    dragging = False
                    erasing = False
                    last_rc = (-1, -1)
            elif event.type == pg.MOUSEMOTION and dragging and not running:
                r, c = mouse_to_rc(event.pos)
                if (r, c) != last_rc:
                    life.set_cell(r, c, paint_val)
                    last_rc = (r, c)

        # --- Update ---
        if running:
            life.step()

        # --- Render ---
        renderer.draw(screen, life.g)

        # On-screen FPS & status
        if SHOW_FPS:
            fps_accum_time += clock.get_time() / 1000.0
            fps_accum_frames += 1
            if fps_accum_time >= 1.0:
                fps_value = fps_accum_frames / fps_accum_time
                pg.display.set_caption(
                    f"Life — Pygame | {'RUN' if running else 'PAUSE'} | "
                    f"{'WRAP' if life.wrap else 'CLAMP'} | "
                    f"Mode: {['OFF','HUE','RAINBOW'][renderer.mode]} | "
                    f"Grid: {'ON' if renderer.show_grid else 'OFF'} | "
                    f"FPS: {fps_value:.1f} | "
                    f"Tips: L-drag draw, R-drag erase, Shift+drag erase, 1/2/3 stamps"
                )
                fps_accum_time = 0.0
                fps_accum_frames = 0

            text = font.render(
                f"{'RUN' if running else 'PAUSE'} | "
                f"{'WRAP' if life.wrap else 'CLAMP'} | "
                f"Mode: {['OFF','HUE','RAINBOW'][renderer.mode]} | "
                f"Grid: {'ON' if renderer.show_grid else 'OFF'} | "
                f"{GRID_COLS}x{GRID_ROWS}@{CELL_SIZE}px | "
                f"Target {TARGET_FPS} FPS",
                True, (230, 230, 230)
            )
            screen.blit(text, (8, 8))

        pg.display.flip()
        clock.tick(TARGET_FPS)

if __name__ == "__main__":
    main()
