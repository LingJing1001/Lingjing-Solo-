"""泡壁内 ACTION6 点击扫描 —— 面积律：只在清晰区 / 显著对象上点。

钱学森转义：迷雾区不浪费点击；泡壁与对象质心优先展开高精度交互。
坐标约定与官方一致：x=列, y=行，范围 [0,63]。
"""
from __future__ import annotations

from collections import Counter, deque
from typing import List, Optional, Tuple

import numpy as np

from ..core import SoloConfig, Logger, clamp


def _guess_bg(grid: np.ndarray) -> int:
    flat = grid.flatten().astype(np.int64)
    return int(Counter(flat.tolist()).most_common(1)[0][0])


class BubbleClickPlanner:
    """在 BubbleWall 清晰区内生成并消费点击队列。"""

    def __init__(self, cfg: SoloConfig, logger: Logger = None):
        self.cfg = cfg
        self.log = logger or Logger()
        self.tried: set[Tuple[int, int]] = set()
        self.queue: deque[Tuple[int, int]] = deque()
        self.stall = 0                  # 连续无像素变化步数
        self.last_xy: Optional[Tuple[int, int]] = None
        self.clicks_done = 0
        self.hits = 0                   # 点击后产生变化的次数
        self.consec_clicks = 0          # 连续点击次数（防点死）
        self.click_noops = 0            # 点击无效果计数
        self.cooldown = 0               # 点击冷却（步）

    def reset(self):
        self.tried.clear()
        self.queue.clear()
        self.stall = 0
        self.last_xy = None
        self.clicks_done = 0
        self.hits = 0
        self.consec_clicks = 0
        self.click_noops = 0
        self.cooldown = 0

    def observe_effect(self, delta_pixels: int, progressed: bool, was_click: bool):
        if self.cooldown > 0:
            self.cooldown -= 1
        if delta_pixels <= 0 and not progressed:
            self.stall += 1
            if was_click:
                self.click_noops += 1
                # 连续点空 → 进入冷却，改回方向键
                if self.click_noops >= 3:
                    self.cooldown = self.cfg.click_cooldown
                    self.consec_clicks = 0
                    self.queue.clear()
        else:
            self.stall = 0
            if was_click and delta_pixels > 0:
                self.hits += 1
                self.click_noops = 0

    def should_click(self, valid_actions, force: bool = False) -> bool:
        if "ACTION6" not in valid_actions:
            return False
        if not (self.cfg.enable_mouse or self.cfg.enable_click_sweep):
            return False
        if self.cooldown > 0:
            return False
        # 连续点击上限：避免 vc33 类关卡被点穿 GAME_OVER
        if self.consec_clicks >= self.cfg.click_max_consecutive:
            return False
        if force and self.stall >= 2:
            return True
        # 仅当卡住时点击；有队列不等于每步都点
        if self.stall >= self.cfg.click_stall_threshold:
            return True
        # 早期少量探针点击（每 click_probe_every 步一次）
        if (
            self.clicks_done < self.cfg.click_warmup
            and self.stall >= 1
            and (self.clicks_done == 0 or self.stall % max(1, self.cfg.click_probe_every) == 0)
        ):
            return True
        return False

    def refill(self, grid: np.ndarray, bubble=None, objects=None, phi=None):
        """按面积律填充候选点：对象质心 → 泡壁网格 → Φ 峰 → 稀有色采样。"""
        if grid is None:
            return
        g = np.asarray(grid)
        H, W = g.shape
        candidates: List[Tuple[int, int]] = []

        def add(x, y):
            xi = int(clamp(int(x), 0, min(63, W - 1)))
            yi = int(clamp(int(y), 0, min(63, H - 1)))
            pt = (xi, yi)
            if pt not in self.tried:
                candidates.append(pt)

        # 1) 感知对象质心（泡壁内高精度结果）
        for obj in objects or []:
            if not obj.pixels:
                continue
            ys = [p[0] for p in obj.pixels]
            xs = [p[1] for p in obj.pixels]
            add(sum(xs) / len(xs), sum(ys) / len(ys))
            # 包围盒角点
            x0, y0, x1, y1 = obj.bbox
            add(x0, y0)
            add(x1, y1)
            add((x0 + x1) / 2, (y0 + y1) / 2)

        # 2) 泡壁清晰区稀疏网格
        rects = list(bubble.clear_rects) if bubble and bubble.clear_rects else []
        if not rects:
            rects = [(0, 0, H - 1, W - 1)]
        step = max(2, self.cfg.click_grid_step)
        for y0, x0, y1, x1 in rects:
            for y in range(y0, y1 + 1, step):
                for x in range(x0, x1 + 1, step):
                    add(x, y)

        # 3) Φ 密度峰映射回像素
        if phi is not None and getattr(phi, "blocks", None) is not None:
            by, bx = phi.peak_block
            bs = max(1, min(H, W) // max(1, phi.blocks.shape[0]))
            add(bx * bs + bs / 2, by * bs + bs / 2)

        # 4) 稀有色采样（限制在首个泡壁矩形内）
        bg = _guess_bg(g)
        hist = Counter(g.flatten().tolist())
        rare = [c for c, _ in sorted(hist.items(), key=lambda kv: kv[1]) if c != bg][:6]
        y0, x0, y1, x1 = rects[0]
        for color in rare:
            sub = g[y0:y1 + 1, x0:x1 + 1]
            local = np.argwhere(sub == color)
            if len(local) == 0:
                local = np.argwhere(g == color)
                offset = (0, 0)
            else:
                offset = (y0, x0)
            if len(local) == 0:
                continue
            stride = max(1, len(local) // 5)
            for i in range(0, len(local), stride):
                yy, xx = int(local[i][0]) + offset[0], int(local[i][1]) + offset[1]
                add(xx, yy)

        # 去重保序入队
        seen = set(self.queue)
        for pt in candidates:
            if pt not in seen and pt not in self.tried:
                self.queue.append(pt)
                seen.add(pt)
                if len(self.queue) >= self.cfg.click_queue_max:
                    break
        self.log.log("Click", f"refill q={len(self.queue)} tried={len(self.tried)}")

    def next_xy(self, grid=None, bubble=None, objects=None, phi=None) -> Optional[Tuple[int, int]]:
        if len(self.queue) < 3:
            self.refill(grid, bubble=bubble, objects=objects, phi=phi)
        while self.queue:
            pt = self.queue.popleft()
            if pt not in self.tried:
                self.tried.add(pt)
                self.last_xy = pt
                self.clicks_done += 1
                self.consec_clicks += 1
                return pt
        # 兜底：泡壁中心
        if bubble and bubble.clear_rects:
            y0, x0, y1, x1 = bubble.clear_rects[0]
            pt = (int((x0 + x1) / 2), int((y0 + y1) / 2))
        else:
            pt = (32, 32)
        if pt not in self.tried:
            self.tried.add(pt)
            self.last_xy = pt
            self.clicks_done += 1
            self.consec_clicks += 1
            return pt
        return None

    def note_simple_action(self):
        """方向键等非点击动作：重置连续点击计数。"""
        self.consec_clicks = 0
