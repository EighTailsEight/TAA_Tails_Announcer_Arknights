r"""左侧队友栏：四人生命值 + 「正在协防」三点指示（免 OCR）。

720p 实测（×1.5 得 1920 规范坐标）：
  头像列 x 20..85；生命数字白字在头像右下角徽章上，x 44..72
  四行数字基线 y 213..226 / 288..301 / 363..376 / 438..451
  行 1 是玩家自己（其数字与右上生命计数器一致）
  三点指示在头像右侧 x 86..114、行底 y 约 205..230 起的 18px 带内，
  三颗点轮流闪烁（常有一颗灰掉），所以按「亮像素数」判有无，不做单帧计数。

字形模板 avhp_<d> 存于 templates/avatar_hp.npz，由 build_avatar_hp.py 采收。
未知数字返回 None，不猜。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import digits as _digits

BASE = Path(__file__).resolve().parent
PACK = BASE / "templates" / "avatar_hp.npz"

ROWS = 4
HP_BOXES = [
    (66, 321, 108, 341),
    (66, 433, 108, 453),
    (66, 546, 108, 566),
    (66, 658, 108, 678),
]
# 三点与行底对齐：中心 x 145/160/176，中心 y 331+112.5*r（规范坐标）
DOT_XS = (145, 160, 176)
DOT_Y0 = 331.0
DOT_DY = 112.5
DOT_BOX = (-5, -4, 5, 4)      # 相对点中心的 10×8 采样框
DOT_ON_PX = 30                # 亮点实测 37–62；没有点时为 0
DOTS_NEEDED = 2               # 三颗点轮流闪烁，至少两颗亮才算
GLYPH = (16, 22)
MIN_SCORE = 0.60
MIN_MARGIN = 0.08

_cache = None


def _pack():
    global _cache
    if _cache is None:
        if not PACK.exists():
            _cache = {}
        else:
            z = np.load(PACK)
            _cache = {k: z[k] for k in z.files}
    return _cache


def _crop(rgb, box):
    from screen_detector import crop_canonical
    return crop_canonical(rgb, box)


def _norm(g: np.ndarray) -> np.ndarray:
    return np.asarray(
        Image.fromarray((g * 255).astype(np.uint8)).resize(GLYPH, Image.BILINEAR),
        dtype=np.float32) / 255.0


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def digit_mask(patch) -> np.ndarray:
    a = patch.astype(np.int16)
    l = a.mean(2)
    return (l > 160) & (a.max(2) - a.min(2) < 45)


def classify(glyph: np.ndarray) -> int | None:
    p = _pack()
    labels = [int(k.split("_")[1]) for k in p if k.startswith("avhp_")]
    if not labels:
        return None
    g = _norm(glyph)
    scores = sorted(((max(_ncc(g, t) for t in p[f"avhp_{d}"]), d) for d in labels), reverse=True)
    best, d = scores[0]
    margin = best - scores[1][0] if len(scores) > 1 else 1.0
    if best < MIN_SCORE or margin < MIN_MARGIN:
        return None
    return d


def _glyphs(mask: np.ndarray) -> list[np.ndarray]:
    """按列切字形，丢掉贴边笔画（裁剪框右侧 4px 的徽章边框）。"""
    on = mask.sum(0) > 0
    runs, st = [], None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((st, i))
            st = None
    if st is not None:
        runs.append((st, len(on)))
    out = []
    for x0, x1 in runs:
        if x0 == 0 or x1 == len(on):
            continue
        sub = mask[:, x0:x1]
        rows = np.nonzero(sub.sum(1) > 0)[0]
        h = int(rows.max() - rows.min() + 1)
        # '1' 的笔画只有 4–5px 宽；边框/图标残片高度不对
        if x1 - x0 >= 4 and 10 <= h <= 40:
            out.append(sub[rows.min():rows.max() + 1].astype(np.float32))
    return out


def read_row(rgb, row: int) -> int | None:
    """第 row 位队友（0=玩家）的生命值；读不出返回 None。"""
    patch = _crop(rgb, HP_BOXES[row])
    gs = _glyphs(digit_mask(patch))
    if not 1 <= len(gs) <= 2:
        return None
    out = []
    for g in gs:
        d = classify(g)
        if d is None:
            return None
        out.append(d)
    return int("".join(str(d) for d in out))


def read_hp(rgb) -> tuple:
    """四人生命值；任一读不出则该位为 None。"""
    return tuple(read_row(rgb, r) for r in range(ROWS))


def dot_score(rgb, row: int) -> int:
    """该行三颗协防点中亮着的颗数（0–3）。"""
    y = int(round(DOT_Y0 + DOT_DY * row))
    x1, y1, x2, y2 = DOT_BOX
    on = 0
    for x in DOT_XS:
        p = _crop(rgb, (x + x1, y + y1, x + x2, y + y2)).astype(np.int16)
        mn = p.min(2)
        if int(((mn > 120) & (p.max(2) - mn < 50)).sum()) >= DOT_ON_PX:
            on += 1
    return on


def dots(rgb) -> tuple:
    """四人谁旁边有协防三点：True/False 四元组。"""
    return tuple(dot_score(rgb, r) >= DOTS_NEEDED for r in range(ROWS))
