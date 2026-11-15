r"""顶部 HUD 数字识别：战斗倒计时 / 首领进度 n/15（免 OCR）。

两套字形模板分开存于 templates/hud_digits.npz：
  clock_<d>：右上倒计时（青色 7 段字，≤10s 变橙），规范下字形约 30×68
  hpnum_<d>：顶部中央首领进度分子（暗橙字，右对齐在斜杠前），字形约 8×18
  hpden_<d>：同一进度条的分母（白字 15）

读数只做「显示值 → int」，语义判断（阈值、只报一次）留给 bridge。
未知数字返回 None，不猜。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
PACK = BASE / "templates" / "hud_digits.npz"

# 1920 规范坐标；时钟框右缘避开倒计时竖条，左缘留够 3 位数
CLOCK_BOX = (1690, 10, 1808, 94)
CLOCK_GLYPH = (12, 28)
CLOCK_THR = 120

COUNTER_NUM_BOX = (706, 16, 746, 64)
COUNTER_DEN_BOX = (750, 16, 800, 64)
COUNTER_GLYPH = (10, 18)

# 左上「回合 N」白底数字；字形随位数伸缩，按整框灰度模板比对。
# 最终攻势期间该框被 n/15 红条取代，读数返回 None。
ROUND_BOX = (750, 34, 790, 56)
ROUND_SHAPE = (22, 40)

MIN_SCORE = 0.55
MIN_MARGIN = 0.06
ROUND_MIN_SCORE = 0.85
ROUND_MIN_MARGIN = 0.06

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


def _norm(g: np.ndarray, shape) -> np.ndarray:
    return np.asarray(
        Image.fromarray((g * 255).astype(np.uint8)).resize(shape, Image.BILINEAR),
        dtype=np.float32) / 255.0


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _segment(mask: np.ndarray, min_w: int, max_h: int) -> list[np.ndarray]:
    from digits import segment
    return segment(mask, min_w=min_w, max_h=max_h)


def _classify(glyph: np.ndarray, slot: str, shape) -> int | None:
    p = _pack()
    labels = [int(k.split("_")[1]) for k in p if k.startswith(slot + "_")]
    if not labels:
        return None
    g = _norm(glyph, shape)
    scores = sorted(((max(_ncc(g, t) for t in p[f"{slot}_{d}"]), d) for d in labels), reverse=True)
    best, d = scores[0]
    margin = best - scores[1][0] if len(scores) > 1 else 1.0
    if best < MIN_SCORE or margin < MIN_MARGIN:
        return None
    return d


def _clock_mask(patch) -> np.ndarray:
    return patch.max(2) > CLOCK_THR


def _num_mask(patch) -> np.ndarray:
    a = patch.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (r > 125) & (r - g > 50) & (r - b > 45) & (g < 100) & (b < 100)


def _den_mask(patch) -> np.ndarray:
    a = patch.astype(np.int16)
    return (a[:, :, 0] > 150) & (a[:, :, 1] > 150) & (a[:, :, 2] > 150)


def read_clock(rgb) -> int | None:
    """右上倒计时秒数；读不出返回 None。"""
    gs = _segment(_clock_mask(_crop(rgb, CLOCK_BOX)), min_w=6, max_h=90)
    if not 1 <= len(gs) <= 3:
        return None
    digits = []
    for g in gs:
        d = _classify(g, "clock", CLOCK_GLYPH)
        if d is None:
            return None
        digits.append(d)
    return int("".join(str(d) for d in digits))


def clock_warning(rgb) -> bool:
    """倒计时是否已进警戒色（橙红，≤10s）；数字读不出时的兜底。"""
    p = _crop(rgb, CLOCK_BOX).astype(np.int16)
    r, g, b = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    m = (r > 130) & (r - g > 45) & (b < 110)
    return float(m.mean()) >= 0.02


def _den_ok(rgb) -> bool:
    """分母恒为 15；白字「1」只有 7px 宽、和斜杠难分，所以只校验右端「5」。

    最终攻势警报里同一区段会出现「回合 14」，没有「5」，据此排除误读。
    """
    for g in _segment(_den_mask(_crop(rgb, COUNTER_DEN_BOX)), min_w=4, max_h=40):
        if _classify(g, "hpden", COUNTER_GLYPH) == 5:
            return True
    return False


def read_boss(rgb) -> dict | None:
    """首领进度 n/15；分子读出且分母校验通过才返回。"""
    num_gs = _segment(_num_mask(_crop(rgb, COUNTER_NUM_BOX)), min_w=4, max_h=40)
    if not 1 <= len(num_gs) <= 2 or not _den_ok(rgb):
        return None
    num = _classify(num_gs[0], "hpnum", COUNTER_GLYPH)
    if len(num_gs) == 2:
        num2 = _classify(num_gs[1], "hpnum", COUNTER_GLYPH)
        if num is None or num2 is None:
            return None
        num = int(f"{num}{num2}")
    if num is None or not 0 <= num <= 15:
        return None
    return {"num": num, "den": 15}


def _round_img(rgb) -> np.ndarray:
    return _crop(rgb, ROUND_BOX).mean(2).astype(np.float32)


def read_round(rgb) -> int | None:
    """左上「回合 N」；整框与各回合模板做 NCC。模板未覆盖的回合返回 None。"""
    p = _pack()
    labels = sorted({int(k.split("_")[1]) for k in p if k.startswith("round_")})
    if not labels:
        return None
    g = _round_img(rgb)
    if g.shape != ROUND_SHAPE:
        g = _norm(g / 255.0, ROUND_SHAPE)
    scores = sorted(((max(_ncc(g, t) for t in p[f"round_{d}"]), d) for d in labels), reverse=True)
    best, d = scores[0]
    margin = best - scores[1][0] if len(scores) > 1 else 1.0
    if best < ROUND_MIN_SCORE or margin < ROUND_MIN_MARGIN:
        return None
    return d
