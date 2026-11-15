r"""生命值 / 扣血 / 商店金币 数值识别（免 OCR）。

字形模板是 14×22 的归一化二值图，按「槽位 + 数字」分组存于 templates/digits.npz：
  life_<d>：生命值计数器的白字（720p 下字形仅 8×12，已接近可分辨下限）
  dmg_<d> ：作战结束横幅「生命值减少 N」里的红字 N
  gold_<d>：休整期金币（右下角金色徽章，右对齐）
模板由 build_digits.py 从带真值的帧采收。未见过的数字返回 None，不做猜测。
生命槽已有 0–4、7–9；5/6 仍靠 gold_* 同字体兜底（模板候选集=life∪gold）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
PACK = BASE / "templates" / "digits.npz"

# 1920 规范坐标
LIFE_NUM_BOX = (1164, 27, 1215, 63)
LIFE_OCR_BOX = (1140, 27, 1222, 63)    # 左留图标余量、右留扣血标记，读前按字块掩码
DMG_NUM_BOX = (1089, 549, 1135, 588)   # 起点避开心形图标（1068–1085）
GOLD_NUM_BOX = (1792, 812, 1830, 838)
GOLD_RING_BOX = (1770, 796, 1874, 850)
GOLD_MIN_FRAC = 0.05
GLYPH = (14, 22)
MIN_SCORE = 0.60
MIN_MARGIN = 0.08
# 生命值字形宽 5–16、高 ≤20：1 只有 5–7px 宽（min_w 必须 ≤5，否则 21→2、14→4），
# 盾牌图标高 21–25 靠 max_h 排除，青绿描边（G−R≈130）再按颜色排一次。
LIFE_GLYPH_MAX_W = 16
LIFE_GLYPH_MAX_H = 20
LIFE_MIN_W = 5
LIFE_CLUSTER_GAP = 10          # 「1」「4」间隙实测 4–5px，渲染抖动会到 6
LIFE_OCR_MIN = 0.70

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


def _norm(g: np.ndarray) -> np.ndarray:
    return np.asarray(
        Image.fromarray((g * 255).astype(np.uint8)).resize((GLYPH[0], GLYPH[1]), Image.BILINEAR),
        dtype=np.float32) / 255.0


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def segment(mask: np.ndarray, min_w: int = 5, max_h: int = 24) -> list[np.ndarray]:
    """按列投影切字形；顺带滤掉贴在左缘的盾牌图标残片（矮宽、比数字高）。"""
    cols = mask.sum(0)
    on = cols > 0
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
        sub = mask[:, x0:x1]
        rows = np.nonzero(sub.sum(1) > 0)[0]
        if len(rows) == 0:
            continue
        h = int(rows.max() - rows.min() + 1)
        if x1 - x0 >= min_w and h <= max_h:
            out.append(sub[rows.min():rows.max() + 1].astype(np.float32))
    return out


def classify(glyph: np.ndarray, slot: str) -> tuple[int | None, float, float]:
    """返回 (数字或 None, 最高分, 与次优的间距)。"""
    p = _pack()
    labels = [int(k.split("_")[1]) for k in p if k.startswith(slot + "_")]
    if not labels:
        return None, 0.0, 0.0
    scores = []
    g = _norm(glyph)
    for d in labels:
        scores.append((max(_ncc(g, t) for t in p[f"{slot}_{d}"]), d))
    scores.sort(reverse=True)
    best, d = scores[0]
    margin = best - scores[1][0] if len(scores) > 1 else 1.0
    if best < MIN_SCORE or margin < MIN_MARGIN:
        return None, best, margin
    return d, best, margin


def _classify_union(glyph: np.ndarray, slots) -> tuple[int | None, float, float]:
    """在多个槽位的字形里取同一数字的最高分：生命槽缺 0/5–9 时由金币字形兜底。"""
    p = _pack()
    g = _norm(glyph)
    best: dict[int, float] = {}
    for k, arr in p.items():
        slot, _, num = k.partition("_")
        if slot not in slots:
            continue
        d = int(num)
        best[d] = max(best.get(d, -1.0), max(_ncc(g, t) for t in arr))
    ranked = sorted(((s, d) for d, s in best.items()), reverse=True)
    if not ranked:
        return None, 0.0, 0.0
    b, d = ranked[0]
    margin = b - ranked[1][0] if len(ranked) > 1 else 1.0
    if b < MIN_SCORE or margin < MIN_MARGIN:
        return None, b, margin
    return d, b, margin


def _crop(rgb: np.ndarray, box):
    from screen_detector import crop_canonical
    return crop_canonical(rgb, box)


def _white_mask(patch, thr=120):
    a = np.asarray(Image.fromarray(patch).convert("L"), dtype=np.float32)
    return a > thr


def _red_mask(patch):
    a = patch.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (r > 140) & (g < 100) & (b < 100)


def _gold_mask(patch):
    a = np.asarray(Image.fromarray(patch).convert("L"), dtype=np.float32)
    p99 = float(np.percentile(a, 99))
    if p99 < 120:            # 金币徽章不在（商店收起）
        return np.zeros(a.shape, dtype=bool)
    return a > max(110.0, 0.60 * p99)


def _digit_runs(mask):
    """列投影切块，返回 (x0, x1, y0, y1, h)。"""
    cols = mask.sum(0) > 0
    runs, st = [], None
    for i, v in enumerate(cols):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((st, i))
            st = None
    if st is not None:
        runs.append((st, len(cols)))
    out = []
    for x0, x1 in runs:
        sub = mask[:, x0:x1]
        rows = np.nonzero(sub.sum(1) > 0)[0]
        if len(rows) == 0:
            continue
        out.append((x0, x1, int(rows.min()), int(rows.max()) + 1,
                    int(rows.max() - rows.min() + 1)))
    return out


def _life_runs(rgb):
    """生命计数器里可能是数字的白字块：宽 ≤16、高 ≤20、非青绿描边。"""
    patch = _crop(rgb, LIFE_OCR_BOX)
    white = _white_mask(patch)
    p = patch.astype(np.int16)
    green = p[:, :, 1] - p[:, :, 0]
    runs = []
    for x0, x1, y0, y1, h in _digit_runs(white):
        if x1 - x0 > LIFE_GLYPH_MAX_W or h > LIFE_GLYPH_MAX_H:
            continue
        sub = white[y0:y1, x0:x1]
        if float(green[y0:y1, x0:x1][sub].mean()) > 25:
            continue                       # 盾牌图标的青绿描边，不是字
        runs.append((x0, x1, y0, y1, h))
    return runs


def read_life_ocr(rgb) -> int | None:
    """计数器 OCR：白字掩码只留 1–2 个数字块，其余涂白后再识别。"""
    import ocr
    patch = _crop(rgb, LIFE_OCR_BOX)
    if patch.size == 0:
        return None
    runs = [(x0, x1) for x0, x1, _, _, _ in _life_runs(rgb)]
    if not runs:
        return None
    groups = [[runs[0]]]
    for r in runs[1:]:
        if r[0] - groups[-1][-1][1] <= LIFE_CLUSTER_GAP:
            groups[-1].append(r)
        else:
            groups.append([r])
    g = groups[0]                          # 最左簇＝计数器；右边是「−N」扣血标记
    if len(g) > 2:
        return None
    x0, x1 = g[0][0], g[-1][1]
    band = np.full_like(patch, 255)
    band[:, x0:x1] = patch[:, x0:x1]
    text = ocr.read_patch_text(band, min_score=LIFE_OCR_MIN)
    if not text or not all(c in "0123456789" for c in text) or len(text) > 2:
        return None
    return int(text)


def _read_life_tpl(rgb) -> int | None:
    """模板读法：候选集=生命槽∪金币槽（同字体），1–2 个字形都必须命中。"""
    patch = _crop(rgb, LIFE_NUM_BOX)
    gs = segment(_white_mask(patch), min_w=LIFE_MIN_W, max_h=LIFE_GLYPH_MAX_H)
    gs = [g for g in gs if g.shape[1] <= LIFE_GLYPH_MAX_W]
    out = []
    for g in gs:
        d, _, _ = _classify_union(g, ("life", "gold"))
        if d is None:
            return None
        out.append(d)
    if not out or len(out) > 2:
        return None
    return int("".join(str(d) for d in out))


def read_life(rgb) -> int | None:
    """生命值数字：OCR 为主，模板交叉校验；两者等长且冲突时不猜。"""
    tpl = _read_life_tpl(rgb)
    got = read_life_ocr(rgb)
    # 模板只切出 1 位、掩码里却有 2 个以上够宽够高的字块：薄「1」很可能被切丢，
    # 单看模板会把 21 读成 2、14 读成 4；OCR 也没结果时宁可不猜。
    if got is None and tpl is not None and tpl < 10:
        wide = [r for r in _life_runs(rgb) if r[1] - r[0] >= LIFE_MIN_W and r[4] >= 8]
        if len(wide) >= 2:
            return None
    if tpl is None:
        return got
    if got is None:
        return tpl
    if got == tpl:
        return tpl
    ts, gs = str(tpl), str(got)
    # 模板漏掉薄「1」会截短（21→2 丢首位、14→4 丢末位），短的是长的前缀/后缀才采信长的
    short, long_ = (ts, gs) if len(gs) >= len(ts) else (gs, ts)
    if long_.startswith(short) or long_.endswith(short):
        return int(long_)
    return None


def _gold_frac(patch) -> float:
    a = patch.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return float(((r > 150) & (g > 110) & (b < 150) & (r - b > 60)).mean())


def read_gold(rgb) -> int | None:
    """休整期金币数；读不出来返回 None。"""
    if _gold_frac(_crop(rgb, GOLD_RING_BOX)) < GOLD_MIN_FRAC:
        return None
    patch = _crop(rgb, GOLD_NUM_BOX)
    gs = segment(_gold_mask(patch), min_w=6, max_h=40)
    if any(g.shape[1] > 22 for g in gs):
        return None
    digits = []
    for g in gs:
        d, _, _ = classify(g, "gold")
        if d is None:
            return None
        digits.append(d)
    if not digits or len(digits) > 3:
        return None
    return int("".join(str(d) for d in digits))


def read_damage(rgb) -> int | None:
    """作战结束横幅红字 N（生命值减少 N）；读不出来返回 None。

    红字与金币/生命值同一套字形，直接复用 gold_* 模板（实测 4→0.91、1→0.98、
    0→0.91，次优 ≤0.36）。生命值上限 10，超出按 10 记。
    """
    patch = _crop(rgb, DMG_NUM_BOX)
    gs = segment(_red_mask(patch), min_w=8, max_h=34)
    digits = []
    for g in gs:
        d, _, _ = classify(g, "gold")
        if d is None:
            return None
        digits.append(d)
    if not digits or len(digits) > 2:
        return None
    return min(int("".join(str(d) for d in digits)), 10)


class LifeTracker:
    """连续 need 帧读数一致才采纳，连续 miss 帧不出现才作废。

    最终攻势橙屏会盖住计数器（h_001 曾把 2 读成 4），单帧读数不足以改判。
    """

    def __init__(self, need: int = 2, miss: int = 2):
        self.need = need
        self.miss_limit = miss
        self.value = None
        self._pending = None
        self._count = 0
        self._miss = 0

    def update(self, value: int | None) -> int | None:
        if value is None:
            self._pending = None
            self._count = 0
            self._miss += 1
            if self._miss >= self.miss_limit:
                self.value = None
            return self.value
        self._miss = 0
        if value == self._pending:
            self._count += 1
        else:
            self._pending = value
            self._count = 1
        if self._count >= self.need:
            self.value = value
        return self.value


class GoldTracker(LifeTracker):
    """金币读数确认：与生命值同语义，单独命名便于日志与调参。"""
