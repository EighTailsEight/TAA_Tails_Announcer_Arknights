r"""卫戍协议 HUD 检测（纯 numpy/PIL，不依赖 cv2）。

内部规范坐标 1920×1080：ROI/模板都按该尺度写，输入帧按比例裁到 ROI 再缩放，
不做整帧缩放（整帧重采样 + 全屏掩码会吃掉 300ms/帧）。

模板用二值掩码做 NCC：背景（地图、半透明横幅）不参与匹配，实测分离度 >0.9。
已验证锚点（720p 录屏 → 1920 乘 1.5）：
  联防开始横幅   文本左上 (602,322) 尺寸 162×52 @720，橙字
  作战结束横幅   文本左上 (604,330) 尺寸 166×58 @720，白字
  顶部状态胶囊   规范 (840,10)-(1160,82)：teal=休息/购买，金/橙=n/m 战斗波次，
                 红条+低绿色均值=最终攻势；文字页（菜单/结果）全部近 0
  扣血红字带     x 600..790, y 368..404 @720
  回合 N 芯片   规范 (750,34)-(790,56)，最终攻势期间被 n/15 红条取代
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import hud_digits

BASE = Path(__file__).resolve().parent
TPL_DIR = BASE / "templates"
CANON = (1920, 1080)
MARK_PACK = TPL_DIR / "marks.npz"


def _as_rgb(img) -> np.ndarray:
    arr = np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) else np.asarray(img)[:, :, :3]
    return arr


def crop_canonical(rgb: np.ndarray, box) -> np.ndarray:
    """按 1920 规范框裁切任意分辨率的帧，并把裁片缩放到规范尺寸。"""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    sx, sy = w / CANON[0], h / CANON[1]
    bx = (int(round(x1 * sx)), int(round(y1 * sy)),
          int(round(x2 * sx)), int(round(y2 * sy)))
    patch = rgb[bx[1]:bx[3], bx[0]:bx[2]]
    tw, th = x2 - x1, y2 - y1
    if patch.shape[1] != tw or patch.shape[0] != th:
        patch = np.asarray(Image.fromarray(patch).resize((tw, th), Image.BILINEAR))
    return patch


def _quarter(patch: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(patch).resize((patch.shape[1] // 4, patch.shape[0] // 4), Image.BILINEAR))


# ------------------------------------------------------------------ 掩码

def _ch(rgb):
    a = rgb.astype(np.int16)
    return a[:, :, 0], a[:, :, 1], a[:, :, 2]


def mask_orange(rgb) -> np.ndarray:
    r, g, b = _ch(rgb)
    return ((r > 150) & (g > 85) & (g < 215) & (b < 110) & (r - b > 70)).astype(np.float32)


def mask_bright(rgb) -> np.ndarray:
    r, g, b = _ch(rgb)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return ((mn > 100) & (mx - mn < 90)).astype(np.float32)


def mask_teal(rgb) -> np.ndarray:
    r, g, b = _ch(rgb)
    return ((g > 100) & (b > 80) & (r < 120) & (g - r > 25)).astype(np.float32)


def mask_red(rgb) -> np.ndarray:
    r, g, b = _ch(rgb)
    return ((r > 150) & (g < 90) & (b < 90)).astype(np.float32)


MASKS = {
    "orange": mask_orange,
    "bright": mask_bright,
    "teal": mask_teal,
    "red": mask_red,
}


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def ncc_at(mask: np.ndarray, tpl_zm: np.ndarray, tpl_norm: float, y: int, x: int) -> float:
    h, w = tpl_zm.shape
    patch = mask[y:y + h, x:x + w]
    patch = patch - patch.mean()
    den = float(np.sqrt((patch * patch).sum())) * tpl_norm
    return float((patch * tpl_zm).sum() / den) if den > 1e-9 else -1.0


def best_ncc(mask: np.ndarray, tpl_zm: np.ndarray, anchor_xy=None, radius: int = 0,
             pad: int = 3) -> tuple[float, tuple | None]:
    """定位 + 校验：先按行/列投影做一维互相关拿到候选位置，再在 ±pad 内做二维 NCC。

    二值文字掩码对 1–2px 错位极敏感（笔画只有 4–6px 宽），所以二维校验不能省；
    但整窗二维全扫太慢（≈80ms/帧），一维投影把候选从 O(r²) 压到 O(1)。
    投影可能锁到窗口里另一条横排文字，此时二维 NCC 分数会掉下去，不会误报。
    """
    H, W = mask.shape
    h, w = tpl_zm.shape

    def proj_peak(profile, tprof):
        p = profile - profile.mean()
        t = tprof - tprof.mean()
        den = float(np.linalg.norm(p) * np.linalg.norm(t))
        if den < 1e-9:
            return 0
        c = np.correlate(p, t, "valid") / den
        return int(np.argmax(c))

    y0 = proj_peak(mask.sum(1), tpl_zm.sum(1))
    x0 = proj_peak(mask.sum(0), tpl_zm.sum(0))
    tpl_norm = float(np.sqrt((tpl_zm * tpl_zm).sum()))
    best, at = -2.0, None
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            x, y = x0 + dx, y0 + dy
            if x < 0 or y < 0 or y + h > H or x + w > W:
                continue
            v = ncc_at(mask, tpl_zm, tpl_norm, y, x)
            if v > best:
                best, at = v, (x, y)
    return best, at


@dataclass(frozen=True)
class Target:
    name: str
    mask: str
    tpl: str
    roi: tuple             # 1920 坐标：搜索窗（掩码只在该窗内算）
    threshold: float


TARGETS = [
    Target("defense_phase", "orange", "defense_phase.png", (860, 440, 1200, 610), 0.55),
    Target("battle_end", "bright", "battle_end.png", (860, 445, 1210, 630), 0.55),
]

# 顶部状态胶囊。休息：teal 占比 0.12–0.24；战斗波次：金/橙；最终攻势：红条且
# 红像素平均绿色分量 ~56（橙色联防横幅/波次胶囊 ~72–93，用 mg 区分）。
PILL_BOX = (840, 10, 1160, 82)
PILL_REST_TEAL = 0.08
PILL_FINAL_RED = 0.025
PILL_FINAL_MG = 68.0
PILL_GOLD = 0.03
PILL_ORANGE = 0.02

# 联防期底部说明条：观战队友时是亮 teal 底 + 眼睛图标，自己防守时是暗底 + 「自己」
DEF_BAR = (742, 912, 1190, 978)
DEF_BAR_TEAL = 0.30
DEF_BAR_WHITE = 0.004

DAMAGE_BAND = (930, 548, 1125, 590)   # 1920 坐标：横幅标题下方红字带
DAMAGE_MIN_PX = 300
DAMAGE_MAX_H = 34                     # 红字行高；更高的红色块是地图上的敌人标记
DAMAGE_MIN_W = 90

# 文字页标记（NCC 0.55，实测同一帧 1.00、异帧 ≤0.21）
MARKS = {
    # 模式入口标题「卫戍协议」logo：比主页淡入早约 2s，独立于 e4_0017=1.00
    "mode_logo": ("red", (294, 192, 690, 293)),
    "difficulty": ("bright", (1540, 12, 1866, 80)),
    "match_found": ("teal", (1480, 850, 1760, 960)),
    "strategy_page": ("bright", (680, 18, 1080, 90)),
    "result_victory": ("teal", (218, 300, 600, 393)),
    "result_defeat": ("red", (218, 300, 600, 393)),
    # 主界面：青色「搜寻队友」+ 两个灰框（实测 a_0007..10 ≈1.00，异帧 ≤0.12）
    "lobby_search": ("teal", (1532, 930, 1851, 1017)),
    "lobby_sweep": ("bright", (1586, 108, 1895, 174)),
    "lobby_mat": ("bright", (1586, 206, 1895, 273)),
    # 匹配中「搜寻中」(0s) 白字；a_0013=1.00，a_0014=0.55
    "searching": ("bright", (1385, 290, 1630, 362)),
    # 决策三页（机密商店/战术/悬赏）左上大标题共用同一框；各模板自匹配 1.00，
    # 互相 ≤0.38
    "decision_shop": ("teal", (248, 46, 456, 98)),
    "decision_tac": ("teal", (248, 46, 456, 98)),
    "decision_bounty": ("teal", (248, 46, 456, 98)),
    # 核心盟约白标题：本帧 1.00，其余 0.00
    "core_covenant": ("bright", (123, 166, 323, 214)),
    # 策略页右侧「请选择策略」；旧 strategy_page 只到 0.50，低于阈值
    "strategy_pick": ("teal", (1391, 32, 1678, 71)),
    # 最终攻势开场橙字「击败敌方首领!」（多帧持续）：同帧 1.00，其余 10 帧 ≤0.16
    "final_overlay": ("orange", (818, 656, 1102, 708)),
    # 隐秘核心已解锁横幅：同帧 1.00，其余 ≤0.16
    "secret_core": ("teal", (796, 600, 1122, 680)),
}
MARK_MIN_SCORE = 0.55
LOBBY_MIN = 0.75
LOBBY_NEED = 2
LOBBY_TEXT_CAND = 0.45   # 标记淡入帧就够触发文字确认（文字才是最终判据）
SEARCHING_MIN = 0.70

# 这两条不依赖胶囊：最终攻势横幅在战斗中出现时 pill 可能已经是 combat/final，
# 隐秘核心横幅也紧接在胶囊消失的过场里。每帧都算，代价是两次小裁片 NCC。
ALWAYS_MARKS = ("final_overlay", "secret_core")

# 准备阶段右上角「准备就绪」(teal) / 「取消准备」(bright，白字) 按钮。
# 实测 ready_on 0.99–1.00、ready_off 0.93–1.00，其它画面 ≤0.06。
READY_BTN = (1703, 132, 1886, 176)
READY_NCC = 0.65

# 最终攻势胶囊里的首领血条：条体在胶囊裁片内 y 24..38、x 25..269（满血 244px）。
# 胶囊里的 n/15 是波次，不是血量；血量按红条长度读（0/15 时条约 100%，超时约 23%）。
BOSS_BAR_X = (25, 269)
BOSS_BAR_Y = (24, 38)
# 3px ≈ 1.2%：旧值 10px 让血条最低只能读到 4.1%，击杀前的最后一两帧
# 直接变成 None，boss_victory 永远等不到 ≤2%。噪声靠列投影 ≥6 行滤掉。
BOSS_BAR_MIN_PX = 3


def boss_bar_fill(patch) -> float | None:
    red = mask_red(patch[BOSS_BAR_Y[0]:BOSS_BAR_Y[1]])
    cols = red.sum(0)
    xs = np.where(cols >= 6)[0]
    xs = xs[xs >= BOSS_BAR_X[0]]
    if len(xs) == 0:
        return None
    end = int(xs.max())
    x0, x1 = BOSS_BAR_X
    if end - x0 < BOSS_BAR_MIN_PX:
        return None
    return float(np.clip((end - x0) / (x1 - x0), 0.0, 1.0))


def pill_state(patch) -> str | None:
    """顶部胶囊状态：rest / final / combat / None。传入已按 PILL_BOX 裁切的规范图。"""
    teal = mask_teal(patch)
    if float(teal.mean()) >= PILL_REST_TEAL:
        return "rest"
    red = mask_red(patch)
    red_frac = float(red.mean())
    if red_frac >= PILL_FINAL_RED:
        r, g, b = _ch(patch)
        loose = (r > 150) & (r - g > 70) & (r - b > 70) & (g < 170)
        mg = float(g[loose].mean()) if loose.any() else 255.0
        if mg < PILL_FINAL_MG:
            return "final"
    r, g, b = _ch(patch)
    gold = (r > 150) & (g > 140) & (b < 100) & (g > 0.65 * r)
    if float(gold.mean()) >= PILL_GOLD:
        return "combat"
    orange = (r > 150) & (g > 70) & (g < 170) & (b < 100) & (r - b > 90) & (r - g > 40)
    if float(orange.mean()) >= PILL_ORANGE:
        return "combat"
    return None


@dataclass
class ScreenReading:
    scores: dict = field(default_factory=dict)
    at: dict = field(default_factory=dict)
    marks: dict = field(default_factory=dict)
    pill: str | None = None
    pill_teal_frac: float = 0.0
    clock: int | None = None
    clock_warn: bool = False
    boss: dict | None = None
    boss_hp: float | None = None
    round: int | None = None
    damage_red_px: int = 0
    damage_shape_ok: bool = False
    bar_teal_frac: float = 0.0
    bar_white_frac: float = 0.0
    ready_btn: bool = False
    lobby_text: bool = False      # OCR：「搜寻队友」/「独立模拟」+ 一条副标题
    rest_text: bool | None = None  # OCR：休整胶囊读「休息一下」；None=读不出

    @property
    def lobby(self) -> bool:
        n = sum(self.mark(k) >= LOBBY_MIN
                for k in ("lobby_search", "lobby_sweep", "lobby_mat"))
        return n >= LOBBY_NEED

    @property
    def searching(self) -> bool:
        return self.mark("searching") >= SEARCHING_MIN

    @property
    def defense_phase(self) -> bool:
        return self.scores.get("defense_phase", 0) >= 0.55

    @property
    def battle_end(self) -> bool:
        return self.scores.get("battle_end", 0) >= 0.55

    @property
    def end_damaged(self) -> bool:
        return self.damage_shape_ok

    @property
    def rest(self) -> bool:
        return self.pill == "rest"

    @property
    def combat(self) -> bool:
        return self.pill == "combat"

    @property
    def final_assault(self) -> bool:
        return self.pill == "final"

    @property
    def watching_teammate(self) -> bool:
        return self.bar_teal_frac >= DEF_BAR_TEAL

    @property
    def defending_self(self) -> bool:
        return self.bar_teal_frac < DEF_BAR_TEAL and self.bar_white_frac >= DEF_BAR_WHITE

    def mark(self, name: str) -> float:
        return self.marks.get(name, 0.0)

    @property
    def result_page(self) -> bool:
        return self.mark("result_victory") >= MARK_MIN_SCORE or self.mark("result_defeat") >= MARK_MIN_SCORE

    @property
    def final_overlay(self) -> bool:
        return self.mark("final_overlay") >= MARK_MIN_SCORE

    @property
    def secret_core(self) -> bool:
        return self.mark("secret_core") >= MARK_MIN_SCORE

    @property
    def decision_page(self) -> bool:
        return max(self.mark(k) for k in
                   ("decision_shop", "decision_tac", "decision_bounty")) >= MARK_MIN_SCORE

    @property
    def core_covenant_page(self) -> bool:
        return self.mark("core_covenant") >= MARK_MIN_SCORE

    @property
    def strategy_pick_page(self) -> bool:
        return self.mark("strategy_pick") >= MARK_MIN_SCORE


class ScreenDetector:
    def __init__(self):
        self._tpls = {}
        for t in TARGETS:
            rgb = np.asarray(Image.open(TPL_DIR / t.tpl).convert("RGB"))
            m = MASKS[t.mask](rgb)
            self._tpls[t.name] = m - m.mean()
        self._marks = {}
        if MARK_PACK.exists():
            z = np.load(MARK_PACK)
            self._marks = {k: z[k] for k in z.files}

    def analyze(self, img) -> ScreenReading:
        rgb = _as_rgb(img)
        read = ScreenReading()
        for t in TARGETS:
            x1, y1, x2, y2 = t.roi
            win = crop_canonical(rgb, t.roi)
            score, at = best_ncc(MASKS[t.mask](win), self._tpls[t.name])
            read.scores[t.name] = score
            read.at[t.name] = None if at is None else (at[0] + x1, at[1] + y1)

        band = crop_canonical(rgb, DAMAGE_BAND)
        m = mask_red(band)
        n = int(m.sum())
        read.damage_red_px = n
        if n >= DAMAGE_MIN_PX:
            ys, xs = np.nonzero(m)
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
            read.damage_shape_ok = h <= DAMAGE_MAX_H and w >= DAMAGE_MIN_W

        pill = crop_canonical(rgb, PILL_BOX)
        read.pill = pill_state(pill)
        read.pill_teal_frac = float(mask_teal(pill).mean())

        # 准备按钮在休整里也会出现（pill=rest），所以不能放进 MARKS（那只在 pill
        # 为空时算）。teal=准备就绪、bright=取消准备，任一对上都算按钮在场。
        on, off = self._marks.get("ready_on"), self._marks.get("ready_off")
        if on is not None and off is not None:
            rb = crop_canonical(rgb, READY_BTN)
            read.ready_btn = max(_ncc(mask_teal(rb), on),
                                 _ncc(mask_bright(rb), off)) >= READY_NCC

        bar = crop_canonical(rgb, DEF_BAR)
        read.bar_teal_frac = float(mask_teal(bar).mean())
        a = bar.astype(np.int16)
        read.bar_white_frac = float(((a[:, :, 0] > 190) & (a[:, :, 1] > 190)
                                     & (a[:, :, 2] > 190)).mean())

        if read.pill is not None:
            read.clock = hud_digits.read_clock(rgb)
            read.clock_warn = hud_digits.clock_warning(rgb)
            if read.pill == "final":
                read.boss = hud_digits.read_boss(rgb)
                # 波次数字读不出时血条仍要播：血条与 n/15 是两个独立锚点
                read.boss_hp = boss_bar_fill(pill)
            else:
                read.round = hud_digits.read_round(rgb)
        else:
            for name, (mask_name, box) in MARKS.items():
                tpl = self._marks.get(name)
                if tpl is None or name in ALWAYS_MARKS:
                    continue
                read.marks[name] = _ncc(MASKS[mask_name](crop_canonical(rgb, box)), tpl)
        for name in ALWAYS_MARKS:
            tpl = self._marks.get(name)
            if tpl is None:
                continue
            mask_name, box = MARKS[name]
            read.marks[name] = _ncc(MASKS[mask_name](crop_canonical(rgb, box)), tpl)
        if read.pill is None:
            # 只在标记沾边的帧上跑 OCR（每次 16–22ms），文字做最终确认
            if max(read.mark(k) for k in
                   ("lobby_search", "lobby_sweep", "lobby_mat")) >= LOBBY_TEXT_CAND:
                from text_anchors import lobby_text
                read.lobby_text = lobby_text(rgb)
        elif read.pill == "rest":
            # 胶囊泛青不等于休整（游戏主界面/战斗横幅那圈也是青的），读字确认
            from text_anchors import rest_pill_text
            read.rest_text = rest_pill_text(rgb)
        return read


# ------------------------------------------------------------------ 事件流

class DefenseTracker:
    """把逐帧读数归纳成 联防 阶段事件。

    - defense_enter：看到 联防阶段 横幅。横幅只显示约 2–3s，轮询 ≥4Hz 才稳；
      进入后忽略后续横幅帧，下次联防由下一段横幅重新触发。
    - add_segment(damaged)：一场子战斗的 作战结束（由 bridge 扣血保持后调用）。
    - phase_end：持续休息 ≥ rest_hold（休整买卡）或看到最终攻势。结束不播报结果，
      success 由 bridge 按段判：有段时 any_clean，否则看整段有没有掉血。
    """

    def __init__(self, bridge: float = 6.0, rest_hold: float = 1.0):
        self.bridge = bridge
        self.rest_hold = rest_hold
        self.in_defense = False
        self.entered_at = None
        self.last_banner = None
        self.segments = []          # [(t, damaged)]
        self._rest_since = None

    def update(self, read: ScreenReading, t: float) -> list[dict]:
        ev = []
        if read.defense_phase:
            self.last_banner = t
        if not self.in_defense:
            if self.last_banner is not None and t - self.last_banner <= self.bridge:
                self.in_defense = True
                self.entered_at = self.last_banner
                self.segments = []
                self._rest_since = None
                ev.append({"kind": "defense_enter", "t": self.entered_at})
            else:
                return ev
        if read.final_assault:
            ev.append(self._end(t))
            return ev
        if read.rest:
            if self.segments:
                # 子战斗已结算 + 回到休整 = 联防阶段确实结束（结算语音可提前约 1s）
                ev.append(self._end(t))
                return ev
            if self._rest_since is None:
                self._rest_since = t
            if t - self._rest_since >= self.rest_hold:
                ev.append(self._end(t))
        else:
            self._rest_since = None
        return ev

    def add_segment(self, t: float, damaged: bool) -> None:
        if self.in_defense:
            self.segments.append((t, damaged))

    def finish(self, t: float) -> list[dict]:
        """结果页/收尾时强制结束联防（bridge 在 run 结束时调用）。"""
        return [self._end(t)] if self.in_defense else []

    def _end(self, t: float) -> dict:
        ev = {
            "kind": "phase_end",
            "t": t,
            "segments": len(self.segments),
            "any_clean": any(not d for _, d in self.segments),
            "any_damage": any(d for _, d in self.segments),
        }
        self.in_defense = False
        self.entered_at = None
        self.segments = []
        self.last_banner = None
        self._rest_since = None
        return ev
