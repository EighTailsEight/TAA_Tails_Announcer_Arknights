r"""休整期商店：卡位、等级徽章、金币。

只读画面。锚点（1920 规范坐标，实测）：
  卡位     SLOT_XS 宽 207，右对齐；等级面板在行首（6 张时 x 201..351）
  名称条   y 1030..1062：teal≥0.20 或金牌实底≥0.30 = 有卡；三连 0 费卡另有判据
  等级徽章 卡位左上 (x+8,816)-(x+52,860)：III/IV 看颜色，I/II 白族、V/VI 金族
           各按灰度 NCC 比形状
  等级     LEVEL 牌识别用掩码 NCC（数字/费用/按钮遮掉）；级别读罗马数字笔画：
           I/II/III = 1/2/3 根窄竖，V = 一根宽 V，IV/VI = 窄+宽 / 宽+窄。
           灰牌=买不起，绿牌=可升级（每轮休整报一次），数字上升两帧确认=升级完成；
           确认后取消不改数字 → 不报。满级 VI 牌面仍绿，用等级数字压掉提醒；
           满级牌顶栏没有费用徽记，单独一套 compact_max 模板才够读数门限。
  金币     (1770,796)-(1874,850)，0/1/4/5/6/7/9 有模板；模板漏读时 OCR 兜底

购买完成 = 卡位从「有卡」变「空」，等级取该卡位上一帧缓存的徽章。
售出 = 金币上涨，且最近画面里出现过干员专属的金色「出售」按钮；右下的拖拽区
       （售出/销毁共用）不出售出背书，装备使用加钱一律不报。
刷新 = 卡位不空、卡面内容整排换掉且稳定 ≥0.4s（不再依赖扣金币）；战斗进入后 3s 静默。
冻结 = 招募区冻结按钮由青底转浅白，只报上升沿，商店收起时保持状态。
同名 = 已有同名干员时卡面滚金光（同名装备滚的是同一套光，按道具徽记排除），
       连续两帧确认后触发，15s 冷却，刷新清除冷却。
三连 = 右侧三张卡上栏正中青色「0」，确认 0.3s 后触发；180s latch，回到普通商店
       可见状态才清除；三连页内卡面变化 = 新三连。
整排同时变化（换回合、刷新）一律当重建；售卖/装备 UI 会盖住卡行，期间暂停读数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from digits import GoldTracker, read_gold
from ocr import read_gold_ocr, read_text
from screen_detector import crop_canonical, mask_bright, mask_teal

BASE = Path(__file__).resolve().parent
TPL = BASE / "templates" / "badges.npz"

SLOT_XS = (363, 582, 801, 1020, 1239, 1458)   # 1920 坐标
SLOT_W = 207
STRIP_Y = (1030, 1062)
STRIP_MIN = 0.20
STRIP_CARD_GOLD = 0.30                        # 金牌实底（0.49–0.51）；场上名字标签 ≤0.12
ART_MIN_STD = 9.0                             # 卡面灰度标准差：空位 ≈7，变暗卡 ≥15
STRIP_GOLD = 0.04                             # 三连 0 费卡只有金文字（0.057–0.139）
TRIPLE_PANEL_TEAL_MAX = 0.05                  # 三连卡身是暗的；场上平台 teal 0.10–0.79

BADGE_BOX = (8, 816, 52, 860)                 # (dx1, y1, dx2, y2) 徽章方牌内部
BADGE_MIN_SCORE = 0.60
BADGE_MIN_MARGIN = 0.10
BADGE_MIN_HI = 100.0                          # 最亮 12% 的中位亮度低于此 = 空位/面板
BADGE_MIN_FRAC = 0.02
BADGE_COLOR_MIN = 20.0                        # B-R 大于此 = 青/绿徽章
BADGE_HUE_SPLIT = 14.0                        # G-B 大于此 = 绿(III)，否则青(IV)
BADGE_GOLD_MIN = 30.0                         # R-B 大于此 = 金/橙徽章(V、VI 族)
# 同名金光/暖色环境会把白牌最亮像素染成金色（实测 II 牌 R−B 81）：颜色族与形状
# 冲突时，只有形状明显压倒颜色族才翻案，否则本帧不读、下帧重试。
BADGE_SHAPE_MIN = 0.90
BADGE_FAMILY_MARGIN = 0.15

PANEL_BOX = (200, 812, 580, 1070)             # 覆盖 5/6 卡两种面板位置
PANEL_SAME_CELLS = 32                         # 12×16 网格里允许的抖动格数

# 道具卡左下角的三圆徽记（与干员的职业/武器图标区分：道具 0.93+，干员 ≤0.58）
ITEM_BOX = (6, 1012, 38, 1046)                # 相对卡位 x 的偏移
ITEM_MIN_SCORE = 0.72
ITEM_PACK = TPL.parent / "item_icon_{n}.png"

# 「升级调度中心」牌永远是一张 150×218 的 compact 图，右对齐在第一个占用卡位：
# 6/5/4/3/2/1 卡时 x = 201 / 420 / 639 / 858 / 1077 / 1296（card_x + 207 - 150）。
# 四/五/六卡实测 0.901 / 0.903 / 0.917，其它位置 ≤0.17。
PLATE_XS = (201, 420, 639, 858, 1077, 1296)
PLATE_KEYS = ("compact", 0, 1, 2, 3, 4, 5)    # 牌占用的卡位（x=201 时贴行首）
PLATE_MIN = 0.66                              # 掩码 NCC（等级/费用/按钮已遮掉）
PLATE_GREEN_MIN = 0.25                        # 绿态占比 0.45–0.66，灰态 ≤0.03
# 等级数字列投影签名：同级实测 ≥0.94（暗帧 0.94，正常 0.99），I→II 掉到 −0.16。
# 换牌位只重建基线，不做跨位置比较；连续两帧同签名才算升级。
PLATE_NUM_BOX = (41, 80, 109, 150)            # 签名用相对牌面裁片
LEVEL_SIG_BINS = 32
LEVEL_SIG_MIN = 0.70
LEVEL_COOLDOWN = 3.0
# 罗马数字笔画框（比签名框宽，VI 的 V 贴左缘也不裁断）；只有牌面 NCC 够稳
# 才采信数字（动画帧 0.50–0.75，落定 0.93）。
NUMERAL_BOX = (30, 80, 120, 150)
NUMERAL_MIN_SCORE = 0.85
NUMERAL_MIN_H = 20                            # 数字高 48；LEVEL 字样残边只有 1–2 行
NUMERAL_MIN_W = 4
NUMERAL_NARROW = 20                           # I 竖 11 px，V 宽 46–47 px
NUMERAL_WIDE = 30
LEVEL_MAX = 6
NUMERAL_CONFIRM = 2                           # 等级读数/升级都要连续两帧一致
GREEN_HOLD = 0.5                              # 绿态持续这么久才提示可升级（滤动画/悬停）
REFRESH_STREAK_EGG = 2                        # 连续刷新 ≥2 次且中途没买 → 彩蛋语音                        # 防同一帧多次读数重复报
REFRESH_GOLD_WINDOW = 2.5                     # 刷新一定扣 1 金币；没扣款的卡面变化是加载/换回合
LEVEL_PACK = TPL.parent / "level_plate.npz"
# 满级牌没有费用徽记（顶栏是空的）、底部是「已满级」，与灰/绿两套模板的掩码外
# 区域不同（实测只到 0.70，数字会被闸门挡掉）；单独采一张 compact_max。
PLATE_TPL_KEYS = ("compact_gray", "compact_green", "compact_max")

READY_BOX = (1050, 990, 1140, 1026)           # 「已就绪，正在等待进入模拟」中段
READY_MIN_FRAC = 0.50

SELL_BTN_BOX = (900, 495, 1035, 578)          # 选中场上干员时的金色「出售」按钮
SELL_BTN_MIN = 0.30                           # 金色占比闸门（先过闸再比形状）
SELL_BTN_NCC = 0.45
SALE_ZONE_BOX = (1680, 792, 1902, 1020)       # 右下拖拽区：售出=金，销毁=红
SALE_ZONE_GOLD_MIN = 0.07
SALE_ZONE_RED_MIN = 0.06
SELL_UI_WINDOW = 2.5                          # 金币上涨在多近的出售按钮之后才算售出

# 卡面内容签名：16×16 归一化灰度；刷新 = 空位没变、整排内容换掉且稳定 0.4s
SIG_BOX = (10, 860, 197, 1000)                # 相对卡位 x 的偏移
SIG_GRID = 16
SIG_MIN_NCC = 0.80
SIG_STABLE = 0.4
SIG_MUTE = 3.0                                # 战斗进入后商店动画静默
SIG_OPEN_MUTE = 2.0                           # 商店刚出现时的淡入/飞卡动画
SIG_REBUILD_MUTE = 1.0                        # 卡行重排/补位动画（实测 0.75s 内落定）

# 同名干员：卡面滚金光（同名装备也滚同样的光，按道具徽记排除）
GLOW_MIN = 0.08                               # 金像素占比；正常 0.0，装备底噪 0.044，闪光 0.19–0.23
GLOW_MAX = 0.45                               # 整卡金牌实底（协防干员/悬停高亮）实测 0.54+，不是同名闪光
GLOW_CONFIRM = 2                              # 连续两帧过阈值才确认（4fps 下闪光持续约 0.5s）
GLOW_MIN_SPAN = 0.10                          # 两帧至少隔这么久，滤掉单帧动画闪
SAME_OP_COOLDOWN = 15.0                       # 冷却期内不重播；刷新清除

# 三连奖励：右侧三张卡上栏正中青色「0」
TRIPLE_SLOTS = (3, 4, 5)
TRIPLE_MIN_CARDS = 2                          # 第三张常常还在加载（NO INFO 占位）
ZERO_BOX = (85, 810, 120, 845)                # 相对卡位 x 的偏移
ZERO_MIN = 0.05                               # 正常 0–0.003，三连 0.106–0.114
TRIPLE_CONFIRM = 0.3
TRIPLE_COOLDOWN = 180.0                       # latch，回到普通商店可见即清除

# 冻结招募区按钮：青底=可冻结，浅白=已冻结。颜色只判「按钮在不在」，
# 冻结状态以按钮文字为准：普通「冻结」，已冻结「解冻」（实测两帧 p=1.00）。
FREEZE_BOX = (1500, 692, 1702, 760)
FREEZE_TEAL_MAX = 0.10
FREEZE_BRIGHT_MIN = 0.30
FREEZE_CONFIRM = 0.2                          # 连续两帧「解冻」才触发（4fps=0.25s；滤过场白闪）
FREEZE_MISS = 0.5                             # 「解冻」读数掉到门限下的容忍时长
FREEZE_RESET = 0.5                            # 普通态持续多久才确认解冻/允许下一次冻结
FREEZE_ONLINE = 6.0                           # 冻结态最后可见到解冻的最大间隔；更久视为换轮静默复位
FREEZE_BLUE_MIN = 15.0                        # 冻结按钮浅蓝（B-R 56）；场上白色卡纸/棋盘为负
# 雪地/战场背景也是低 teal＋高亮＋偏蓝，单看颜色会把地图认成按钮。按钮底是整块
# 高亮填充（实测亮像素 0.736–0.741），雪地过场 0.548、战斗白闪 0.19–0.37。
FREEZE_FILL_MIN = 0.65
FREEZE_TEXT_NORMAL = "冻结"
FREEZE_TEXT_FROZEN = "解冻"
FREEZE_TEXT_MIN = 0.75                        # 冰封遮罩把「解冻」降到 0.78（1080p 0.998）
FREEZE_TEXTS = (FREEZE_TEXT_NORMAL, FREEZE_TEXT_FROZEN)


def _luma(patch: np.ndarray) -> np.ndarray:
    p = patch.astype(np.float32)
    return 0.299 * p[:, :, 0] + 0.587 * p[:, :, 1] + 0.114 * p[:, :, 2]


def _glyph_color(patch: np.ndarray):
    """徽章最亮的 12% 像素的均值色；非徽章（空位/面板）返回 None。"""
    L = _luma(patch)
    sel = L >= max(BADGE_MIN_HI, float(np.percentile(L, 88)))
    if float(sel.mean()) < BADGE_MIN_FRAC:
        return None
    px = patch[sel]
    return tuple(float(px[:, k].mean()) for k in range(3))


def _native(rgb: np.ndarray, box) -> np.ndarray:
    """按 1920 规范框直接裁原生分辨率（不缩放，比率类判据与尺度无关）。"""
    h, w = rgb.shape[:2]
    sx, sy = w / 1920.0, h / 1080.0
    x1, y1, x2, y2 = box
    return rgb[int(round(y1 * sy)):int(round(y2 * sy)),
               int(round(x1 * sx)):int(round(x2 * sx))]


def _strip_fracs(rgb) -> list[tuple[float, float]]:
    out = []
    for x in SLOT_XS:
        s = _native(rgb, (x + 3, STRIP_Y[0], x + SLOT_W - 3, STRIP_Y[1])).astype(np.int16)
        teal = ((s[:, :, 1] > 80) & (s[:, :, 2] > 60) & (s[:, :, 1] - s[:, :, 0] > 15))
        r, g, b = s[:, :, 0], s[:, :, 1], s[:, :, 2]
        gold = ((r > 120) & (g > 90) & (b < 130) & (r - b > 40))
        out.append((float(teal.mean()), float(gold.mean())))
    return out


def _art_stds(rgb) -> list[float]:
    """卡面区灰度标准差：空位（NO INFO 暗板）≈7，变暗的买不起卡 ≈15–27，正常卡 ≈40+。"""
    out = []
    for x in SLOT_XS:
        p = _native(rgb, (x + SIG_BOX[0], SIG_BOX[1], x + SIG_BOX[2], SIG_BOX[3])).astype(np.float32)
        L = 0.299 * p[:, :, 0] + 0.587 * p[:, :, 1] + 0.114 * p[:, :, 2]
        out.append(float(L.std()))
    return out


def slot_filled(rgb, art_ok: bool = False) -> list[bool]:
    """卡名条：青底或金牌实底；商店可见时再用卡面纹理兜底（买不起的卡会变暗，
    名称条整条失去青色，只看名称条会把整排卡误判成空位）。"""
    fr = _strip_fracs(rgb)
    stds = _art_stds(rgb) if art_ok else [0.0] * len(fr)
    return [teal >= STRIP_MIN or gold >= STRIP_CARD_GOLD
            or (art_ok and std >= ART_MIN_STD)
            for (teal, gold), std in zip(fr, stds)]


def triple_cards(rgb, zero) -> list[bool]:
    """三连 0 费卡：金文字名称条 + 上栏青色「0」+ 卡身是暗的。

    收起后的场上平台会把青色灌进 ZERO_BOX 和卡身区（teal 0.10–0.79），
    所以卡身 teal 必须低才算数。
    """
    out = [False] * len(SLOT_XS)
    idx = [i for i in range(len(SLOT_XS)) if zero[i]]
    if not idx:
        return out
    for i in idx:
        x = SLOT_XS[i]
        p = crop_canonical(
            rgb, (x + SIG_BOX[0], SIG_BOX[1], x + SIG_BOX[2], SIG_BOX[3]))
        out[i] = float(mask_teal(p).mean()) < TRIPLE_PANEL_TEAL_MAX
    return out


def panel_state(rgb) -> np.ndarray:
    return _grid(_luma(_native(rgb, PANEL_BOX).astype(np.float32)) > 120, rows=12, cols=16, need=8)


def ready_banner(rgb) -> bool:
    p = _native(rgb, READY_BOX).astype(np.int16)
    g = ((p[:, :, 1] > 110) & (p[:, :, 2] > 80) & (p[:, :, 1] - p[:, :, 0] > 30))
    return float(g.mean()) >= READY_MIN_FRAC


def _gold_red_fracs(rgb, box) -> tuple[float, float]:
    p = _native(rgb, box).astype(np.int16)
    r, g, b = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    gold = ((r > 140) & (g > 100) & (b < 130) & (r - b > 50))
    red = ((r > 100) & (g < 85) & (b < 95) & (r - g > 45))
    return float(gold.mean()), float(red.mean())


def card_sigs(rgb) -> list:
    """六张卡面的 16×16 归一化灰度签名（未归一化外积 = NCC）。"""
    out = []
    for x in SLOT_XS:
        p = crop_canonical(
            rgb, (x + SIG_BOX[0], SIG_BOX[1], x + SIG_BOX[2], SIG_BOX[3])
        ).astype(np.uint8)
        g = np.asarray(Image.fromarray(p).convert("L"), dtype=np.float32)
        g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize(
            (SIG_GRID, SIG_GRID), Image.BILINEAR), dtype=np.float32)
        g -= g.mean()
        out.append(g / (float(np.linalg.norm(g)) + 1e-9))
    return out


def _sig_score(cur: list, ref: list | None, slots) -> float | None:
    if ref is None:
        return None
    vals = [float((cur[i] * ref[i]).sum()) for i in slots
            if cur[i] is not None and ref[i] is not None]
    return float(np.mean(vals)) if vals else None


def glow_fracs(rgb) -> list:
    """每张卡的金色闪光占比（同名干员出现时卡面滚动金光）。"""
    out = []
    for x in SLOT_XS:
        p = crop_canonical(
            rgb, (x + SIG_BOX[0], SIG_BOX[1], x + SIG_BOX[2], SIG_BOX[3])
        ).astype(np.int16)
        r, g, b = p[:, :, 0], p[:, :, 1], p[:, :, 2]
        out.append(float(((r > 170) & (g > 130) & (b < 110) & (r - b > 80)).mean()))
    return out


def zero_flags(rgb) -> list:
    """上栏正中青色「0」= 三连奖励的 0 费卡。"""
    out = []
    for x in SLOT_XS:
        p = crop_canonical(
            rgb, (x + ZERO_BOX[0], ZERO_BOX[1], x + ZERO_BOX[2], ZERO_BOX[3])
        )
        out.append(float(mask_teal(p).mean()) >= ZERO_MIN)
    return out


def _freeze_teal_bright(rgb) -> tuple[float, float, float]:
    p = crop_canonical(rgb, FREEZE_BOX).astype(np.int16)
    teal = float(mask_teal(p.astype(np.float32)).mean())
    b = np.asarray(mask_bright(p.astype(np.float32))).astype(bool)
    bright = float(b.mean())
    blue = float((p[:, :, 2][b] - p[:, :, 0][b]).mean()) if b.any() else 0.0
    return teal, bright, blue


def _freeze_button_shape(rgb) -> bool:
    """按钮底必须是整块高亮填充：战场/雪地图案的浅蓝区域不算。

    实测（1280 录屏，ROI 含雪花图标和「冻结/解冻」）实按钮亮像素 0.736–0.741，
    雪地过场 0.548、战斗白闪 0.19–0.37。用「暗块/中灰占比」分不开：冻结态按钮
    的图标和文字让中灰占到 30%，反而比平滑雪景更"花"。
    """
    p = crop_canonical(rgb, FREEZE_BOX).astype(np.int16)
    lum = p.mean(axis=2)
    return float((lum >= 150).mean()) >= FREEZE_FILL_MIN


def _freeze_color(rgb) -> bool:
    """按钮位置的旧颜色判据：正常青底（teal≈0.6），冻结后浅蓝（bright≈0.63）。
    只用来兜底「按钮在不在」，不参与冻结状态判断。青底只认颜色（雪地 teal ≤0.07）。"""
    teal, bright, blue = _freeze_teal_bright(rgb)
    if teal >= 0.30:
        return True
    return (bright >= FREEZE_BRIGHT_MIN and blue >= FREEZE_BLUE_MIN
            and _freeze_button_shape(rgb))


def freeze_button_pale(rgb) -> bool:
    """已冻结按钮的浅白底：青底退掉、亮部占多数且偏蓝、底是整块高亮填充。

    实测正常青底 teal 0.59–0.73 / bright 0.11–0.18 / B−R 50–63，
    已冻结 teal 0.047 / bright 0.61–0.63 / B−R 54.9。
    浅蓝判据把三连页灰禁用钮（bright 0.78、B−R≈0）和白闪排除；
    它是主判据（不依赖 OCR 读字），读字只做加分确认。
    """
    teal, bright, blue = _freeze_teal_bright(rgb)
    return (teal < FREEZE_TEAL_MAX and bright >= FREEZE_BRIGHT_MIN
            and blue >= FREEZE_BLUE_MIN and _freeze_button_shape(rgb))


def freeze_text(rgb) -> str | None:
    """按钮文字：普通「冻结」，已冻结「解冻」。"""
    return read_text(rgb, FREEZE_BOX, min_score=FREEZE_TEXT_MIN)


def freeze_state(rgb) -> bool:
    """已冻结 = 按钮文字变成「解冻」。蓝色场地/白闪不会再误报。"""
    return freeze_text(rgb) == FREEZE_TEXT_FROZEN


def shop_visible(rgb) -> bool:
    """商店卡行在不在：冻结按钮（文字或颜色）在场 = 在。"""
    txt = freeze_text(rgb)
    if txt in FREEZE_TEXTS:
        return True
    return _freeze_color(rgb)


_sale_tpl = None


def _sell_btn_tpl():
    global _sale_tpl
    if _sale_tpl is None:
        path = BASE / "templates" / "sale.npz"
        _sale_tpl = _unit(np.load(path)["sell_btn"]) if path.exists() else False
    return _sale_tpl


def _sell_button(rgb) -> bool:
    """金色闸门先挡掉绝大多数帧，再过形状（金色建筑/联防横幅颜色几乎一样）。"""
    g, _ = _gold_red_fracs(rgb, SELL_BTN_BOX)
    if g < SELL_BTN_MIN:
        return False
    tpl = _sell_btn_tpl()
    if tpl is False:
        return True
    p = crop_canonical(rgb, SELL_BTN_BOX).astype(np.float32)
    return _ncc(_unit(_luma(p)), tpl) >= SELL_BTN_NCC


def sale_state(rgb) -> str | None:
    """售卖/装备 UI：'button'=场上出售按钮，'zone'=右下拉拽售出，
    'destroy'=右下拉拽销毁（装备），None=无。

    这三种界面都会盖住商店卡行，且装备使用也可能金币上涨（用户 2026-11-12
    指出），所以售出必须由这里的独立标记背书，不能只看金币增加。
    """
    if _sell_button(rgb):
        return "button"
    g, r = _gold_red_fracs(rgb, SALE_ZONE_BOX)
    if r >= SALE_ZONE_RED_MIN:
        return "destroy"
    if g >= SALE_ZONE_GOLD_MIN:
        return "zone"
    return None


def _grid(m: np.ndarray, rows: int, cols: int, need: int) -> np.ndarray:
    h, w = m.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    return np.array([[m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum() >= need
                      for j in range(cols)] for i in range(rows)])


def _unit(x: np.ndarray) -> np.ndarray:
    v = x.astype(np.float32)
    v = v - v.mean()
    return v / (float(np.linalg.norm(v)) + 1e-9)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def _unit_masked(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = x[mask].astype(np.float32)
    v = v - v.mean()
    return v / (float(np.linalg.norm(v)) + 1e-9)


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """只在 mask=True 的像素上做 NCC：牌子的等级/费用/按钮会变，牌框不变。"""
    x, y = a[mask].astype(np.float32), b[mask].astype(np.float32)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-9 else 0.0


class BadgeReader:
    def __init__(self, path: Path = TPL):
        self.tpl = {k: _unit(v) for k, v in np.load(path).items()}

    def read(self, rgb, slot: int) -> str | None:
        x = SLOT_XS[slot]
        patch = crop_canonical(
            rgb, (x + BADGE_BOX[0], BADGE_BOX[1], x + BADGE_BOX[2], BADGE_BOX[3])
        ).astype(np.float32)
        color = _glyph_color(patch)
        if color is None:
            return None
        r, g, b = color
        # 四个模板全算：颜色给的族之外还要看形状会不会压倒它
        scores = sorted(((_ncc(_unit(_luma(patch)), self.tpl[k]), k)
                         for k in self.tpl), reverse=True)
        best, label = scores[0]
        if b - r >= BADGE_COLOR_MIN:          # 青/绿牌：颜色即等级，形状只做否决
            if (label in ("I", "II") and best >= BADGE_SHAPE_MIN
                    and best - scores[1][0] >= BADGE_FAMILY_MARGIN):
                return label                  # 白牌被青绿光染色
            return "III" if g - b > BADGE_HUE_SPLIT else "IV"
        # 白牌(II 与金 V 的灰度相关性高达 0.8)必须先按颜色分族，族内再比形状
        fam = ("V", "VI") if r - b >= BADGE_GOLD_MIN else ("I", "II")
        if label not in fam:
            other = max(s for s, k in scores if k in fam)
            if best < BADGE_SHAPE_MIN or best - other < BADGE_FAMILY_MARGIN:
                return None                   # 颜色与形状打架：不猜，下帧再看
            return label
        if best < BADGE_MIN_SCORE:
            return None
        fam_scores = [s for s, k in scores if k in fam]
        if len(fam_scores) > 1 and best - fam_scores[1] < BADGE_MIN_MARGIN:
            return None
        return label


class ItemReader:
    """道具卡左下角三圆徽记（二值形状比 NCC）。"""

    def __init__(self, pack: str = str(ITEM_PACK)):
        self.tpl = []
        for n in (1, 2):
            path = Path(pack.format(n=n))
            if path.exists():
                rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
                self.tpl.append(_unit((_luma(rgb) > 150).astype(np.float32)))

    def read(self, rgb, slot: int) -> bool:
        if not self.tpl:
            return False
        x = SLOT_XS[slot]
        patch = crop_canonical(
            rgb, (x + ITEM_BOX[0], ITEM_BOX[1], x + ITEM_BOX[2], ITEM_BOX[3])
        ).astype(np.float32)
        # 亮卡固定 150；装备已满时整卡变暗，三圆徽记峰值只到 150（NCC 0.16），
        # 按 72% 分位压阈值才读得到（0.93），而亮卡仍用固定阈值保住 margin。
        L = _luma(patch)
        th = min(150.0, max(60.0, 0.72 * float(np.percentile(L, 99))))
        m = (L > th).astype(np.float32)
        return max(_ncc(m, t) for t in self.tpl) >= ITEM_MIN_SCORE


_plate_tpl = None
_plate_vec = None

# 等级牌里会变的内容：费用圈、LEVEL 字样、等级数字、底部按钮（升级/确认升级）。
# 掩码只留牌框与斜纹底，等级/状态怎么变都不影响「牌子在不在」。
_PLATE_TEXT_BOXES = {
    "slot": ((84, 0, 140, 48), (76, 56, 166, 94),
             (94, 96, 156, 160), (76, 166, 162, 212)),
    "compact": ((45, 0, 108, 36), (42, 62, 108, 84),
                (42, 86, 108, 142), (36, 170, 112, 206)),
}


def _plate_mask(key: str, w: int) -> np.ndarray:
    m = np.ones((218, w), dtype=bool)
    fam = "compact" if key.startswith("compact") else "slot"
    for x1, y1, x2, y2 in _PLATE_TEXT_BOXES[fam]:
        a, b = max(0, x1), min(w, x2)
        if b > a:
            m[y1:y2, a:b] = False
    return m


def _plate_templates():
    global _plate_tpl
    if _plate_tpl is None:
        _plate_tpl = ({k: _unit(v) for k, v in np.load(LEVEL_PACK).items()}
                      if LEVEL_PACK.exists() else {})
    return _plate_tpl


def _plate_vectors():
    global _plate_vec
    if _plate_vec is None:
        _plate_vec = {}
        for k, v in _plate_templates().items():
            m = _plate_mask(k, v.shape[1])
            _plate_vec[k] = (_unit_masked(v, m), m)
    return _plate_vec


def _plate_score(patch: np.ndarray, keys) -> float:
    L = _luma(patch)
    best = 0.0
    for k in keys:
        vec, mask = _plate_vectors()[k]
        best = max(best, float((_unit_masked(L, mask) * vec).sum()))
    return best


def _green_frac(patch: np.ndarray) -> float:
    p = patch.astype(np.int16)
    return float(((p[:, :, 1] > 110) & (p[:, :, 2] > 80)
                  & (p[:, :, 1] - p[:, :, 0] > 30)).mean())


def _level_sig(patch: np.ndarray) -> np.ndarray | None:
    """等级数字的列投影签名：数字被遮暗/动画盖住时返回 None（这一帧不参与比较）。"""
    L = _luma(patch)
    m = L >= max(50.0, 0.72 * float(np.percentile(L, 99)))
    if float(m.mean()) < 0.01:
        return None
    col = m.sum(axis=0).astype(np.float32)
    if col.size != LEVEL_SIG_BINS:
        col = np.interp(np.linspace(0, col.size - 1, LEVEL_SIG_BINS),
                        np.arange(col.size), col)
    v = col - col.mean()
    return v / (float(np.linalg.norm(v)) + 1e-9)


def _numeral(patch: np.ndarray) -> int | None:
    """罗马数字 I–VI：按柱投影切字，窄竖= I，宽 V= V；组合即等级。

    实测 I 11 px、V 46 px；III/IV 没有录屏样本，按笔画组合推定（见模块文档）。
    """
    L = _luma(patch)
    if L.size == 0:
        return None
    m = L >= max(60.0, 0.72 * float(np.percentile(L, 99)))
    cols = m.sum(axis=0)
    runs, st = [], None
    for i, v in enumerate(cols > 0):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((st, i))
            st = None
    if st is not None:
        runs.append((st, len(cols)))
    widths = []
    for a, b in runs:
        sub = m[:, a:b]
        rows = np.nonzero(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        if rows.max() - rows.min() + 1 >= NUMERAL_MIN_H and b - a >= NUMERAL_MIN_W:
            widths.append(int(b - a))
    narrow = [w <= NUMERAL_NARROW for w in widths]
    wide = [w >= NUMERAL_WIDE for w in widths]
    if len(widths) == 1:
        if narrow[0]:
            return 1
        return 5 if wide[0] else None
    if len(widths) == 2:
        if all(narrow):
            return 2
        if wide[0] and narrow[1]:
            return 6
        if narrow[0] and wide[1]:
            return 4
        return None
    if len(widths) == 3 and all(narrow):
        return 3
    return None


def level_state(rgb):
    """LEVEL 牌的（是否在，是否绿态，位置键，等级数字列投影签名，罗马数字等级）。

    位置键 = "compact"（贴行首）或牌所在的卡位下标。四/五/六卡以及以后 1–3 卡
    都只是这张 150 宽牌贴在第一个占用卡位的右缘，所以六点扫一套模板就够。
    等级数字只在牌面 NCC ≥ NUMERAL_MIN_SCORE 时给出，缩放动画帧一律 None。
    """
    tpl = _plate_templates()
    if not tpl or "compact_gray" not in tpl:
        return False, False, None, None, None
    best, key, plate = 0.0, None, None
    for x, k in zip(PLATE_XS, PLATE_KEYS):
        p = crop_canonical(rgb, (x, 812, x + 150, 1030)).astype(np.float32)
        score = _plate_score(p, PLATE_TPL_KEYS)
        if score >= PLATE_MIN and score > best:
            best, key, plate = score, k, p
    if key is None:
        return False, False, None, None, None
    x1, y1, x2, y2 = PLATE_NUM_BOX
    sig = _level_sig(plate[y1:y2, x1:x2])
    nx1, ny1, nx2, ny2 = NUMERAL_BOX
    num = (_numeral(plate[ny1:ny2, nx1:nx2]) if best >= NUMERAL_MIN_SCORE
           else None)
    return True, _green_frac(plate) >= PLATE_GREEN_MIN, key, sig, num

@dataclass
class ShopTracker:
    settle: float = 1.0

    reader: BadgeReader = field(default_factory=BadgeReader)
    items: ItemReader = field(default_factory=ItemReader)
    # 金币数字变色/闪光约 0.5s 读不出；放宽掉检容忍，避免整段读数被清空
    gold: GoldTracker = field(default_factory=lambda: GoldTracker(miss=6))

    prev_filled: list | None = None
    prev_gold: int | None = None
    prev_panel: np.ndarray | None = None
    tier_cache: dict = field(default_factory=dict)
    item_cache: dict = field(default_factory=dict)     # 卡位 → 是道具卡（购买时不报干员）
    tier_item: dict = field(default_factory=dict)      # 读档位同一帧的道具徽记（冻结证据）
    pending: dict = field(default_factory=dict)   # 刚变空的卡位 → {since, tier, item, gold}
    level_ref: np.ndarray | None = None           # 等级数字列投影基线
    level_key: object = None                      # 牌的位置键（"compact" 或卡位下标）
    level_cand: np.ndarray | None = None          # 与基线不同的新数字（下一帧确认）
    level_cand_num: int | None = None             # 候选帧读到的罗马数字
    level_last_t: float = -1e9                    # 上次报升级的时刻
    level: int | None = None                      # 调度中心等级（罗马数字，1–6）
    level_read_cand: int | None = None            # 等级读数候选（连续同读数帧数用它确认）
    level_read_count: int = 0                     # 连续同读数帧数
    green_since: float | None = None              # 绿牌（可升级）开始时刻
    can_upgrade_fired: bool = False               # 本轮休整已提示过可升级
    plate_vacated: int | None = None              # 等级牌让出的卡位：下一帧变空不是购买
    plate_vacated_t: float = -1e9
    refresh_streak: int = 0                       # 连续刷新次数（购买会打断）
    refresh_egg: int = REFRESH_STREAK_EGG          # 连续刷新多少次触发彩蛋（可调）
    refresh_egg_fired: bool = False               # 本轮连续刷新已播过彩蛋
    gold_drop_t: float = -1e9                     # 最近一次金币减少（刷新扣款/购买/升级）
    cards_changed_t: float = -1e9                 # 最近一次卡位增减（卡位读数可信窗口）
    shop_since: float | None = None
    visible: bool = False                         # 上一帧商店是否在场（Bridge 用来判闲置）
    sale_ui_t: float = -1e9                       # 最近一次看到「出售」按钮的时刻
    ready_fired: bool = False

    sig_ref: list | None = None                   # 卡面内容基线
    sig_cand: list | None = None                  # 候选新内容（稳定后才替换基线）
    sig_since: float | None = None                # 内容开始与基线不同的时刻
    sig_mute_until: float = 0.0                   # 战斗进入/商店开场的静默截止
    freeze_on: bool = False                       # 冻结按钮 latch（只报上升沿）
    freeze_since: float | None = None             # 冻结态开始时刻
    freeze_miss_t: float | None = None            # 冻结读数首次读空的时刻（容忍计时）
    freeze_normal_since: float | None = None      # 普通态开始时刻（用于解 latch）
    freeze_last_seen: float | None = None         # 冻结态最后一次被确认的时刻（解冻播报门槛）
    glow_since: float | None = None               # 金光开始时刻
    glow_count: int = 0                           # 连续过阈值帧数
    glow_on: bool = False                         # 同名闪光 latch
    glow_last_t: float = -1e9                     # 上次同名语音时刻
    triple_on: bool = False                       # 三连奖励界面 latch
    triple_since: float | None = None             # 三连「0」费连续出现时刻
    triple_normal_since: float | None = None      # 普通商店连续出现时刻
    triple_until: float = 0.0                     # 三连 180s latch
    triple_sig_ref: list | None = None            # 三连三张卡的内容基线
    triple_sig_since: float | None = None
    same_op_armed: bool = False                   # 三连页内报过同名 → 退出/更新时丢弃

    def update(self, rgb, t: float, rest: bool = True) -> list[dict]:
        if not rest:
            self._reset()
            return []

        ev: list[dict] = []
        # 准备横幅的 ROI 落在卡行区域：休整时卡面加载/冰封遮罩会把这段灌成绿色。
        # 冻结按钮（文字或颜色）还在场就说明商店没结束，横幅读数不可信。
        if ready_banner(rgb) and not shop_visible(rgb):
            if not self.ready_fired:
                self.ready_fired = True
                ev.append({"type": "ready", "t": t})
            # 横幅压在名称条带上方，读数不可信；商店已结束，直接清状态
            self._clear_shop(triple=False)
            return ev
        self.ready_fired = False

        state = sale_state(rgb)
        if state is not None:
            # 详情面板/拖拽遮罩盖住卡行，卡位读数不可信；退出后重新初始化
            if state == "button":          # 只有干员「出售」按钮给售出背书
                self.sale_ui_t = t
            self.prev_filled = None
            self.pending.clear()
            self._clear_cards()
            self.sig_ref = None
            self.sig_since = None
            self.glow_on = False
            self.glow_since = None
            self.glow_count = 0
            if state == "button":          # 只有按钮模式下金币计数器还露着
                raw = read_gold(rgb)
                if raw is None:
                    raw = read_gold_ocr(rgb)
                gold = self.gold.update(raw)
                if gold is not None:
                    if self.prev_gold is not None and gold > self.prev_gold:
                        ev.append({"type": "sell", "t": t})
                    self.prev_gold = gold
            return ev

        # 冻结按钮文字一次 OCR，供「商店可见」与「已冻结」共用
        ftxt = freeze_text(rgb)
        vis = ftxt in FREEZE_TEXTS or _freeze_color(rgb)
        self.visible = vis
        filled = slot_filled(rgb, art_ok=vis)
        zero = zero_flags(rgb)
        tc = triple_cards(rgb, zero)
        if not all(filled):
            filled = [a or b for a, b in zip(filled, tc)]
        triple_vis = sum(tc[i] for i in TRIPLE_SLOTS) >= TRIPLE_MIN_CARDS
        raw_gold = read_gold(rgb)
        if raw_gold is None and vis:
            raw_gold = read_gold_ocr(rgb)     # 数字闪光时模板漏读，OCR 兜底
        gold = self.gold.update(raw_gold)
        panel = panel_state(rgb)
        prev = self.prev_filled
        cleared = ([i for i in range(len(filled)) if prev[i] and not filled[i]]
                   if prev is not None else [])
        refilled = ([i for i in range(len(filled)) if filled[i] and not prev[i]]
                    if prev is not None else [])
        if cleared:
            # 只有「卡位变空」才是买卡；补位/换位（升级后 LEVEL 牌让出卡位）不算
            self.cards_changed_t = t
        if gold is not None and self.prev_gold is not None and gold < self.prev_gold:
            self.gold_drop_t = t

        # 升级调度中心三态：灰牌（买不起）→ 绿牌（可升级）→ 罗马数字上升（完成）。
        # 绿态只代表「买得起」；满级 VI 的牌面仍是绿底「已满级」，靠等级数字压掉。
        # 升级只认 LEVEL 数字本身：同级列投影 ≥0.94，I→II −0.16；数字连续两帧
        # 同签名、且罗马数字确实变大才报。确认后取消不改数字 → 无声。
        found, green, key, lvl_sig, lvl_num = level_state(rgb)
        if found:
            if lvl_num is not None:
                if lvl_num == self.level_read_cand:
                    self.level_read_count += 1
                else:
                    self.level_read_cand = lvl_num
                    self.level_read_count = 1
                if self.level_read_count >= NUMERAL_CONFIRM:
                    # 等级只作为内部基线（升级语音只认上升），不再每次休整播报
                    if self.level is None:
                        self.level = lvl_num
                    elif lvl_num < self.level:
                        # 没看到新局入口或低读：静默跟上，升级语音只认上升
                        self.level = lvl_num
            if green:
                if self.green_since is None:
                    self.green_since = t
                elif (self.level is not None and self.level < LEVEL_MAX
                      and not self.can_upgrade_fired
                      and t - self.green_since >= GREEN_HOLD):
                    self.can_upgrade_fired = True
                    ev.append({"type": "can_upgrade", "t": t})
            else:
                self.green_since = None

        if found and lvl_sig is not None:
            if self.level_ref is None:
                self.level_ref = lvl_sig
                self.level_key = key
                if self.level is None and lvl_num is not None:
                    self.level = lvl_num
            else:
                # 换牌位不动基线：同数字在两种底纹上实测 0.975，照常比；
                # 移动那一帧常带升级动画（箭头盖住数字），下一帧若同数字会自然回到基线
                if key != self.level_key:
                    if isinstance(self.level_key, int):
                        # 牌让出的卡位下一帧是 NO INFO：升级/重排不是买卡
                        self.plate_vacated = self.level_key
                        self.plate_vacated_t = t
                        # 空位可能先于牌位移动被扫到：这一刻起挂起的购买作废
                        self.pending.pop(self.level_key, None)
                    self.level_key = key
                    self.level_cand = None
                    self.level_cand_num = None
                if float((lvl_sig * self.level_ref).sum()) >= LEVEL_SIG_MIN:
                    self.level_ref = lvl_sig
                    self.level_cand = None
                    self.level_cand_num = None
                elif (self.level_cand is not None
                      and float((lvl_sig * self.level_cand).sum()) >= LEVEL_SIG_MIN):
                    new_num = lvl_num if lvl_num is not None else self.level_cand_num
                    self.level_cand = None
                    self.level_cand_num = None
                    self.level_ref = lvl_sig
                    if new_num is not None and self.level is not None:
                        if new_num > self.level:
                            if t - self.level_last_t >= LEVEL_COOLDOWN:
                                self.level_last_t = t
                                self.level = new_num
                                # 升级动画会让卡位短暂变空：挂起的购买作废
                                self.pending.clear()
                                ev.append({"type": "level_up", "t": t})
                        else:
                            # 数字没升：牌面重排/误读，不报升级
                            self.level = new_num
                else:
                    self.level_cand = lvl_sig
                    self.level_cand_num = lvl_num

        if (found and lvl_num is not None
                and self.level_read_count >= NUMERAL_CONFIRM
                and self.level is not None and lvl_num > self.level
                and not any(e["type"] == "level_up" for e in ev)):
            # 等级在牌看不到的时候涨过（升级帧没读上）：静默跟上，不补播升级
            self.level = lvl_num

        if not any(filled) and not self.triple_on and not vis:
            # 商店收起：底排是场上干员，卡位读数作废；冻结/三连状态要保持（用户口径）
            self._clear_shop(triple=True)
            return ev

        if self.shop_since is None or self.prev_filled is None:
            if self.prev_filled is None:      # 出售 UI 结束后的首帧：补看跨界的金币上涨
                if (gold is not None and self.prev_gold is not None
                        and gold > self.prev_gold
                        and t - self.sale_ui_t <= SELL_UI_WINDOW):
                    ev.append({"type": "sell", "t": t})
            self.shop_since = t
            self.prev_filled = filled
            self.prev_gold = gold if gold is not None else self.prev_gold
            self.prev_panel = panel
            self._cache_badges(rgb, filled)
            self._reset_sig(rgb)
            # 商店刚出现：卡面淡入/飞卡也会改签名，先静默再判刷新
            self.sig_mute_until = max(self.sig_mute_until, t + SIG_OPEN_MUTE)
            return ev

        prev = self.prev_filled
        if (sum(prev) < 3 or sum(filled) < 3) and not triple_vis and not self.triple_on:
            # 只有零星卡片的过渡帧（卡行滑动/图片未加载）：基线保留，等落定再比；
            # 已挂起的购买仍要确认，否则读数不稳的商店里买卡全丢
            self._confirm_pending(ev, filled, gold, raw_gold, t)
            self.shop_since = t
            self.prev_filled = filled
            if gold is not None:
                self.prev_gold = gold
            self.prev_panel = panel
            self._clear_cards()
            return ev

        # 售出：金币上涨，且 2.5s 内见过干员专属金色「出售」按钮
        if (gold is not None and self.prev_gold is not None and gold > self.prev_gold
                and t - self.sale_ui_t <= SELL_UI_WINDOW):
            ev.append({"type": "sell", "t": t})

        cleared = [i for i in range(len(filled)) if prev[i] and not filled[i]]
        refilled = [i for i in range(len(filled)) if filled[i] and not prev[i]]
        wholesale = len(cleared) + len(refilled) >= 3
        rebuild = wholesale or len(cleared) >= 2 or (bool(cleared) and bool(refilled))
        # 面板里有经验条/金币飞出动画：小额差异算同一状态，只有整体换位才算变了
        panel_same = (self.prev_panel is not None
                      and int(np.abs(panel.astype(np.int8) - self.prev_panel.astype(np.int8)).sum())
                      <= PANEL_SAME_CELLS)

        # 卡位变空先进待定：持续为空 settle 秒、且金币确实减少，才算购买。
        # 不要求 panel_same：买卡后左下详情面板/LEVEL 牌换位是常态（买了第 3 张
        # 后行首变成升级牌），旧口径会把这种真实购买全部丢掉。
        # 先建 pending 再清等级/道具缓存，否则重排帧会把刚买的道具卡记成干员。
        # 一次只挂一张：同帧 ≥2 格变空是升级/刷新的重排与 NO INFO 加载（实测
        # a_0186 升级后 1、2 格同时空）；已有 pending 时再来的空缺不覆盖，
        # 避免「等级牌空位 + 真购买」串成一条错误的高阶干员语音。
        if self.plate_vacated is not None and t - self.plate_vacated_t > 1.0:
            self.plate_vacated = None     # 让出的卡位迟迟没变空：不留着误伤后面的买卡
        if len(cleared) == 1 and cleared[0] != self.plate_vacated and not self.pending:
            i = cleared[0]
            item = self.item_cache.get(i, False) or self.tier_item.get(i, False)
            tier = self.tier_cache.get(i)
            # 没档位也没道具的 pending 不会产生任何语音，只会挡住下一张真购买
            if item or tier is not None:
                self.pending[i] = {"since": t, "tier": tier, "item": item,
                                   "gold": self.prev_gold}
        if self.plate_vacated in cleared:
            # LEVEL 牌让出的卡位：缓存里的档位/道具属于更早那张卡，不算购买
            self.pending.pop(self.plate_vacated, None)
            self.tier_cache.pop(self.plate_vacated, None)
            self.item_cache.pop(self.plate_vacated, None)
            self.tier_item.pop(self.plate_vacated, None)
            self.plate_vacated = None

        if rebuild:
            self._clear_cards()
            # 不能清 pending：买卡后 LEVEL 牌让出卡位也会造成「整排重排」，
            # 清掉就把刚买的那张连同升级语音一起吞了（只有该卡位重新有卡才取消）
        elif (gold is not None and self.prev_gold is not None and gold < self.prev_gold
                and not cleared and not refilled and panel_same):
            # 金币下降但卡位没变：刷新/升级动画，等级缓存先作废；是否算刷新看签名
            self._clear_cards()

        self._confirm_pending(ev, filled, gold, raw_gold, t)

        new_sigs = card_sigs(rgb)
        glow = glow_fracs(rgb)

        # ---- 三连奖励：右侧三卡上栏青色「0」，确认 0.3s ----
        triple_normal = (sum(filled) >= 3
                         and not any(zero[i] for i in TRIPLE_SLOTS))
        if triple_vis:
            self.triple_normal_since = None
            if not self.triple_on:
                if self.triple_since is None:
                    self.triple_since = t
                if t - self.triple_since >= TRIPLE_CONFIRM and t >= self.triple_until:
                    self.triple_on = True
                    self.triple_since = None
                    self.triple_until = t + TRIPLE_COOLDOWN
                    self.triple_sig_ref = new_sigs
                    self.triple_sig_since = None
                    self.same_op_armed = False
                    ev.append({"type": "triple_reward", "t": t})
            else:
                self.triple_since = None
                score = _sig_score(new_sigs, self.triple_sig_ref, TRIPLE_SLOTS)
                if score is not None and score < SIG_MIN_NCC:
                    if self.triple_sig_since is None:
                        self.triple_sig_since = t
                    elif t - self.triple_sig_since >= TRIPLE_CONFIRM:
                        # 三连页内三张卡换人 = 新三连（连锁），重播并重置冷却
                        self.triple_sig_since = None
                        self.triple_sig_ref = new_sigs
                        self.triple_until = t + TRIPLE_COOLDOWN
                        self._drop_same_op(ev, t)
                        ev.append({"type": "triple_reward", "t": t})
                else:
                    self.triple_sig_since = None
        else:
            self.triple_since = None
            self.triple_sig_since = None
            if self.triple_on:
                if triple_normal:
                    if self.triple_normal_since is None:
                        self.triple_normal_since = t
                    elif t - self.triple_normal_since >= TRIPLE_CONFIRM:
                        # 回到可见的普通商店：清除冷却
                        self.triple_on = False
                        self.triple_normal_since = None
                        self.triple_until = 0.0
                        self._drop_same_op(ev, t)
                else:
                    self.triple_normal_since = None

        # ---- 刷新：卡位没变、整排卡面换掉且稳定 SIG_STABLE ----
        if not self.triple_on and not cleared and not refilled:
            if t < self.sig_mute_until:
                self.sig_ref = new_sigs
                self.sig_since = None
            else:
                score = _sig_score(new_sigs, self.sig_ref,
                                   [i for i in range(6) if filled[i]])
                if any(e["type"] == "level_up" for e in ev):
                    # 升级会改卡面，基线跟着走，不能当成刷新
                    self.sig_ref = new_sigs
                    self.sig_since = None
                elif score is not None and score < SIG_MIN_NCC:
                    # 候选内容必须自己稳定 SIG_STABLE（防止逐张翻牌的动画被当成多次刷新）
                    settle = _sig_score(new_sigs, self.sig_cand,
                                        [i for i in range(6) if filled[i]])
                    if self.sig_since is None or settle is None or settle < SIG_MIN_NCC:
                        self.sig_since = t
                        self.sig_cand = new_sigs
                    elif t - self.sig_since >= SIG_STABLE:
                        self.sig_since = None
                        self.sig_cand = None
                        self.sig_ref = new_sigs
                        # 刷新一定扣 1 金币：没扣款的整排换卡是卡面加载/换回合
                        # （第 10 回合刚进休整就播刷新彩蛋的误报来源）
                        if t - self.gold_drop_t <= REFRESH_GOLD_WINDOW:
                            # 真刷新会整排换代：之前挂起的空缺不能再算购买
                            self.pending.clear()
                            # 换出来的新卡必须重读等级，否则沿用上一排的 tier
                            # （一阶卡被报成高阶的主要来源）
                            self._clear_cards()
                            self.refresh_streak += 1
                            # 刷新清除同名冷却（金光还在就等下一帧确认重新触发）
                            self.glow_on = False
                            self.glow_count = 1 if self._glow_lit(
                                rgb, glow, filled) else 0
                            self.glow_since = t if self.glow_count else None
                            self.glow_last_t = -1e9
                            if (self.refresh_streak >= self.refresh_egg
                                    and not self.refresh_egg_fired):
                                self.refresh_egg_fired = True
                                ev.append({"type": "refresh", "t": t})
                else:
                    self.sig_since = None
                    self.sig_cand = None
        elif cleared or refilled:
            self._reset_sig(rgb, new_sigs)
            # 卡行重排/补人的动画会改签名，静默到动画结束再判刷新
            self.sig_mute_until = max(self.sig_mute_until, t + SIG_REBUILD_MUTE)

        # ---- 冻结按钮：报上升沿（冻结）和回落沿（解冻） ----
        # 主判据是按钮本身的浅蓝底（不依赖 OCR）；「解冻」文字只做加分确认。
        # 三连页冻结按钮是灰禁用态（teal 0 / bright 0.78 / B−R≈0），浅蓝判据天然排除；
        # 卡位读数不完整时只要商店还在（冻结按钮可见）也照样判
        # 冰封遮罩会把卡面压暗、卡位读数掉到 3 张以下，所以浅蓝底自己也必须
        # 能单独开闸，否则「卡位读不全 + OCR 读不出解冻」时整段冻结无人认领
        pale = freeze_button_pale(rgb)
        if not self.triple_on and (sum(filled) >= 3 or vis or (pale and rest)):
            if pale or ftxt == FREEZE_TEXT_FROZEN:
                self.freeze_last_seen = t
                self.freeze_miss_t = None
                self.freeze_normal_since = None
                if not self.freeze_on:
                    if self.freeze_since is None:
                        self.freeze_since = t
                    # 浅蓝底单帧即可；只有文字时仍要持续 FREEZE_CONFIRM
                    if pale or t - self.freeze_since >= FREEZE_CONFIRM:
                        self.freeze_since = None
                        self.freeze_on = True
                        ev.append({"type": "freeze", "t": t})
            elif ftxt is None:
                # live 的 PrintWindow 帧文字锚点全程读空，颜色是唯一锚点。
                # 待确认期读空不清计时（冰封遮罩下「解冻」会掉到 0.78）；
                # 已冻结后青底回来（vis 且非 pale）就是解冻，不能再等文字。
                if not self.freeze_on:
                    if self.freeze_since is None:
                        self.freeze_miss_t = None
                    elif t - self.freeze_since >= FREEZE_CONFIRM:
                        # 确认时长已到、最后一帧只是掉检：照样算冻结
                        self.freeze_since = None
                        self.freeze_miss_t = None
                        self.freeze_on = True
                        ev.append({"type": "freeze", "t": t})
                    elif self.freeze_miss_t is None:
                        self.freeze_miss_t = t
                    elif t - self.freeze_miss_t > FREEZE_MISS:
                        self.freeze_since = None
                        self.freeze_miss_t = None
                elif not pale and vis:
                    self._freeze_normal(ev, t)
                # 按钮整个不在场（商店收起/面板遮罩）→ 保持冻结态
            else:
                self._freeze_normal(ev, t)

        # ---- 同名干员：金光连续两帧确认后触发；冷却期内每 15s 重播一次 ----
        # 三连页只加载出两张候选时也要判（用户口径：待选里已有同名就报）；
        # 同名装备滚的是同一套光，按道具徽记排除；
        # 必须真的在商店里且过了开场/重排静默：策略页等金色 UI 也会灌满卡面区
        if sum(filled) >= 2 and vis and t >= self.sig_mute_until:
            lit = self._glow_lit(rgb, glow, filled)
            if lit:
                if self.glow_since is None:
                    self.glow_since = t
                    self.glow_count = 1
                else:
                    self.glow_count += 1
                confirmed = (self.glow_count >= GLOW_CONFIRM
                             and t - self.glow_since >= GLOW_MIN_SPAN)
                if confirmed:
                    self.glow_on = True
                # 冷却跨「光灭→再亮」的脉冲：同样闪一次不会隔几秒又报一遍
                if confirmed and t - self.glow_last_t >= SAME_OP_COOLDOWN:
                    self.glow_last_t = t
                    ev.append({"type": "same_operator", "t": t})
                    if self.triple_on:
                        self.same_op_armed = True
            else:
                self.glow_since = None
                self.glow_count = 0
                self.glow_on = False
        else:
            # 商店收起/开场静默期间不累计，避免把两次闪光拼成一次
            self.glow_since = None
            self.glow_count = 0

        self._cache_badges(rgb, filled)

        self.prev_filled = filled
        if gold is not None:      # 读数闪断时保住基准，否则跨帧的买卖都判不出来
            self.prev_gold = gold
        self.prev_panel = panel
        return ev

    def on_run_start(self):
        """新一局：调度中心从 I 级开始，等级/播报状态全部重来。"""
        self.level = None
        self.on_rest_enter()

    def on_rest_enter(self):
        """一轮休整开始：可升级提醒只播一次。

        商店在读帧时会闪成「收起」（选中卡/面板/动画），不能用它切分休整，
        否则同一轮休整会反复提示。
        """
        self.level_read_cand = None
        self.level_read_count = 0
        self.green_since = None
        self.can_upgrade_fired = False

    def on_battle_enter(self, t: float):
        """进入战斗：商店状态全部作废；刷新基线静默 3s 等过场动画。"""
        self._clear_shop(triple=False)
        self.ready_fired = False
        self.prev_gold = None
        self.sig_mute_until = t + SIG_MUTE

    def _confirm_pending(self, ev: list, filled, gold, raw_gold, t: float):
        """卡位持续为空 + 金币确实减少 = 购买完成（金币读数闪断时退用原始读数）。"""
        now = gold if gold is not None else raw_gold
        for i in list(self.pending):
            if filled[i]:                       # 卡又回来了 = 抖动，取消
                del self.pending[i]
            elif t - self.pending[i]["since"] >= self.settle:
                info = self.pending.pop(i)
                if now is not None and info["gold"] is not None and now < info["gold"]:
                    if info["item"]:
                        ev.append({"type": "buy_item", "t": t})
                    elif info["tier"] is not None:
                        # 升级牌从卡位移到收纳位也会空出一个卡位：没档位没语音，不报
                        ev.append({"type": "purchase", "t": t, "tier": info["tier"]})
                    self.tier_cache.pop(i, None)
                    self.item_cache.pop(i, None)
                    self.refresh_streak = 0
                    self.refresh_egg_fired = False

    def _glow_lit(self, rgb, glow, filled) -> bool:
        """有干员卡在滚金光：道具卡、整卡金牌实底都不算。

        同名闪光只罩卡面（实测占比 0.19–0.23）；协防干员等特卡/悬停高亮
        会整卡镀金（0.54+），名称条也变金，按上限剔掉。
        道具徽记在闪光/变暗帧会读丢，所以一旦某卡位被认成道具就锁住（直到卡位换人）。
        """
        return any(GLOW_MIN <= g <= GLOW_MAX
                   and not (self.item_cache.get(i) or self.items.read(rgb, i))
                   for i, g in enumerate(glow) if filled[i])

    def _drop_same_op(self, ev: list, t: float):
        """三连页退出/换卡：丢弃还没播的「同名干员」语音。"""
        if self.same_op_armed:
            self.same_op_armed = False
            ev.append({"type": "voice_drop", "t": t,
                       "category": "same_operator"})

    def _reset_sig(self, rgb=None, sigs=None):
        self.sig_ref = sigs if sigs is not None else (card_sigs(rgb) if rgb is not None else None)
        self.sig_since = None

    def _clear_cards(self):
        self.tier_cache.clear()
        self.item_cache.clear()
        self.tier_item.clear()

    def _clear_shop(self, triple: bool = False):
        self.prev_filled = None
        self.prev_gold = None
        self.prev_panel = None
        self._clear_cards()
        self.pending.clear()
        self.level_ref = None
        self.level_key = None
        self.level_cand = None
        self.level_cand_num = None
        self.plate_vacated = None
        self.shop_since = None
        self.refresh_streak = 0
        self.refresh_egg_fired = False
        self.sig_ref = None
        self.sig_cand = None
        self.sig_since = None
        self.glow_on = False
        self.glow_since = None
        self.glow_count = 0
        self.freeze_since = None
        self.freeze_miss_t = None
        self.freeze_normal_since = None
        self.freeze_last_seen = None
        self.visible = False
        if not triple:
            self.freeze_on = False
            self.glow_last_t = -1e9
            self.triple_on = False
            self.triple_since = None
            self.triple_normal_since = None
            self.triple_until = 0.0
            self.triple_sig_ref = None
            self.triple_sig_since = None
            self.same_op_armed = False

    def _freeze_normal(self, ev, t):
        """按钮回到普通态：持续 FREEZE_RESET 后解除冻结 latch，并报一次解冻。"""
        self.freeze_miss_t = None
        self.freeze_since = None
        if not self.freeze_on:
            self.freeze_normal_since = None
            return
        if self.freeze_normal_since is None:
            self.freeze_normal_since = t
            return
        if t - self.freeze_normal_since < FREEZE_RESET:
            return
        self.freeze_on = False
        self.freeze_normal_since = None
        # 只有本轮商店里刚看过冻结态才报解冻；商店收起/换轮时的复位是静默的
        if (self.freeze_last_seen is not None
                and t - self.freeze_last_seen <= FREEZE_ONLINE):
            ev.append({"type": "unfreeze", "t": t})
        self.freeze_last_seen = None

    def _cache_badges(self, rgb, filled):
        for i, f in enumerate(filled):
            if f and i not in self.tier_cache:
                tier = self.reader.read(rgb, i)
                if tier is not None:      # 淡入帧读不出就下帧重试
                    self.tier_cache[i] = tier
                    # 档位与道具徽记在同一帧落账：卡片消失后无法再补读
                    self.tier_item[i] = self.items.read(rgb, i)
            # 买不起时卡面变暗，道具徽记可能读不到；False 不缓存，下帧重看
            if f and not self.item_cache.get(i):
                self.item_cache[i] = self.items.read(rgb, i)

    def _reset(self):
        self._clear_shop()
        self.sig_mute_until = 0.0
