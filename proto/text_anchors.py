r"""文字锚点：给主界面加 OCR 背书。

标记模板在淡入/缩放帧上分数会掉；这里用固定 ROI 的 rec-only OCR 直接读字。
ROI 均为 1920 规范坐标，实测（2026-11-14 实况帧）：
  主界面   搜寻队友 / 独立模拟 0.94–1.00 / S.W.E.E.P.报告 0.99–1.00 / 资调配处 1.00
  负样本   规程页「开始搜寻」0.95、商店「收起/准备就绪」
"""

from __future__ import annotations

from ocr import read_text

LOBBY_SEARCH_BOX = (1652, 956, 1858, 1048)
LOBBY_SWEEP_BOX = (1690, 110, 1902, 158)
LOBBY_MAT_BOX = (1730, 230, 1912, 282)

# 组队模式是「搜寻队友」，单人模式同一个框读「独立模拟」
LOBBY_PRIMARY = ("搜寻队友", "独立模拟")

# 休整胶囊「休息一下」：1920 规范坐标，只框文字不框胶囊边。
# 实测 all1 172 个休整帧：读对 150、读空 20、读错 2（都是淡入/淡出帧的糊字）。
REST_PILL_BOX = (880, 22, 1122, 70)
REST_WORDS = ("休息", "息一")   # 720p 偶尔掉首字，读成「息一下」


def rest_pill_text(rgb) -> bool | None:
    """休整胶囊绑定文字：读得出就只认「休息一下」，读不出返回 None 交给 HUD 兜底。"""
    txt = read_text(rgb, REST_PILL_BOX)
    if txt is None:
        return None
    return any(k in txt for k in REST_WORDS)


def lobby_text(rgb) -> bool:
    """主界面：入口按钮文字为必要条件，再要一条副标题背书。

    规程页的「开始搜寻」和商店的「收起/准备就绪」都不含这些词，
    所以顺序短路：第一条不对就不再读另外两框。
    """
    search = read_text(rgb, LOBBY_SEARCH_BOX)
    if search is None or not any(k in search for k in LOBBY_PRIMARY):
        return False
    sweep = read_text(rgb, LOBBY_SWEEP_BOX) or ""
    if "SWEEP" in sweep.upper().replace(".", "").replace(" ", ""):
        return True
    return "配" in (read_text(rgb, LOBBY_MAT_BOX) or "")
