r"""Win32 窗口截图（只读：PrintWindow 取像素，不注入输入）。

python capture.py --list                 列出可见窗口
python capture.py --grab out.png [--title 明日方舟]
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import numpy as np
from PIL import Image

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def enable_dpi_awareness() -> None:
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


def list_windows(substr: str | None = None) -> list[dict]:
    out: list[dict] = []
    CB = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if not n:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if substr and substr.casefold() not in title.casefold():
            return True
        rc = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rc))
        name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, name, 256)
        out.append({
            "hwnd": int(hwnd),
            "title": title,
            "cls": name.value,
            "rect": (rc.left, rc.top, rc.right, rc.bottom),
        })
        return True

    user32.EnumWindows(CB(cb), 0)
    return out


def find_window(substr: str) -> int | None:
    hits = list_windows(substr)
    if not hits:
        return None
    # 同标题多开时取窗口面积最大的那个
    best = max(hits, key=lambda w: (w["rect"][2] - w["rect"][0]) * (w["rect"][3] - w["rect"][1]))
    return best["hwnd"]


class WindowCapture:
    """按窗口抓帧；复用 DIB，尺寸变化时重建。"""

    def __init__(self, hwnd: int):
        enable_dpi_awareness()
        self.hwnd = hwnd
        self.wdc = user32.GetWindowDC(hwnd)
        self.mdc = gdi32.CreateCompatibleDC(self.wdc)
        self._bmp = None
        self._old = None
        self._bits = ctypes.c_void_p()
        self._size = (0, 0)
        self.fail_streak = 0

    def _release(self) -> None:
        if self._bmp:
            gdi32.SelectObject(self.mdc, self._old)
            gdi32.DeleteObject(self._bmp)
            self._bmp = None
            self._old = None
            self._bits = ctypes.c_void_p()

    def _ensure(self, w: int, h: int) -> None:
        if (w, h) == self._size and self._bmp:
            return
        self._release()
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # 顶朝下
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        self._bmp = gdi32.CreateDIBSection(
            self.mdc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(self._bits), None, 0)
        if not self._bmp:
            raise OSError("CreateDIBSection failed")
        self._old = gdi32.SelectObject(self.mdc, self._bmp)
        self._size = (w, h)

    def _size_now(self) -> tuple[int, int]:
        wr = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(wr))
        return wr.right - wr.left, wr.bottom - wr.top

    def _print(self, flags: int) -> bool:
        return bool(user32.PrintWindow(self.hwnd, self.mdc, flags))

    def grab(self) -> Image.Image | None:
        w, h = self._size_now()
        if w <= 0 or h <= 0:
            return None
        self._ensure(w, h)
        if not self._print(PW_RENDERFULLCONTENT):
            self._print(0)
        raw = ctypes.string_at(self._bits, w * h * 4)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        rgb = arr[:, :, [2, 1, 0]]  # BGRA → RGB
        if int(rgb[::8, ::8].mean()) < 1:
            # 窗口最小化或 DWM 不给帧：再试一次旧接口
            self._print(0)
            arr = np.frombuffer(ctypes.string_at(self._bits, w * h * 4), dtype=np.uint8).reshape(h, w, 4)
            rgb = arr[:, :, [2, 1, 0]]
            if int(rgb[::8, ::8].mean()) < 1:
                self.fail_streak += 1
                return None
        self.fail_streak = 0
        # 只取客户区（游戏全屏时两者一致）
        wr = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(wr))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        cr = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(cr))
        ox, oy = pt.x - wr.left, pt.y - wr.top
        cw, ch = cr.right - cr.left, cr.bottom - cr.top
        if (cw, ch) != (w, h) and cw > 0 and ch > 0:
            rgb = rgb[oy:oy + ch, ox:ox + cw]
        return Image.fromarray(np.ascontiguousarray(rgb))

    def close(self) -> None:
        self._release()
        gdi32.DeleteDC(self.mdc)
        user32.ReleaseDC(self.hwnd, self.wdc)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--list" in sys.argv or len(sys.argv) == 1:
        for w in list_windows():
            print(w["hwnd"], w["rect"], w["cls"], w["title"])
        return
    title = "明日方舟"
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]
    hwnd = find_window(title)
    if not hwnd:
        print("未找到窗口:", title)
        return
    cap = WindowCapture(hwnd)
    img = cap.grab()
    if img is None:
        print("抓帧失败（窗口最小化？）")
        return
    print("size", img.size, "mean", round(float(np.asarray(img).mean()), 1))
    if "--grab" in sys.argv:
        out = sys.argv[sys.argv.index("--grab") + 1]
        img.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()
