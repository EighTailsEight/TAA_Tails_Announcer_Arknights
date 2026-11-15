r"""Tails Announcer 发布版：卫戍协议屏幕语音播报。

打开游戏 → 选窗口 → 截图测试 → 「开始监控」。程序每秒截取约 4 帧画面，
识别界面状态并播放对应语音；只截屏，不点击、不注入、不读游戏内存。

截图测试会真实抓一帧并跑完整检测：抓帧返回正常（不是黑帧/超时）才允许
开始监控，避免「监控灯亮着但一帧都读不到」的假运行。

start.bat 会请求管理员权限（与 OBS 采集窗口同权限）。
"""

from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

APP_DIR = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):          # PyInstaller one-dir：数据在 _internal，产物在 exe 旁
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = APP_DIR.parent
    sys.path.insert(0, str(BASE / "proto"))
sys.path.insert(0, str(APP_DIR))

import avatar_hud  # noqa: E402
import capture  # noqa: E402
from bridge import Bridge  # noqa: E402
from digits import read_life  # noqa: E402
from player import Player  # noqa: E402
from screen_detector import ScreenDetector  # noqa: E402
import voice_pack  # noqa: E402

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")

try:                                   # 150% 缩放的机器上让 Tk 用物理像素，界面不糊
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL
user32.GetParent.argtypes = (wintypes.HWND,)
user32.GetParent.restype = wintypes.HWND
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
kernel32.QueryFullProcessImageNameW.argtypes = (
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

LIVE_FPS = 4.0
LOG_MAX = 400
FONT = "Microsoft YaHei UI"
MONO = "Consolas"

BG = "#0b0e14"
CARD = "#141924"
CARD_EDGE = "#212a3a"
INPUT = "#0f1420"
LOG_BG = "#0f131b"
TEXT = "#e7ecf5"
MUTED = "#8b96ab"
DIM = "#3b4559"
ACCENT = "#4c9aff"
ACCENT_HOVER = "#6cb0ff"
DANGER = "#e2574c"
DANGER_HOVER = "#ef6c60"
GHOST = "#1d2534"
GHOST_HOVER = "#283246"
OK = "#33d6a6"
WARN = "#ffb454"
ERR = "#ff6b6b"

# 截图测试的「返回时间正常」：150% DPI 下 PrintWindow 通常 10–60ms，
# 超过 500ms 说明 DWM 在等帧，监控会掉到 4fps 以下。
CAP_OK_MS = 150.0
CAP_WARN_MS = 500.0

PACKS_DIR = BASE / "packs"

GAME_HINTS = ("明日方舟", "arknights")
NOISE_TITLES = {"Program Manager", "Windows 输入体验", "Windows Input Experience",
                "MSCTFIME UI", "Default IME"}
NOISE_CLASSES = {"IME", "MSCTFIME UI"}


def window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def exe_name(hwnd: int) -> str:
    pid = window_pid(hwnd)
    if not pid:
        return ""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        n = wintypes.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return ""
        return Path(buf.value).stem
    finally:
        kernel32.CloseHandle(h)


def is_game_window(win: dict) -> bool:
    hay = f'{win.get("title", "")} {win.get("exe", "")}'.casefold()
    return any(h.casefold() in hay for h in GAME_HINTS)


class Card(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=CARD, bd=0,
                         highlightbackground=CARD_EDGE, highlightcolor=CARD_EDGE,
                         highlightthickness=1)


_BUTTON_COLORS = {
    "primary": (ACCENT, "#ffffff", ACCENT_HOVER),
    "danger": (DANGER, "#ffffff", DANGER_HOVER),
    "ghost": (GHOST, TEXT, GHOST_HOVER),
}


def button(parent, text, command, kind="ghost", small=False):
    bg, fg, hover = _BUTTON_COLORS[kind]
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                  activebackground=hover, activeforeground=fg, bd=0,
                  relief="flat", highlightthickness=0, cursor="hand2",
                  font=(FONT, 9 if small else 10, "bold"),
                  padx=10 if small else 16, pady=3 if small else 7)

    def apply(kind_name):
        bgb, fgb, hoverb = _BUTTON_COLORS[kind_name]
        b._bg, b._fg, b._hover = bgb, fgb, hoverb
        b.configure(bg=bgb, fg=fgb, activebackground=hoverb)

    def on_enter(_):
        if str(b["state"]) != "disabled":
            b.configure(bg=b._hover)

    def on_leave(_):
        if str(b["state"]) != "disabled":
            b.configure(bg=b._bg)

    b.set_kind = apply
    apply(kind)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)
    return b


class Switch(tk.Canvas):
    """自绘开关：整条点击切换。"""

    def __init__(self, parent, value: bool, command):
        super().__init__(parent, width=44, height=24, bg=CARD,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.value = bool(value)
        self.command = command
        self.bind("<Button-1>", self._toggle)
        self._draw()

    def _toggle(self, _event) -> None:
        self.value = not self.value
        self._draw()
        self.command(self.value)

    def _draw(self) -> None:
        self.delete("all")
        track = ACCENT if self.value else "#2a3345"
        self.create_oval(2, 4, 18, 20, fill=track, outline="")
        self.create_oval(26, 4, 42, 20, fill=track, outline="")
        self.create_rectangle(10, 4, 34, 20, fill=track, outline="")
        kx = 34 if self.value else 10
        self.create_oval(kx - 7, 5, kx + 7, 19, fill="#ffffff", outline="")


class MiniSlider(tk.Canvas):
    """细轨道 + 圆钮的音量滑杆；value 0..1。"""

    def __init__(self, parent, value: float, command):
        super().__init__(parent, width=140, height=22, bg=CARD,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.value = max(0.0, min(1.0, float(value)))
        self.command = command
        self.bind("<Button-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Configure>", lambda _e: self._draw())
        self._draw()

    def _on_drag(self, event) -> None:
        pad = 9
        span = max(1, self.winfo_width() - pad * 2)
        self.value = max(0.0, min(1.0, (event.x - pad) / span))
        self._draw()
        self.command(self.value)

    def _draw(self) -> None:
        self.delete("all")
        w = self.winfo_width() or 140
        cy = (self.winfo_height() or 22) // 2
        pad = 9
        x = pad + (w - pad * 2) * self.value
        self.create_line(pad, cy, w - pad, cy, fill="#2a3345",
                         width=4, capstyle="round")
        self.create_line(pad, cy, x, cy, fill=ACCENT, width=4, capstyle="round")
        self.create_oval(x - 6, cy - 6, x + 6, cy + 6, fill=TEXT, outline="")


def spin(parent, var, lo, hi, inc):
    return tk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=inc,
                      width=5, bg=INPUT, fg=TEXT, buttonbackground=GHOST,
                      insertbackground=TEXT, relief="flat", bd=0,
                      highlightthickness=1, highlightbackground=CARD_EDGE,
                      highlightcolor=ACCENT, justify="center",
                      font=(MONO, 9))


class LiveMonitor:
    """后台线程：窗口抓帧 → Bridge → Player；跨线程只走回调。"""

    def __init__(self, hwnd: int, title: str, settings: dict, on_event,
                 on_stat, on_stop, log, gen: int = 0):
        self.hwnd = hwnd
        self.title = title
        self.gen = gen
        self.interval = 1.0 / LIVE_FPS
        self.on_event = on_event
        self.on_stat = on_stat
        self.on_stop = on_stop
        self.log = log
        pack = settings.get("pack") or ""
        try:
            self.player = Player(BASE / "voices", log=log,
                                 pack=Path(pack) if pack else None)
        except Exception as exc:
            log(f"[系统] 语音包加载失败，本次监控没有语音：{exc!r}")
            self.player = Player(BASE / "voices", log=log)
        self.bridge = Bridge()
        self.bridge.on_drop_voice = self.player.drop_category
        self.bridge.on_cut_voice = self.player.cut
        self.bridge.set_timings(**settings["timings"])
        for name, enabled in settings["groups"].items():
            self.player.set_group(name, enabled)
        self.paused = False
        self.frames = 0
        self.voices = 0
        self.fps_now = 0.0
        self.cap_ms = 0.0
        self.det_ms = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            self.player.pause()
        else:
            self.player.resume()

    def set_muted(self, muted: bool) -> None:
        self.player.set_muted(muted)

    def set_volume(self, volume: float) -> None:
        self.player.set_volume(volume)

    def set_group(self, name: str, enabled: bool) -> None:
        self.player.set_group(name, enabled)

    def set_pack(self, path: Path | None) -> None:
        self.player.set_pack(path)

    def stop(self, wait: float = 4.0) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=wait)
        self.player.stop()

    def _warm_ocr(self) -> None:
        try:
            import ocr
            if not ocr.warmup():
                self.log("[系统] 文字识别不可用，部分文字锚点降级")
        except Exception as exc:
            self.log(f"[系统] 文字识别预热失败：{exc!r}")

    def _stat(self) -> dict:
        return {"fps": self.fps_now, "cap": self.cap_ms,
                "det": self.det_ms, "frames": self.frames,
                "voice": self.player.current}

    def _run(self) -> None:
        cap = None
        t0 = time.perf_counter()
        paused_total = 0.0
        pause_t0 = None
        stat_t = time.perf_counter()
        frame0 = 0
        stat_emit = 0.0
        threading.Thread(target=self._warm_ocr, name="ocr-warm",
                         daemon=True).start()
        try:
            while not self._stop.is_set():
                if self.paused:
                    if pause_t0 is None:
                        pause_t0 = time.perf_counter()
                    self._stop.wait(0.05)
                    continue
                if pause_t0 is not None:
                    paused_total += time.perf_counter() - pause_t0
                    pause_t0 = None
                if cap is None:
                    if not user32.IsWindow(self.hwnd):
                        self.log("[系统] 游戏窗口已关闭")
                        return
                    try:
                        cap = capture.WindowCapture(self.hwnd)
                    except OSError as exc:
                        self.log(f"[系统] 连接窗口失败：{exc!r}")
                        return
                    self.log(f"[系统] 已连接窗口「{self.title}」")
                loop0 = time.perf_counter()
                img = cap.grab()
                cap_ms = (time.perf_counter() - loop0) * 1000
                if img is None:
                    if not user32.IsWindow(self.hwnd):
                        self.log("[系统] 游戏窗口已关闭")
                        return
                    if cap.fail_streak > 30:
                        self.log("[系统] 连续取不到画面，正在重连窗口")
                        cap.close()
                        cap = None
                    self._stop.wait(0.5)
                    continue
                t = time.perf_counter() - t0 - paused_total
                det0 = time.perf_counter()
                try:
                    out = self.bridge.update(img, t)
                except Exception as exc:
                    self.log(f"[系统] 画面处理失败：{exc!r}")
                    out = []
                det_ms = (time.perf_counter() - det0) * 1000
                self.frames += 1
                self.cap_ms = cap_ms if not self.cap_ms else 0.7 * self.cap_ms + 0.3 * cap_ms
                self.det_ms = det_ms if not self.det_ms else 0.7 * self.det_ms + 0.3 * det_ms
                for v in out:
                    self.voices += 1
                    self.on_event(t, v.key)
                    self.player.submit(v.key, t)
                now = time.perf_counter()
                if now - stat_t >= 1.0:
                    self.fps_now = (self.frames - frame0) / (now - stat_t)
                    frame0 = self.frames
                    stat_t = now
                if now - stat_emit >= 0.5:
                    stat_emit = now
                    try:
                        self.on_stat(self._stat())
                    except Exception:
                        pass
                wait = self.interval - (time.perf_counter() - loop0)
                if wait > 0:
                    self._stop.wait(wait)
        finally:
            if cap is not None:
                cap.close()
            self.log(f"[系统] 监控结束：{self.frames} 帧 / {self.voices} 条语音")
            self.on_stop(self.gen)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.live: LiveMonitor | None = None
        self.windows: list[dict] = []
        self.events: queue.Queue = queue.Queue()
        self.detector: ScreenDetector | None = None
        self.tested_hwnd: int | None = None
        self.voices = 0
        self.live_stat: dict = {}
        self._closing = False
        self._live_gen = 0
        self._fps_mark = time.perf_counter()
        self._logf = None

        self._open_log()
        self._style()
        self._build()
        root.bind("<space>", self._on_space)
        root.report_callback_exception = self._on_tk_error
        root.after(60, self.refresh_windows)
        root.after(30, self.pump)

    # ---------------- 外观
    def _style(self) -> None:
        root = self.root
        root.title("Tails Announcer · 卫戍协议")
        root.configure(bg=BG)
        root.geometry("1000x1060")
        root.minsize(860, 720)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.update_idletasks()
        self._dark_title_bar()

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=INPUT, background=INPUT,
                        foreground=TEXT, arrowcolor=MUTED, bordercolor=CARD_EDGE,
                        lightcolor=INPUT, darkcolor=INPUT, insertcolor=TEXT,
                        padding=6, font=(FONT, 9))
        style.map("TCombobox",
                  fieldbackground=[("readonly", INPUT), ("disabled", INPUT)],
                  foreground=[("readonly", TEXT), ("disabled", MUTED)],
                  selectbackground=[("readonly", INPUT)],
                  selectforeground=[("readonly", TEXT)],
                  bordercolor=[("disabled", CARD_EDGE)])
        style.configure("Vertical.TScrollbar", background="#232a38",
                        troughcolor=BG, bordercolor=BG, arrowcolor=MUTED,
                        lightcolor="#232a38", darkcolor="#232a38",
                        gripcount=0, arrowsize=12)
        style.map("Vertical.TScrollbar",
                  background=[("active", "#2d3750"), ("disabled", "#232a38")])
        root.option_add("*TCombobox*Listbox.background", INPUT)
        root.option_add("*TCombobox*Listbox.foreground", TEXT)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", "#0b0e14")
        root.option_add("*TCombobox*Listbox.font", (FONT, 9))

    def _dark_title_bar(self) -> None:
        try:
            hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            value = ctypes.c_int(1)
            for attr in (20, 19):          # Win11 / Win10 的深色标题栏
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        wintypes.HWND(hwnd), attr, ctypes.byref(value),
                        ctypes.sizeof(value)) == 0:
                    break
        except Exception:
            pass

    def _build(self) -> None:
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=20, pady=(16, 12))
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(4, weight=1)

        # ---- 标题栏
        head = tk.Frame(outer, bg=BG)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        head.grid_columnconfigure(1, weight=1)
        title = tk.Frame(head, bg=BG)
        title.grid(row=0, column=0, sticky="w")
        tk.Label(title, text="Tails Announcer", bg=BG, fg=TEXT,
                 font=(FONT, 17, "bold")).pack(anchor="w")
        tk.Label(title, text="明日方舟 · 卫戍协议 实时语音播报", bg=BG, fg=MUTED,
                 font=(FONT, 9)).pack(anchor="w", pady=(2, 0))

        chip = tk.Frame(head, bg=CARD, highlightbackground=CARD_EDGE,
                        highlightthickness=1)
        chip.grid(row=0, column=2, sticky="e")
        self.dot = tk.Canvas(chip, width=10, height=10, bg=CARD,
                             highlightthickness=0, bd=0)
        self.dot.pack(side="left", padx=(12, 7), pady=10)
        self.dot_id = self.dot.create_oval(1, 1, 9, 9, fill=MUTED, outline="")
        self.chip_text = tk.Label(chip, text="未连接", bg=CARD, fg=MUTED,
                                  font=(FONT, 9))
        self.chip_text.pack(side="left", padx=(0, 13), pady=8)

        # ---- 截图测试
        ts = Card(outer)
        ts.grid(row=1, column=0, sticky="ew")
        th = tk.Frame(ts, bg=CARD)
        th.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(th, text="截图测试", bg=CARD, fg=TEXT,
                 font=(FONT, 10, "bold")).pack(side="left")
        self.test_state = tk.Label(th, text="尚未测试", bg=CARD, fg=WARN,
                                   font=(FONT, 9))
        self.test_state.pack(side="left", padx=(10, 0))
        self.btn_test = button(th, "截图测试", self.screenshot_test, "ghost")
        self.btn_test.pack(side="right")

        tb = tk.Frame(ts, bg=CARD)
        tb.pack(fill="x", padx=16, pady=(0, 12))
        for col in range(4):
            tb.grid_columnconfigure(col, weight=1 if col % 2 else 0)
        self.tv_window = tk.StringVar(value="—")
        self.tv_size = tk.StringVar(value="—")
        self.tv_cap = tk.StringVar(value="—")
        self.tv_det = tk.StringVar(value="—")
        self.tv_mean = tk.StringVar(value="—")
        self.tv_read = tk.StringVar(value="—")
        self.tv_shot = tk.StringVar(value="—")
        self._kv(tb, 0, 0, "窗口", self.tv_window)
        self._kv(tb, 0, 2, "分辨率", self.tv_size)
        self.lb_cap = self._kv(tb, 1, 0, "抓帧返回", self.tv_cap)
        self._kv(tb, 1, 2, "识别耗时", self.tv_det)
        self._kv(tb, 2, 0, "画面均值", self.tv_mean)
        self._kv(tb, 2, 2, "识别结果", self.tv_read)
        self._kv(tb, 3, 0, "存图", self.tv_shot)
        tk.Label(tb, text="抓帧正常（非黑帧、返回时间稳定）后才能开始监控",
                 bg=CARD, fg=MUTED, font=(FONT, 9)).grid(
            row=3, column=2, columnspan=2, sticky="w", padx=(14, 0), pady=3)

        # ---- 控制区
        ctl = Card(outer)
        ctl.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        c1 = tk.Frame(ctl, bg=CARD)
        c1.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(c1, text="游戏窗口", bg=CARD, fg=MUTED,
                 font=(FONT, 9)).pack(side="left", padx=(0, 8))
        self.win_var = tk.StringVar()
        self.win_box = ttk.Combobox(c1, textvariable=self.win_var,
                                    state="readonly", font=(FONT, 9))
        self.win_box.pack(side="left", fill="x", expand=True)
        self.win_box.bind("<<ComboboxSelected>>", self._on_window_changed)
        self.btn_refresh = button(c1, "刷新", self.refresh_windows, "ghost")
        self.btn_refresh.pack(side="left", padx=(8, 0))

        c2 = tk.Frame(ctl, bg=CARD)
        c2.pack(fill="x", padx=16, pady=(0, 14))
        self.btn_live = button(c2, "开始监控", self.on_live_toggle, "primary")
        self.btn_live.pack(side="left")
        self.btn_pause = button(c2, "暂停", self.toggle_pause, "ghost")
        self.btn_pause.pack(side="left", padx=(8, 0))
        self.btn_pause.configure(state="disabled", bg=GHOST, fg="#5a6478")

        self.switch = Switch(c2, True, self.on_voice)
        self.switch.pack(side="right", padx=(14, 0))
        tk.Label(c2, text="语音播报", bg=CARD, fg=TEXT,
                 font=(FONT, 9)).pack(side="right", padx=(0, 6))
        tk.Label(c2, text="音量", bg=CARD, fg=MUTED,
                 font=(FONT, 9)).pack(side="right", padx=(10, 2))
        self.vol = MiniSlider(c2, 0.5, self.on_volume)
        self.vol.pack(side="right")

        # ---- 播报设置
        st = Card(outer)
        st.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        sh = tk.Frame(st, bg=CARD)
        sh.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(sh, text="播报设置", bg=CARD, fg=TEXT,
                 font=(FONT, 10, "bold")).pack(side="left")
        tk.Label(sh, text="数值在开始监控时生效 · 开关立即生效", bg=CARD, fg=MUTED,
                 font=(FONT, 9)).pack(side="right")

        sr = tk.Frame(st, bg=CARD)
        sr.pack(fill="x", padx=16, pady=(4, 8))
        self.v_rest_idle = tk.StringVar(value="15")
        self.v_lobby_idle = tk.StringVar(value="30")
        self.v_home = tk.StringVar(value="180")
        self.v_refresh = tk.StringVar(value="2")
        for var, lo, hi, inc, label, unit in (
                (self.v_rest_idle, 5, 120, 5, "休息闲置", "秒"),
                (self.v_lobby_idle, 10, 300, 10, "主页闲置", "秒"),
                (self.v_home, 30, 600, 30, "回主页冷却", "秒"),
                (self.v_refresh, 1, 20, 1, "连续刷新彩蛋", "次")):
            cell = tk.Frame(sr, bg=CARD)
            cell.pack(side="left", padx=(0, 22))
            tk.Label(cell, text=label, bg=CARD, fg=MUTED,
                     font=(FONT, 9)).pack(side="left", padx=(0, 6))
            spin(cell, var, lo, hi, inc).pack(side="left")
            tk.Label(cell, text=unit, bg=CARD, fg=DIM,
                     font=(FONT, 9)).pack(side="left", padx=(4, 0))

        sp = tk.Frame(st, bg=CARD)
        sp.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(sp, text="语音包", bg=CARD, fg=MUTED,
                 font=(FONT, 9)).pack(side="left", padx=(0, 6))
        self.pack_var = tk.StringVar()
        self.pack_box = ttk.Combobox(sp, textvariable=self.pack_var,
                                     state="readonly", font=(FONT, 9))
        self.pack_box.pack(side="left", fill="x", expand=True)
        self.pack_box.bind("<<ComboboxSelected>>", self._on_pack_changed)
        tk.Label(sp, text="切换立即生效 · 包里缺的语音静默跳过", bg=CARD, fg=DIM,
                 font=(FONT, 8)).pack(side="left", padx=(10, 0))
        self._load_packs()

        sg = tk.Frame(st, bg=CARD)
        sg.pack(fill="x", padx=16, pady=(0, 12))
        self.sw_lobby = Switch(sg, True, lambda on: self.on_group("lobby_idle", on))
        self.sw_auto = Switch(sg, True, lambda on: self.on_group("auto", on))
        self.sw_battle = Switch(sg, True, lambda on: self.on_group("battle", on))
        self.sw_flow = Switch(sg, True, lambda on: self.on_group("flow", on))
        for sw, label, hint in (
                (self.sw_lobby, "主页闲置", "主界面停留"),
                (self.sw_auto, "自动提醒", "闲置/升级/同名"),
                (self.sw_battle, "战斗播报", "血条/倒计时/防守"),
                (self.sw_flow, "流程播报", "进入/匹配/结算")):
            cell = tk.Frame(sg, bg=CARD)
            cell.pack(side="left", padx=(0, 20))
            sw.pack(side="left")
            box = tk.Frame(cell, bg=CARD)
            box.pack(side="left", padx=(7, 0))
            tk.Label(box, text=label, bg=CARD, fg=TEXT,
                     font=(FONT, 9)).pack(anchor="w")
            tk.Label(box, text=hint, bg=CARD, fg=DIM,
                     font=(FONT, 8)).pack(anchor="w")

        # ---- 日志
        lg = Card(outer)
        lg.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        lh = tk.Frame(lg, bg=CARD)
        lh.pack(fill="x", padx=16, pady=(10, 6))
        tk.Label(lh, text="运行日志", bg=CARD, fg=TEXT,
                 font=(FONT, 10, "bold")).pack(side="left")
        button(lh, "清空", self.clear_log, "ghost", small=True).pack(side="right")

        body = tk.Frame(lg, bg=LOG_BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.log = tk.Text(body, height=7, bg=LOG_BG, fg="#c8d2e4", bd=0,
                           font=(MONO, 9), highlightthickness=0, wrap="none",
                           state="disabled", padx=10, pady=8,
                           insertbackground=TEXT)
        sb = ttk.Scrollbar(body, orient="vertical", command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("event", foreground=ACCENT)
        self.log.tag_configure("info", foreground=MUTED)
        self.log.tag_configure("warn", foreground=WARN)

        # ---- 底部
        foot = tk.Frame(outer, bg=BG)
        foot.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        tk.Label(foot, text="空格 暂停 / 继续 · 只截屏识别，不操作游戏",
                 bg=BG, fg=MUTED, font=(FONT, 9)).pack(side="left")
        self.foot_right = tk.Label(foot, text="", bg=BG, fg=MUTED,
                                   font=(MONO, 9))
        self.foot_right.pack(side="right")

        self._set_start_enabled(False)
        self._log("[系统] 就绪：选择游戏窗口 → 点「截图测试」→ 开始监控", "info")

    def _kv(self, parent, row: int, col: int, label: str,
            var: tk.StringVar) -> tk.Label:
        tk.Label(parent, text=label, bg=CARD, fg=MUTED,
                 font=(FONT, 9)).grid(row=row, column=col, sticky="w", pady=3)
        value = tk.Label(parent, textvariable=var, bg=CARD, fg=TEXT,
                         font=(MONO, 9))
        value.grid(row=row, column=col + 1, sticky="w", padx=(10, 26), pady=3)
        return value

    # ---------------- 日志
    def _open_log(self) -> None:
        try:
            d = BASE / "_logs"
            d.mkdir(exist_ok=True)
            p = d / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
            self._logf = open(p, "w", encoding="utf-8")
        except OSError:
            self._logf = None

    def _log(self, msg: str, tag: str = "info") -> None:
        if self._logf is not None:
            try:
                self._logf.write(msg + "\n")
                self._logf.flush()
            except OSError:
                pass
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n", tag)
        last = int(self.log.index("end-1c").split(".")[0])
        if last > LOG_MAX:
            self.log.delete("1.0", f"{last - LOG_MAX + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def push_log(self, msg: str) -> None:
        self.events.put(("log", msg))

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ---------------- 窗口选择
    def _own_hwnds(self) -> set[int]:
        out = set()
        for get in (lambda: int(self.root.wm_frame(), 16),
                    lambda: int(self.root.winfo_id())):
            try:
                out.add(get())
            except Exception:
                pass
        return out

    def refresh_windows(self) -> None:
        if self.live is not None:
            return
        own = self._own_hwnds()
        me = os.getpid()
        items = []
        for w in capture.list_windows():
            x1, y1, x2, y2 = w["rect"]
            ww, hh = x2 - x1, y2 - y1
            if ww < 160 or hh < 120 or x1 <= -30000 or y1 <= -30000:
                continue
            if w["title"] in NOISE_TITLES or w["cls"] in NOISE_CLASSES:
                continue
            if w["hwnd"] in own or window_pid(w["hwnd"]) == me:
                continue
            exe = exe_name(w["hwnd"])
            items.append({**w, "exe": exe, "w": ww, "h": hh,
                          "label": f'{w["title"][:48]} · {ww}×{hh}'
                                   + (f" · {exe}" if exe else "")})
        items.sort(key=lambda v: (0 if is_game_window(v) else 1, -v["w"] * v["h"]))
        prev = self.current_window()
        self.windows = items
        self.win_box.configure(values=[v["label"] for v in items])
        idx = 0
        if prev is not None:
            for i, v in enumerate(items):
                if v["hwnd"] == prev["hwnd"]:
                    idx = i
                    break
        if items:
            self.win_box.current(idx)
            game = is_game_window(items[idx])
            self._log(f"[窗口] 发现 {len(items)} 个窗口，当前「{items[idx]['title'][:30]}」"
                      + ("（已识别为游戏）" if game else ""), "info")
        else:
            self.win_var.set("")
            self._log("[窗口] 没有可选窗口：请先打开游戏再点「刷新」", "warn")
        self._refresh_start_state()
        self._show_test_state()

    def current_window(self) -> dict | None:
        i = self.win_box.current()
        if 0 <= i < len(self.windows):
            return self.windows[i]
        return None

    def _on_window_changed(self, _event=None) -> None:
        self._refresh_start_state()
        self._show_test_state()

    def _show_test_state(self) -> None:
        win = self.current_window()
        if win is not None and win["hwnd"] == self.tested_hwnd:
            self.test_state.configure(text="测试通过", fg=OK)
        elif win is not None and self.tested_hwnd is not None:
            self.test_state.configure(text="窗口已切换，请重新测试", fg=WARN)
        else:
            self.test_state.configure(text="尚未测试", fg=WARN)

    # ---------------- 截图测试
    def _detector(self) -> ScreenDetector:
        if self.detector is None:
            self.detector = ScreenDetector()
        return self.detector

    def _test_fail(self, msg: str) -> None:
        self.tested_hwnd = None
        self.test_state.configure(text="测试失败", fg=ERR)
        self.tv_cap.set("—")
        self.push_log(f"[截图测试] {msg}")
        self._chip("测试失败", ERR)
        self._refresh_start_state()
        messagebox.showerror("截图测试", msg)

    def screenshot_test(self, win: dict | None = None) -> bool:
        win = win or self.current_window()
        if win is None:
            messagebox.showwarning("截图测试", "没有选中的窗口：先点「刷新」。")
            return False
        if self.live is not None:
            messagebox.showinfo("截图测试", "监控进行中：先停止监控再测试。")
            return False
        t0 = time.perf_counter()
        try:
            cap = capture.WindowCapture(win["hwnd"])
        except OSError as exc:
            self._test_fail(f"无法连接窗口：{exc!r}")
            return False
        try:
            img = cap.grab()
        finally:
            cap.close()
        cap_ms = (time.perf_counter() - t0) * 1000
        if img is None:
            self._test_fail("取不到画面（窗口最小化 / 被遮挡 / 黑帧）")
            return False
        rgb = np.asarray(img.convert("RGB"))
        mean = float(rgb.mean())
        if mean < 1.0:
            self._test_fail("整帧全黑：窗口最小化，或系统/反作弊保护不让截图")
            return False
        try:                                  # 留一张原图，现场标定用
            d = BASE / "_shots"
            d.mkdir(exist_ok=True)
            shot_path = d / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            img.save(shot_path)
        except OSError as exc:
            shot_path = None
            self.push_log(f"[截图测试] 存图失败: {exc!r}")
        t1 = time.perf_counter()
        read = self._detector().analyze(rgb)
        det_ms = (time.perf_counter() - t1) * 1000
        hp = avatar_hud.read_hp(rgb)
        life = read_life(rgb)

        self.tested_hwnd = win["hwnd"]
        self.tv_window.set(win["title"][:30])
        self.tv_size.set(f"{img.width}×{img.height}")
        self.tv_cap.set(f"{cap_ms:.0f} ms")
        self.lb_cap.configure(
            fg=OK if cap_ms <= CAP_OK_MS else WARN if cap_ms <= CAP_WARN_MS else ERR)
        self.tv_det.set(f"{det_ms:.0f} ms")
        self.tv_mean.set(f"{mean:.1f}")
        pill = read.pill or "—"
        life_text = life if life is not None else "—"
        self.tv_read.set(f"胶囊 {pill} · 生命 {life_text} · 单人 {hp}")
        self.tv_shot.set(shot_path.name if shot_path is not None else "未保存")
        self.test_state.configure(text="测试通过", fg=OK)
        self._chip("已就绪", OK)
        self._refresh_start_state()
        self._log(f"[截图测试] 「{win['title'][:26]}」 {img.width}×{img.height} "
                  f"抓帧 {cap_ms:.1f}ms 识别 {det_ms:.1f}ms 均值 {mean:.1f} "
                  f"胶囊 {pill} 生命 {life_text} 单人 {hp}"
                  + (f" 存图 {shot_path.name}" if shot_path is not None else ""))
        if cap_ms > CAP_WARN_MS:
            self._log(f"[截图测试] 抓帧 {cap_ms:.0f}ms 偏慢：可能被遮挡或 DWM 等帧，"
                      "监控帧率会下降", "warn")
        return True

    # ---------------- 播报设置
    def _read_number(self, var: tk.StringVar, default: float,
                     lo: float, hi: float) -> float:
        try:
            value = float(var.get())
        except (TypeError, ValueError):
            value = default
        return max(lo, min(hi, value))

    def _load_packs(self) -> None:
        # 本构建只带 packs/；voices 只在开发目录里存在
        self.pack_choices: list[tuple[str, str]] = []
        if (BASE / "voices").is_dir():
            self.pack_choices.append(("文件夹 voices", ""))
        try:
            for path in sorted(PACKS_DIR.glob("*.zip")):
                try:
                    info = voice_pack.summarize(path)
                except Exception as exc:
                    if hasattr(self, "log"):
                        self._log(f"[系统] 语音包读取失败 {path.name}：{exc!r}", "warn")
                    continue
                self.pack_choices.append(
                    (f"{info['name']}（{len(info['keys'])} 个 key）", str(path)))
        except OSError:
            pass
        if not self.pack_choices:
            self.pack_choices = [("（未找到语音包）", "")]
        self.pack_box.configure(values=[label for label, _ in self.pack_choices])
        index = next((i for i, (_, p) in enumerate(self.pack_choices) if p), 0)
        self.pack_box.current(index)
        self.pack_var.set(self.pack_choices[index][0])

    def _on_pack_changed(self, _event=None) -> None:
        i = self.pack_box.current()
        if i < 0:
            return
        label, path = self.pack_choices[i]
        self.pack_var.set(label)
        if self.live is None:
            if hasattr(self, "log"):
                self._log(f"[系统] 语音包：{label}（开始监控时生效）", "info")
            return
        try:
            self.live.set_pack(Path(path) if path else None)
            self._log(f"[系统] 已切换语音包：{label}", "info")
        except Exception as exc:
            self._log(f"[系统] 语音包切换失败：{exc!r}", "warn")

    def _settings(self) -> dict:
        return {
            "timings": {
                "rest_idle": self._read_number(self.v_rest_idle, 15, 5, 120),
                "lobby_idle": self._read_number(self.v_lobby_idle, 30, 10, 300),
                "home_cooldown": self._read_number(self.v_home, 180, 30, 600),
                "refresh_egg": int(self._read_number(self.v_refresh, 2, 1, 20)),
            },
            "groups": {
                "lobby_idle": self.sw_lobby.value,
                "auto": self.sw_auto.value,
                "battle": self.sw_battle.value,
                "flow": self.sw_flow.value,
            },
            "pack": self.pack_choices[self.pack_box.current()][1]
                    if self.pack_box.current() >= 0 else "",
        }

    def on_group(self, name: str, enabled: bool) -> None:
        if self.live is not None:
            self.live.set_group(name, enabled)

    # ---------------- 状态
    def _chip(self, text: str, color: str) -> None:
        self.dot.itemconfigure(self.dot_id, fill=color)
        self.chip_text.configure(text=text, fg=color)

    def _refresh_start_state(self) -> None:
        win = self.current_window()
        ok = (win is not None and win["hwnd"] == self.tested_hwnd)
        self._set_start_enabled(ok)

    def _set_start_enabled(self, enabled: bool) -> None:
        if self.live is not None:
            return
        if enabled:
            self.btn_live.set_kind("primary")
            self.btn_live.configure(state="normal", text="开始监控")
        else:
            self.btn_live.configure(state="disabled", bg=GHOST,
                                    fg="#5a6478", text="开始监控")

    def _set_pause(self, paused: bool) -> None:
        self.btn_pause.configure(text="继续" if paused else "暂停",
                                 state="normal" if self.live is not None else "disabled")
        if self.live is None:
            self.btn_pause.configure(bg=GHOST, fg="#5a6478")

    # ---------------- 监控
    def on_live_toggle(self) -> None:
        if self.live is not None:
            self.stop_live()
        else:
            self.start_live()

    def start_live(self) -> None:
        if self.live is not None:
            return
        win = self.current_window()
        if win is None:
            messagebox.showwarning("开始监控", "没有可用的窗口。\n请先打开游戏，再点「刷新」。")
            return
        if win["hwnd"] != self.tested_hwnd:
            messagebox.showwarning("开始监控", "请先对这个窗口做一次「截图测试」。")
            return
        self.voices = 0
        self.live_stat = {}
        self._live_gen += 1
        self.live = LiveMonitor(win["hwnd"], win["title"],
                                settings=self._settings(),
                                on_event=self.on_live_event,
                                on_stat=self.on_live_stat,
                                on_stop=self.on_live_stop,
                                log=self.push_log, gen=self._live_gen)
        self.live.set_muted(not self.switch.value)
        self.live.set_volume(self.vol.value)
        self.live.start()
        self.btn_live.set_kind("danger")
        self.btn_live.configure(text="停止监控", state="normal")
        self.btn_pause.set_kind("ghost")
        self._set_pause(False)
        self.win_box.configure(state="disabled")
        self.btn_refresh.configure(state="disabled", bg=GHOST, fg="#5a6478")
        self.btn_test.configure(state="disabled", bg=GHOST, fg="#5a6478")
        self._chip("监控中", OK)
        self.foot_right.configure(text="")
        self._log(f"[系统] 开始监控「{win['title']}」", "info")

    def stop_live(self, reason: str | None = None) -> None:
        m, self.live = self.live, None
        self._live_gen += 1
        if m is not None:
            m.stop()
        self.live_stat = {}
        self.btn_live.set_kind("primary")
        self.btn_pause.configure(state="disabled", bg=GHOST, fg="#5a6478",
                                 text="暂停")
        self.win_box.configure(state="readonly")
        self.btn_refresh.configure(state="normal", bg=GHOST, fg=TEXT)
        self.btn_test.set_kind("ghost")
        self.btn_test.configure(state="normal")
        self._refresh_start_state()
        if reason == "lost":
            self._chip("连接断开", ERR)
        elif reason == "error":
            self._chip("出错", ERR)
        else:
            self._chip("已停止" if m is not None else "未连接", MUTED)
        self.foot_right.configure(text="")

    def on_live_event(self, vt: float, key: str) -> None:
        self.events.put(("voice", (vt, key)))

    def on_live_stat(self, info: dict) -> None:
        self.events.put(("lstat", info))

    def on_live_stop(self, gen: int) -> None:
        self.events.put(("lstop", gen))

    def toggle_pause(self) -> None:
        if self.live is None:
            return
        paused = not self.live.paused
        self.live.set_paused(paused)
        self.btn_pause.configure(text="继续" if paused else "暂停")
        self.foot_right.configure(text="已暂停" if paused else "")
        self._chip("已暂停" if paused else "监控中", WARN if paused else OK)

    def on_voice(self, on: bool) -> None:
        if self.live is not None:
            self.live.set_muted(not on)

    def on_volume(self, value: float) -> None:
        if self.live is not None:
            self.live.set_volume(value)

    def _on_space(self, _evt) -> None:
        if isinstance(self.root.focus_get(), tk.Text):
            return
        self.toggle_pause()

    def _on_tk_error(self, _exc, val, _tb) -> None:
        self._log(f"[系统] 界面操作失败：{val!r}", "warn")

    # ---------------- 主线程轮询
    def pump(self) -> None:
        if self._closing:
            return
        try:
            for _ in range(200):
                kind, payload = self.events.get_nowait()
                if kind == "voice":
                    _vt, key = payload
                    self.voices += 1
                    self._log(f"{time.strftime('%H:%M:%S')}   {key}", "event")
                elif kind == "log":
                    self._log(payload, "info")
                elif kind == "lstat":
                    self.live_stat = payload
                elif kind == "lstop":
                    if self.live is not None and payload == self._live_gen:
                        self.stop_live("lost")
        except queue.Empty:
            pass

        now = time.perf_counter()
        if now - self._fps_mark >= 1.0:
            self._fps_mark = now
            if self.live is not None and not self.live.paused:
                st = self.live_stat
                self.foot_right.configure(
                    text=f"{st.get('fps', 0):4.1f} 帧/s · 抓帧 {st.get('cap', 0):.0f} ms"
                         f" · 检测 {st.get('det', 0):.0f} ms · 语音 {self.voices} 条")
        self.root.after(30, self.pump)

    def on_close(self) -> None:
        self._closing = True
        try:
            if self.live is not None:
                self.stop_live()
        finally:
            if self._logf is not None:
                try:
                    self._logf.close()
                except OSError:
                    pass
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
