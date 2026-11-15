r"""语音播放：后台线程 + winsound。

调度按「优先级 + 存活时间」而不是纯 FIFO：
  P0 结算/战败/首领击破/濒死 —— 可以打断低优先级正在播的语音
  P1 战斗流程（进入战斗、作战结果、休整开始、首领血量/倒计时）
  P2 战斗总结/流程页/升级
  P3 商店操作（购买/售出/刷新/冻结/三连）
  P4 闲置与同名提醒（排队超过 TTL 就丢，避免旧提醒迟到）
- 队列最多 3 条；高优先级入队时挤掉更低优先级的待播项。
- 每项都有 TTL：排队太久（前面的语音太长）就放弃，而不是十几秒后才响。
- 同一分类正在播/已排队时，新的同类触发丢弃；P0 可以抢占低优先级。
- 每个 key 可放多条同名变体（sell.wav / sell_2.wav / sell_3.wav），每次随机选一条。
"""

from __future__ import annotations

import os
import random
import shutil
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import winsound

import voice_pack

DEFAULT_COOLDOWN = 1.2
DEFAULT_VOLUME = 0.5
VOLUME_STEP = 0.05     # 缩放缓存按 5% 粒度落盘，避免滑杆拖动生成大量文件
FAST_KEYS = {"buy_low", "buy_high", "sell", "freeze", "unfreeze"}
FAST_COOLDOWN = 0.6
# 同步 PlaySound 在部分声卡/会话上直接失败，异步 + WAV 时长等待更稳
SND_ASYNC = 0x0001

QUEUE_MAX = 3
PREEMPT_LOWER = 2      # P0 可以打断 P2 及以下正在播的语音

# 同分类共用一次播放窗口；不同分类才排队。
CATEGORY = {
    "buy_low": "recruit", "buy_high": "recruit",
    "buy_item": "item",
    "sell": "sell",
    "level_up": "upgrade", "can_upgrade": "upgrade",
    "refresh": "refresh",
    "idle": "idle", "idle_lobby": "lobby_idle",
    "battle_enter": "battle",
    "battle_perfect": "battle_result", "battle_small": "battle_result",
    "battle_heavy": "battle_result",
    "near_death": "critical", "battle_lethal": "critical",
    "perfect_settlement": "settle", "all_fail_encourage": "settle",
    "rest_normal": "rest", "rest_boss": "rest",
    "mode_enter": "flow", "match_found": "flow", "strategy_select": "flow",
    "settle_return_home": "flow", "quick_leave": "flow",
    "final_start": "boss", "boss_victory": "boss_kill",
    "run_victory": "result", "run_defeat": "result",
    "freeze": "freeze", "unfreeze": "unfreeze",
    "same_operator": "same_op", "triple_reward": "triple",
}

P0 = {"run_victory", "run_defeat", "battle_lethal", "boss_victory", "near_death"}
P1 = {"battle_enter", "battle_perfect", "battle_small", "battle_heavy",
      "rest_normal", "rest_boss", "final_start", "mode_enter", "match_found"}
P2 = {"perfect_settlement", "all_fail_encourage", "strategy_select",
      "settle_return_home", "decision_select", "secret_core",
      "level_up", "can_upgrade"}
P3 = {"buy_low", "buy_high", "buy_item", "sell", "refresh", "freeze",
      "unfreeze", "triple_reward"}
# 其余（idle / idle_lobby / same_operator / quick_leave）都是 P4

TTL = {0: 10.0, 1: 5.0, 2: 3.0, 3: 2.5, 4: 1.5}
# 连续操作类语音再加一层按键冷却（秒）：比默认 1.2s 长，防止一串操作连播
KEY_COOLDOWN = {
    "buy_item": 6.0, "sell": 6.0, "refresh": 6.0,
    "same_operator": 30.0, "can_upgrade": 10.0, "triple_reward": 10.0,
    "freeze": 1.0, "unfreeze": 1.0,
}

# 非主动操作触发的语音分组（GUI 可关）；未列出的 key 永远可用
GROUP = {}
for _k in ("idle_lobby",):
    GROUP[_k] = "lobby_idle"
for _k in ("idle", "rest_normal", "rest_boss", "perfect_settlement",
           "all_fail_encourage", "can_upgrade", "same_operator",
           "triple_reward", "decision_select", "secret_core"):
    GROUP[_k] = "auto"
for _k in ("battle_enter", "final_start", "near_death", "battle_lethal",
           "battle_perfect", "battle_small", "battle_heavy", "boss_victory"):
    GROUP[_k] = "battle"
for _k in ("mode_enter", "match_found", "strategy_select",
           "settle_return_home", "quick_leave", "run_victory", "run_defeat"):
    GROUP[_k] = "flow"
# 首领血量 / 倒计时归入战斗组
GROUP.update({k: "battle" for k in
              (f"boss_hp_{n}" for n in (75, 50, 25, 10))})
GROUP.update({k: "battle" for k in ("time_60", "time_30", "time_10")})
# 防守结果也都是自动播报
# （defense_* 在 group_of 里按前缀归组）

def category_of(key: str) -> str:
    if key.startswith("defense_"):
        return "defense"
    if key.startswith("boss_hp_") or key.startswith("time_"):
        return "boss"
    if key.startswith("level_read_"):
        return "level"
    return CATEGORY.get(key, key)


def priority_of(key: str) -> int:
    if key in P0:
        return 0
    if key in P1 or key.startswith("boss_hp_") or key.startswith("time_"):
        return 1
    if key in P2 or key.startswith("defense_"):
        return 2
    if key in P3:
        return 3
    return 4


def group_of(key: str) -> str | None:
    if key.startswith("defense_"):
        return "battle"
    return GROUP.get(key)


def known_keys() -> set[str]:
    """程序会触发的全部 key；打包工具用它提示拼错的文件名。"""
    keys = set(CATEGORY) | P0 | P1 | P2 | P3 | set(KEY_COOLDOWN)
    keys.add("defense_enter")
    keys.update(f"boss_hp_{n}" for n in (75, 50, 25, 10))
    keys.update(f"time_{n}" for n in (60, 30, 10))
    keys.update(f"level_read_{n}" for n in range(1, 11))
    keys.update(f"defense_{role}_{result}"
                for role in ("help", "helped", "peer")
                for result in ("full", "basic", "fail"))
    return keys


@dataclass
class _Item:
    key: str
    cat: str
    path: Path
    epoch: int
    prio: int
    expires: float
    seq: int


class Player:
    def __init__(self, voice_dir: Path, cooldown: float = DEFAULT_COOLDOWN,
                 queue_max: int = QUEUE_MAX, log=print, volume: float = DEFAULT_VOLUME,
                 pack: Path | None = None):
        self.voice_dir = Path(voice_dir)
        self.cache_dir = self.voice_dir.parent / "_volume_cache"
        self.pack_path: Path | None = None
        self.pack_name = ""
        self.cooldown = cooldown
        self.log = log
        self.queue_max = max(1, int(queue_max))
        self._pending: list[_Item] = []
        self._cv = threading.Condition()
        self._seq = 0
        self._last: dict[str, float] = {}
        self._last_path: dict[str, Path] = {}
        self._missing: set[str] = set()
        self._index: dict[str, list[Path]] = {}
        self._paused = False
        self._muted = False
        self._stop = False
        self._current = ""
        self._current_cat = ""
        self._current_prio = 99
        self._stop_evt = threading.Event()
        self._dur: dict[Path, float] = {}
        self._lock = threading.Lock()
        self._epoch: dict[str, int] = {}
        self._off_groups: set[str] = set()
        self._volume = min(1.0, max(0.0, float(volume)))
        if pack is not None:
            self.set_pack(pack)
        else:
            self._scan()
        self._thread = threading.Thread(target=self._run, name="voice-player", daemon=True)
        self._thread.start()

    @property
    def current(self) -> str:
        with self._lock:
            return self._current

    @property
    def playing_category(self) -> str:
        with self._lock:
            return self._current_cat

    @property
    def pending(self) -> list[str]:
        with self._cv:
            return [it.key for it in self._pending]

    def path(self, key: str) -> Path:
        with self._lock:
            paths = self._index.get(key)
        return paths[0] if paths else self.voice_dir / (key + ".wav")

    def set_group(self, name: str, enabled: bool) -> None:
        with self._cv:
            if enabled:
                self._off_groups.discard(name)
            else:
                self._off_groups.add(name)
                self._pending = [it for it in self._pending
                                 if group_of(it.key) != name]

    def group_enabled(self, name: str) -> bool:
        with self._cv:
            return name not in self._off_groups

    def submit(self, key: str, t: float | None = None) -> bool:
        if self._paused or self._muted:
            return False
        if group_of(key) in self._off_groups:
            return False
        now = time.monotonic() if t is None else t
        last = self._last.get(key)
        cooldown = KEY_COOLDOWN.get(
            key, FAST_COOLDOWN if key in FAST_KEYS else self.cooldown)
        if last is not None and now - last < cooldown:
            return False
        p = self._pick(key)
        if p is None:
            if key not in self._missing:
                self._missing.add(key)
                where = self.pack_name or str(self.voice_dir)
                self.log(f"[player] {where} 里没有 {key}.wav，已跳过")
            return False
        cat = category_of(key)
        prio = priority_of(key)
        with self._cv:
            wall = time.monotonic()
            self._purge_locked(wall)
            same = [it for it in self._pending if it.cat == cat]
            if same:
                if min(it.prio for it in same) <= prio:
                    return False
                self._pending = [it for it in self._pending if it.cat != cat]
            # 高优先级入队：挤掉更低优先级的待播项
            if any(it.prio > prio for it in self._pending):
                self._pending = [it for it in self._pending if it.prio <= prio]
            if self._current_cat == cat:
                if self._current_prio <= prio:
                    return False
                if prio == 0 and self._current_prio >= PREEMPT_LOWER:
                    self._interrupt_locked()
                else:
                    return False
            elif (prio == 0 and self._current_cat
                  and self._current_prio >= PREEMPT_LOWER):
                self._interrupt_locked()
            while len(self._pending) >= self.queue_max:
                worst = max(self._pending, key=lambda it: (it.prio, -it.seq))
                if worst.prio <= prio:
                    self.log(f"[player] 队列已满，丢弃 {key}")
                    return False
                self._pending.remove(worst)
                self.log(f"[player] 队列已满，丢弃低优先级 {worst.key}")
            self._seq += 1
            self._pending.append(_Item(
                key=key, cat=cat, path=p, epoch=self._epoch.get(cat, 0),
                prio=prio, expires=wall + TTL.get(prio, 2.0), seq=self._seq))
            self._last[key] = now
            self._cv.notify()
        return True

    def _purge_locked(self, wall: float) -> None:
        if self._paused or self._muted or not self._pending:
            return
        keep = [it for it in self._pending if it.expires >= wall]
        for it in self._pending:
            if it.expires < wall:
                self.log(f"[player] 排队超时丢弃 {it.key}")
        self._pending = keep

    def _interrupt_locked(self) -> None:
        cat = self._current_cat
        self._epoch[cat] = self._epoch.get(cat, 0) + 1
        self._stop_evt.set()
        try:
            winsound.PlaySound(None, 0)
        except RuntimeError:
            pass

    def drop_category(self, category: str) -> None:
        """丢弃该分类里还没开始播放的语音（已在播的不管）。"""
        with self._cv:
            self._epoch[category] = self._epoch.get(category, 0) + 1
            self._pending = [it for it in self._pending if it.cat != category]

    def cut(self, category: str) -> None:
        """丢弃该分类排队的语音；正在播的也立刻停掉（结算返回要马上接主界面）。"""
        with self._cv:
            self._epoch[category] = self._epoch.get(category, 0) + 1
            self._pending = [it for it in self._pending if it.cat != category]
            if self._current_cat != category:
                return
            self._stop_evt.set()
            try:
                winsound.PlaySound(None, 0)
            except RuntimeError:
                pass

    @property
    def volume(self) -> float:
        return self._volume

    def set_volume(self, volume: float) -> None:
        self._volume = min(1.0, max(0.0, float(volume)))

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if muted:
            with self._cv:
                self._pending.clear()
            self._stop_evt.set()
            try:
                winsound.PlaySound(None, 0)
            except RuntimeError:
                pass

    def stop(self, wait: float = 3.0) -> None:
        self._stop = True
        self._stop_evt.set()
        try:
            winsound.PlaySound(None, 0)
        except RuntimeError:
            pass
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=wait)
        with self._cv:
            self._pending.clear()

    def _scan(self) -> None:
        self.pack_path = None
        self.pack_name = ""
        if not self.voice_dir.is_dir():
            return
        names = [p.name for p in self.voice_dir.glob("*.wav")]
        index = {key: [self.voice_dir / m for m in members]
                 for key, members in voice_pack.index_names(names).items()}
        with self._lock:
            self._index = index

    def set_pack(self, pack: Path | None) -> None:
        """切到某个语音包；None 表示回到 voices 目录。切换时停掉正在播的语音。"""
        if pack is None:
            with self._lock:
                self._missing.clear()
                self._last_path.clear()
                self._dur.clear()
            self._scan()
            self._drop_all()
            return
        pack = Path(pack)
        tag = voice_pack.pack_tag(pack)
        root = self.cache_dir / "packs"
        dest = root / tag
        index = voice_pack.extract(pack, dest)
        self._prune_cache(root, dest)
        info = voice_pack.read_manifest(pack)
        with self._lock:
            self._index = index
            self._missing.clear()
            self._last_path.clear()
            self._dur.clear()
            self.pack_path = pack
            self.pack_name = info["name"]
        self._drop_all()

    def _drop_all(self) -> None:
        with self._cv:
            for cat in self._epoch:
                self._epoch[cat] += 1
            self._pending.clear()
            self._stop_evt.set()
            try:
                winsound.PlaySound(None, 0)
            except RuntimeError:
                pass
            self._cv.notify_all()

    @staticmethod
    def _prune_cache(root: Path, keep: Path) -> None:
        try:
            for d in root.iterdir():
                if d.is_dir() and d != keep:
                    shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

    def _pick(self, key: str) -> Path | None:
        with self._lock:
            paths = self._index.get(key)
        if not paths and self.pack_path is None:
            self._scan()
            paths = self._index.get(key)
        if not paths:
            return None
        if len(paths) == 1:
            p = paths[0]
        else:
            last = self._last_path.get(key)
            choices = [x for x in paths if x != last] or paths
            p = random.choice(choices)
        self._last_path[key] = p
        return p

    def _scaled(self, p: Path) -> Path:
        """按当前音量生成缩放后的 WAV 缓存；winsound 只能靠文件改音量。"""
        vol = round(self._volume / VOLUME_STEP) * VOLUME_STEP
        if vol >= 0.999:
            return p
        # 包里同名文件可能属于不同语音包，按所在目录分开放，避免互相复用
        out = self.cache_dir / f"v{int(round(vol * 100)):03d}" / p.parent.name / p.name
        try:
            if out.exists() and out.stat().st_mtime >= p.stat().st_mtime:
                return out
        except OSError:
            return p
        try:
            with wave.open(str(p)) as w:
                params = w.getparams()
                raw = w.readframes(w.getnframes())
            if params.sampwidth == 2 and params.comptype == "NONE":
                import numpy as np
                arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) * vol
                raw = np.clip(arr, -32768, 32767).astype("<i2").tobytes()
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            with wave.open(str(tmp), "wb") as w:
                w.setnchannels(params.nchannels)
                w.setsampwidth(params.sampwidth)
                w.setframerate(params.framerate)
                w.writeframes(raw)
            os.replace(tmp, out)
        except (OSError, wave.Error, ValueError):
            return p
        return out

    def _duration(self, p: Path) -> float:
        with self._lock:
            d = self._dur.get(p)
        if d is None:
            try:
                with wave.open(str(p)) as w:
                    d = w.getnframes() / float(w.getframerate())
            except (OSError, wave.Error, ZeroDivisionError):
                d = 1.0
            with self._lock:
                self._dur[p] = d
        return d + 0.05

    def _take(self) -> _Item | None:
        with self._cv:
            while not self._stop:
                wall = time.monotonic()
                self._purge_locked(wall)
                if self._paused or self._muted or not self._pending:
                    self._cv.wait(0.2)
                    continue
                item = min(self._pending, key=lambda it: (it.prio, it.seq))
                self._pending.remove(item)
                if item.epoch != self._epoch.get(item.cat, 0):
                    continue
                self._current = item.key
                self._current_cat = item.cat
                self._current_prio = item.prio
                self._stop_evt.clear()
                return item
        return None

    def _run(self) -> None:
        while not self._stop:
            item = self._take()
            if item is None:
                break
            try:
                winsound.PlaySound(
                    str(self._scaled(item.path)),
                    winsound.SND_FILENAME | winsound.SND_NODEFAULT | SND_ASYNC)
                self._stop_evt.wait(self._duration(item.path))
            except RuntimeError as exc:
                self.log(f"[player] 播放失败 {item.key}: {exc}")
            finally:
                with self._lock:
                    self._current = ""
                    self._current_cat = ""
                    self._current_prio = 99
                with self._cv:
                    self._cv.notify_all()
