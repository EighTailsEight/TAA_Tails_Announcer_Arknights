r"""把逐帧检测结果接到语音状态机。

Bridge.update(img, t) → [Voice, ...]。只读画面，不点击、不注入。
时间语义：
  - rest 需连续 1.0s 才确认（消除购买界面/加载转轮的 teal 闪帧），退出用 3s 迟滞
  - 战斗进入只认金色/橙色波次胶囊（1.0s 内见过休整则延后），菜单/载入/结果页不会误报
  - 作战结束先保持 1.2s 并累计最大扣血：横幅首帧没有红字，立即判会误成完美
  - 联防结果看四人血量：行 1 亮点+teal 底栏=协防队友、+“自己”=队友协防、无亮点=队友互协防；
    血量在整波结算时才落定，等 2s 稳定再播
  - 最终攻势只看红胶囊（红占比 + 红像素平均绿色分量），持续解锁首领血量/剩余时间播报
  - 结果页（橙/红标题 NCC）latch，直到再次看到模式 logo
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from digits import LifeTracker, read_damage, read_life
from screen_detector import MARK_MIN_SCORE, DefenseTracker, ScreenDetector
from shop import ShopTracker
from state_machine import Reactor, Voice
import avatar_hud

REST_RECENT = 1.0      # 判定「刚离开休整」的窗口（越短，进战斗语音越早）
REST_CONFIRM = 1.0     # 休整需连续出现这么久才确认
REST_EXIT_HOLD = 3.0   # 休整掉检容忍：单帧缺检不重播休整语音
BATTLE_CONFIRM = 0.4
FINAL_CONFIRM = 0.4    # 红胶囊连成一段才认最终攻势（过场/淡出帧会闪一帧）
END_HOLD = 1.2         # 作战结束横幅保持时长（等红字扣血出现）
END_MIN = 0.35         # 横幅提前消失时的最短保持
END_DEDUP = 4.0        # 同一场作战结束的去重
END_RETRY = 2.0        # 横幅还在、红字却读不出：继续重试到这么久
END_COUNTER_WAIT = 3.0 # 横幅消失后用计数器差（战前−现在）反推扣血的最长等待
SEG_DEDUP = 4.0        # 联防内子战结束去重
BOSS_WINDOW = 120.0    # 最终攻势结束后多久内的休整仍算 boss 前准备
MODE_COOLDOWN = 8.0
STRATEGY_MIN = 0.55    # 策略页标题标记；编队页 0.555，2/3 页 ≥0.97（旧 0.75 实战漏报）
LOBBY_IDLE = 30.0      # 卫戍协议主界面停留多久算闲置（彩蛋）
HOME_COOLDOWN = 180.0  # 回到主界面语音冷却（用户口径：3 分钟）
# 休整期开头可能压着盟约/策略选择页：进入休整语音等选择页消失后再播，
# 2.5s 覆盖选择页的渲染延迟；之后如果选择页又出现，由 _in_rest 心跳保持会话。
REST_SELECT_GRACE = 2.5
SEARCH_CONFIRM = 0.5   # 「搜寻中」持续多久才算发起过搜索
SEARCH_LATCH = 300.0   # latch 过期，防止隔很久回主页仍报「快速离开」
PAGE_EXIT_HOLD = 2.0   # 事件页消失这么久才重新武装（页面切换有闪帧）
# 联防结果按四人血量结算，但血量在整波子战结束（作战结束横幅）时才统一扣除，
# 可能比 teal 底栏消失晚 2–15s（实测 a_0298、d_0191），所以要等血量稳定后再播。
DEF_SETTLE_HOLD = 2.0
DEF_SETTLE_MAX = 25.0

# 首领血条：胶囊内红条长度（满 1.0，超时约 0.23）。旧版按 n/15 波次计数，
# 波次冲得比血条快得多（434s 就报 75%，实际血条 445s 才到 75%），故改为血条。
BOSS_HP_MARKS = [(0.75, "boss_hp_75"), (0.50, "boss_hp_50"),
                 (0.25, "boss_hp_25"), (0.10, "boss_hp_10")]
BOSS_DEAD_HP = 0.025
BOSS_DEAD_NEED = 2      # 贴底读数要连续两帧：单帧 VFX/遮挡会把 10% 读成空
BOSS_ARM_HP = 0.06      # 血条低于此值才有资格按「消失」判击杀；超时结束约 22%
BOSS_ARM_NEED = 2       # 同样要连续两帧，单帧 0.05 然后回升不算
BOSS_KILL_WINDOW = 3.0
BOSS_GONE_HOLD = 0.8    # 血条/胶囊消失要持续这么久才算击破，单帧闪断不算
BOSS_END_MUTE = 20.0    # 击破后这段时间内的通用作战结束语音静音
TIME_MARKS = (60, 30, 10)

# 休整期闲置门控：整帧（去掉顶部 HUD，计时数字每秒都在动）降采样差分。
# 布阵/拖动/购买/刷新/冻结都会让画面变化；静止挂机时商店背景动画远低于门限。
ACT_LUMA = 12
# 0.45 只抓「卡行整排换掉/干员落场/面板切换」这类大变化；实测静止看商店时
# 悬停闪光/待机动画最高 0.37，再低会让闲置永远不触发。
ACT_FRAC = 0.45
ACT_SIZE = (96, 54)


def resolve_life_after(counter: int | None, life_before: int | None,
                       damage: int | None) -> int | None:
    """扣后生命 = 扣前 − 红字 N，兼容计数器提前刷新。

    扣血横幅期间计数器多数时候还是扣前值（f_286 期间 24，289s 休整才变 23），
    但也有已经变成扣后值的情况：第 13 回合扣 7 剩 6，计数器读到 6，旧算法
    6−7=−1 误判成战败。所以以战斗入场快照 life_before 为准算一次；计数器
    若等于扣前或扣后都采信快照结果，只有快照算出来是负数（快照过期/误读）
    才退回计数器。没有快照时，计数器减出负数说明它已是扣后值，直接用计数器。
    """
    if life_before is not None:
        if damage is None:
            return counter if counter is not None else life_before
        after = life_before - damage
        if counter is None or counter == after:
            return after
        if counter == life_before:
            return after
        # 快照过期（录屏剪辑/中途进入）：计数器减红字为负说明它已是扣后值，
        # 直接采信；否则按扣前值算，避免把「扣后 4 血」误当成剩余 4 血。
        sub = counter - damage
        return counter if sub < 0 else sub
    if counter is None:
        return None
    if damage is None:
        return counter
    after = counter - damage
    return counter if after < 0 and counter > 0 else after


class Bridge:
    def __init__(self, visual_activity: bool = True):
        self.visual_activity = bool(visual_activity)
        self.det = ScreenDetector()
        self.tracker = DefenseTracker()
        self.reactor = Reactor()
        self.life = LifeTracker()
        self.shop = ShopTracker()

        self._rest_t = None
        self._rest_run_start = None
        self._in_rest = False
        self._rest_voiced = False
        self._end = None
        self._last_end_t = None
        self._last_seg_t = None
        self._battle_since = None
        self._battle_fired = False
        self._def_pending = None
        self._game_active = False
        self._result_seen = False
        self._strategy_seen = False
        self._match_t = None
        self._mode_t = None
        self._lobby_since = None
        self._lobby_last_t = None
        self._lobby_idle_fired = False
        self._lobby_enter_fired = False
        self._lobby_matched = False
        self._last_home_t = None
        self._settle_return = False
        self._search_since = None
        self._search_t = None
        self._search_seen = False
        self._ready_seen = False
        self._ready_absent_since = None
        self._final_active = False
        self._final_last_t = None
        self._final_since = None
        self._ever_final = False
        self._boss_dead = False
        self._boss_dead_t = None
        self._boss_last_hp = None
        self._boss_low_t = None
        self._boss_low_n = 0
        self._boss_dead_n = 0
        self._bar_gone_since = None
        self._life_pre = None
        self._life_at_battle = None
        self._prev_small = None
        self._last_visual_t = -1e9
        self._hp_fired = set()
        self._time_fired = set()
        self._pages = {}
        self.last_read = None
        self.last_life_debug = None
        self.lobby_idle = LOBBY_IDLE
        self.home_cooldown = HOME_COOLDOWN
        # 语音队列控制挂点（GUI 里接 Player.drop_category / Player.cut），测试可为 None
        self.on_drop_voice = None
        self.on_cut_voice = None
        self.on_note = None

    def _note(self, msg: str) -> None:
        if self.on_note is not None:
            self.on_note(msg)

    def set_timings(self, *, rest_idle: float | None = None,
                    lobby_idle: float | None = None,
                    home_cooldown: float | None = None,
                    refresh_egg: int | None = None) -> None:
        if rest_idle is not None:
            self.reactor.config.idle_seconds = float(rest_idle)
        if lobby_idle is not None:
            self.lobby_idle = float(lobby_idle)
        if home_cooldown is not None:
            self.home_cooldown = float(home_cooldown)
        if refresh_egg is not None:
            self.shop.refresh_egg = max(1, int(refresh_egg))

    @property
    def rest_now(self) -> bool:
        return self._rest_t is not None and self._now - self._rest_t <= REST_EXIT_HOLD

    def _reset_run_flags(self):
        self._rest_t = None
        self._rest_run_start = None
        self._in_rest = False
        self._rest_voiced = False
        self._end = None
        self._last_end_t = None
        self._last_seg_t = None
        self._def_pending = None
        self._game_active = False
        self._result_seen = False
        self._strategy_seen = False
        self._search_since = None
        self._search_seen = False
        self._ready_seen = False
        self._ready_absent_since = None
        self._final_active = False
        self._final_last_t = None
        self._final_since = None
        self._ever_final = False
        self._boss_dead = False
        self._boss_dead_t = None
        self._boss_last_hp = None
        self._boss_low_t = None
        self._boss_low_n = 0
        self._boss_dead_n = 0
        self._bar_gone_since = None
        self._life_at_battle = None
        self.last_life_debug = None
        self._hp_fired = set()
        self._time_fired = set()
        self._pages.clear()
        self.shop.on_run_start()

    def _page_edge(self, t, on: bool, name: str) -> bool:
        """事件页上升沿：进入后报一次，页面离开 PAGE_EXIT_HOLD 后才重新武装。"""
        st = self._pages.setdefault(name, {"seen": False, "absent": None})
        if on:
            st["absent"] = None
            if not st["seen"]:
                st["seen"] = True
                return True
        elif st["seen"]:
            if st["absent"] is None:
                st["absent"] = t
            elif t - st["absent"] >= PAGE_EXIT_HOLD:
                st["seen"] = False
                st["absent"] = None
        return False

    def _fire_boss(self, t, out):
        if self._boss_dead:
            return
        self._boss_dead = True
        self._boss_dead_t = t
        out += self.reactor.handle({"type": "boss_victory", "t": t})
        self._note(f"[bridge] 首领击破 t={t:.2f} 最后血条={self._boss_last_hp}")

    def _visual_activity(self, rgb, t, rest: bool) -> None:
        """休整期画面差分：部署/拖动/买卡/刷新/冻结都会改变画面。"""
        if not self.visual_activity or not rest:
            self._prev_small = None
            return
        h = rgb.shape[0]
        y0 = int(round(120 * h / 1080))
        small = np.asarray(
            Image.fromarray(np.ascontiguousarray(rgb[y0:]))
            .resize(ACT_SIZE, Image.BILINEAR).convert("L"), dtype=np.int16)
        prev = self._prev_small
        self._prev_small = small
        if prev is None or prev.shape != small.shape:
            return
        frac = float((np.abs(small - prev) > ACT_LUMA).mean())
        if frac >= ACT_FRAC:
            self._last_visual_t = t

    def _defense_start(self, t, hp):
        self._def_pending = {
            "t0": t, "phase_end": None, "event": None,
            "start": None, "min": None, "last": None, "last_change": t,
            "dots": False, "teal": 0, "own": 0,
        }
        self._defense_hp(t, hp)

    def _defense_hp(self, t, hp):
        """记录四人血量：起始值 + 全程最小值（血量只降不升）+ 最后一次变化时刻。"""
        p = self._def_pending
        if p is None or any(v is None for v in hp):
            return
        if p["start"] is None:
            p["start"] = list(hp)
            p["min"] = list(hp)
            p["last"] = list(hp)
            return
        p["min"] = [min(a, b) for a, b in zip(p["min"], hp)]
        if list(hp) != p["last"]:
            p["last"] = list(hp)
            p["last_change"] = t

    def _defense_settle(self, t):
        p = self._def_pending
        self._def_pending = None
        if p is None or p["start"] is None or p["event"] is None:
            return []
        if p["dots"]:
            role = "help" if p["teal"] > p["own"] else "helped"
        else:
            role = "peer"
        loss = [s - m for s, m in zip(p["start"], p["min"])]
        return self.reactor.handle({
            "type": "defense_end", "t": t, "role": role,
            "loss": loss, "player_after": p["start"][0] - loss[0],
        })

    def update(self, img, t: float) -> list[Voice]:
        self._now = t
        rgb = np.asarray(img.convert("RGB"))
        read = self.det.analyze(rgb)
        self.last_read = read
        out: list[Voice] = []

        # ---- 文字页：结果 latch / 模式入口 / 匹配完成
        if read.result_page:
            if not self._result_seen:
                self._result_seen = True
                self._game_active = False
                self._end = None
                for ev in self.tracker.finish(t):
                    if self._def_pending is not None:
                        self._def_pending["phase_end"] = t
                        self._def_pending["event"] = ev
                out += self._defense_settle(t)
                result = ("victory" if read.mark("result_victory") >= read.mark("result_defeat")
                          else "defeat")
                # 兜底：本局进过最终攻势、结果页是胜利但实时击杀没认出来
                # （4fps 下血条可能从 20% 直接跳到结果页），补播一次击杀语音。
                # 只对胜利生效：超时结束时血条约 22%，结果是失败，不能报击破。
                if result == "victory" and self._ever_final and not self._boss_dead:
                    self._boss_dead = True
                    self._boss_dead_t = t
                    out += self.reactor.handle({"type": "boss_victory", "t": t})
                    self._note(f"[bridge] 结果页兜底首领击破 t={t:.2f}")
                out += self.reactor.handle({"type": "run_result", "t": t, "result": result})
                # 每局必有结算：下一帧主页若中间没有新的战斗识别，就是「结算回到主界面」
                self._settle_return = True
                # 结算后不允许再报上一局战斗（准备键 latch 必须清掉）
                self._ready_seen = False
                self._ready_absent_since = None
                # 一局结束即关 boss 窗口：否则下一局开头（录屏拼接处）的休整
                # 会落在上一局最终攻势的 120s 内，被误播成「最终攻势」
                self._final_last_t = None
                self._final_active = False
                self._final_since = None
            out += self.reactor.handle({"type": "tick", "t": t})
            return out

        # ---- 模式入口：卫戍协议标题画面，比主界面淡入早约 2s（e4_0017 @4s）
        if (read.mark("mode_logo") >= MARK_MIN_SCORE
                and (self._mode_t is None or t - self._mode_t > MODE_COOLDOWN)):
            self._mode_t = t
            self._lobby_enter_fired = True
            self._lobby_matched = False
            self._lobby_since = None
            self._last_home_t = t
            self._settle_return = False
            self._search_seen = False
            self._reset_run_flags()
            out += self.reactor.handle({"type": "mode_enter", "t": t})
            out += self.reactor.handle({"type": "tick", "t": t})
            return out

        # ---- 主界面：青色「搜寻队友」+ 两个灰框；标记淡入帧由 OCR 文字兜底
        lobby = read.lobby or read.lobby_text
        if not lobby and self._lobby_since is not None:
            # 不设掉检容忍：标记+文字已经双锚，掉一帧就按离开处理（用户口径）
            self._lobby_since = None
            self._lobby_last_t = None
            self._lobby_idle_fired = False
            self._lobby_enter_fired = False
            self._lobby_matched = False

        if lobby:
            if self._lobby_since is None:
                self._lobby_since = t
                self._lobby_idle_fired = False
            self._lobby_last_t = t
            # 只在主界面且没在匹配流程里才算闲置；每次停留只播一次
            if (not self._lobby_matched and not self._lobby_idle_fired
                    and t - self._lobby_since >= self.lobby_idle):
                self._lobby_idle_fired = True
                out += self.reactor.handle({"type": "lobby_idle", "t": t})
            if not self._lobby_enter_fired:
                self._lobby_enter_fired = True
                settle, leave = self._settle_return, self._search_seen
                self._result_seen = False
                self._reset_run_flags()
                self._settle_return = False
                self._search_seen = False
                if settle:
                    # 结算返回：上一局的「胜利/失败」还在播就把主界面播报压后了，
                    # 直接切掉结果语音，回主界面立刻报
                    if self.on_cut_voice is not None:
                        self.on_cut_voice("result")
                    out += self.reactor.handle({"type": "settle_return_home", "t": t})
                elif leave:
                    out += self.reactor.handle({"type": "quick_leave", "t": t})
                elif (self._last_home_t is None
                      or t - self._last_home_t > self.home_cooldown):
                    out += self.reactor.handle({"type": "mode_enter", "t": t})
            self._last_home_t = t
            out += self.reactor.handle({"type": "tick", "t": t})
            return out

        if read.mark("match_found") >= 0.55:
            self._lobby_matched = True
            self._search_seen = False
            self._search_since = None
            # 新一局已确定：上一次的「策略页已报」不能带过来（快速再匹配时
            # 主界面可能只扫到一帧或没扫到，_reset_run_flags 没机会跑）
            self._strategy_seen = False
            if self._match_t is None or t - self._match_t > MODE_COOLDOWN:
                self._match_t = t
                out += self.reactor.handle({"type": "match_found", "t": t})
            out += self.reactor.handle({"type": "tick", "t": t})
            return out

        # ---- 事件页锚点：决策三页每次进入报一次；隐秘核心是盖在战斗画面上的
        # 横幅，只报事件、不能提前 return
        decision_now = read.decision_page
        if self._page_edge(t, decision_now, "decision"):
            out += self.reactor.handle({"type": "decision_select", "t": t})
        if self._page_edge(t, read.secret_core, "secret"):
            out += self.reactor.handle({"type": "secret_core", "t": t})
        if decision_now:
            # 休整期里压着选择页：续上休整心跳，页面关掉后不再重播「进入休整」
            if self._in_rest:
                self._rest_t = t
            out += self.reactor.handle({"type": "tick", "t": t})
            return out

        # ---- 策略选择页：一局只报一次（三个页签标题标记都在同一位置）。
        # 「请选择策略」是旧 strategy_page 只到 0.50 时的补充锚点。必须排在
        # 「搜寻中」前面：搜寻 ROI 和策略页中部文字位置重叠，
        # 误判成 searching 会提前 return，策略语音就永远播不出来
        if (read.mark("strategy_page") >= STRATEGY_MIN
                or read.strategy_pick_page):
            if not self._strategy_seen:
                self._strategy_seen = True
                out += self.reactor.handle({"type": "strategy_select", "t": t})
            out += self.reactor.handle({"type": "tick", "t": t})
            return out

        # ---- 匹配中：搜索过且没匹配成功就回主页 = 快速离开
        if read.searching:
            if self._search_since is None:
                self._search_since = t
            if t - self._search_since >= SEARCH_CONFIRM:
                self._search_seen = True
                self._search_t = t
            out += self.reactor.handle({"type": "tick", "t": t})
            return out
        self._search_since = None
        if (self._search_seen and self._search_t is not None
                and t - self._search_t > SEARCH_LATCH):
            self._search_seen = False

        # ---- 休整 / 战斗计时
        watch_avatars = (read.defense_phase or self.tracker.in_defense
                         or self._def_pending is not None)
        av_hp = avatar_hud.read_hp(rgb) if watch_avatars else (None, None, None, None)
        av_dots = (avatar_hud.dots(rgb)
                   if (read.defense_phase or self.tracker.in_defense) else None)
        raw_life = read_life(rgb)
        life = self.life.update(raw_life)
        # 战斗外的生命值快照：战斗入场时冻结，扣血横幅里计数器可能提前刷新
        if read.pill not in ("combat", "final") and self._end is None and life is not None:
            self._life_pre = life
        if read.pill is not None:
            self._game_active = True
            # 结果页 latch 只靠 mode logo 清除会跨局失效（录屏拼接：上一局结果 →
            # 这一局游戏内画面 → 这一局结算），任何游戏内证据都说明已在新一局
            self._result_seen = False
            self._strategy_seen = False
        elif read.battle_end or read.defense_phase:
            self._result_seen = False
            self._strategy_seen = False
        # 休整绑定文字：胶囊上的「休息一下」是主锚点（all1 172 个休整帧读对 150），
        # 读到别的字（战斗横幅、退出后的游戏主界面）直接否决——这些页面顶部也泛青，
        # 单看颜色会 1s 后报「进入休整」、再过 10s 报「歇够了」。
        # 只有 OCR 读不出（None）时才退回局内 HUD 三选一：回合数/倒计时/准备键。
        rest_hud = read.rest and (
            read.rest_text is True
            or (read.rest_text is None
                and (read.round is not None or read.clock is not None
                     or read.ready_btn)))
        if rest_hud:
            self._rest_t = t
            if self._rest_run_start is None:
                self._rest_run_start = t
        else:
            self._rest_run_start = None
        rest_confirmed = (self._rest_run_start is not None
                          and t - self._rest_run_start >= REST_CONFIRM)

        # ---- 作战结束（先保持，累计最大扣血；先于联防结算，子战末尾的横幅要算进段）
        if read.battle_end:
            if self._end is None:
                self._end = {"t0": t, "damaged": False, "damage": 0}
            if read.end_damaged:
                self._end["damaged"] = True
                dmg = read_damage(rgb)
                if dmg is not None:
                    self._end["damage"] = max(self._end["damage"], dmg)
        if self._end is not None:
            e = self._end
            t0 = e["t0"]
            damaged = e["damaged"]
            damage = e["damage"] if damaged else 0
            if damaged and damage <= 0 and self._life_at_battle is not None and life is not None:
                # 计数器刷新得比红字早（也可能红字被遮挡）：战前−现在就是实际扣血
                delta = self._life_at_battle - life
                if delta > 0:
                    damage = delta
            waiting = False
            if damaged and damage <= 0:
                if read.battle_end:
                    waiting = t - t0 < END_RETRY
                else:
                    waiting = (self._life_at_battle is not None
                               and t - t0 < END_RETRY + END_COUNTER_WAIT)
            ready = (t - t0 >= END_HOLD
                     or (not read.battle_end and t - t0 >= END_MIN))
            if ready and not waiting:
                self._end = None
                unknown = damaged and damage <= 0
                if self.tracker.in_defense:
                    if self._last_seg_t is None or t - self._last_seg_t > SEG_DEDUP:
                        self._last_seg_t = t
                        self.tracker.add_segment(t, damaged)
                elif self._last_end_t is None or t - self._last_end_t > END_DEDUP:
                    self._last_end_t = t
                    # 计数器走跟踪值（连续 2 帧一致才采纳）：单帧 OCR 抖动会让
                    # 「21→19」按 2−2 算成 0 并误报游戏结束
                    counter = life if life is not None else raw_life
                    life_after = None if unknown else resolve_life_after(
                        counter, self._life_at_battle, damage)
                    self.last_life_debug = {
                        "t": t, "counter": raw_life, "tracked": life,
                        "before": self._life_at_battle, "damage": damage,
                        "after": life_after, "unknown": unknown,
                    }
                    self._note(
                        f"[bridge] 作战结束 扣{damage} 血 计数器{raw_life} "
                        f"跟踪{life} 战前{self._life_at_battle} → 剩余{life_after}"
                        + ("（读数缺失，跳过伤势判断）" if unknown else ""))
                    if (self._boss_dead and self._boss_dead_t is not None
                            and t - self._boss_dead_t <= BOSS_END_MUTE):
                        # 首领击破后的通用作战结束语音静音；结算语音不受影响
                        self._note(f"[bridge] 首领已击破，跳过通用作战结束语音 t={t:.2f}")
                    else:
                        ev = {
                            "type": "battle_end", "t": t,
                            "perfect": not damaged,
                            "damage": damage,
                            "life_after": life_after,
                        }
                        if unknown:
                            ev["unknown"] = True
                        out += self.reactor.handle(ev)

        # ---- 联防事件：入场/结束只记状态，结果等四人血量结算后再播
        for ev in self.tracker.update(read, t):
            if ev["kind"] == "defense_enter":
                out += self._defense_settle(t)
                self._defense_start(t, av_hp)
                out += self.reactor.handle({"type": "defense_enter", "t": t})
            elif ev["kind"] == "phase_end":
                if self._def_pending is not None:
                    self._def_pending["phase_end"] = t
                    self._def_pending["event"] = ev
        if self.tracker.in_defense:
            p = self._def_pending
            if p is not None:
                if av_dots is not None and av_dots[0]:
                    p["dots"] = True
                if read.watching_teammate:
                    p["teal"] += 1
                elif read.defending_self:
                    p["own"] += 1
        if self._def_pending is not None:
            self._defense_hp(t, av_hp)
            p = self._def_pending
            if p["phase_end"] is not None:
                settled = max(p["phase_end"], p["last_change"])
                if t - settled >= DEF_SETTLE_HOLD or t - p["phase_end"] >= DEF_SETTLE_MAX:
                    out += self._defense_settle(t)

        # ---- 进入战斗：右上「准备就绪/取消准备」消失 + 顶部休息胶囊消失。
        # 0/5 波次时胶囊识别为 None，旧版只看 combat 胶囊会漏报/迟报。
        if read.ready_btn:
            self._ready_seen = True
            self._ready_absent_since = None
        else:
            if self._ready_seen and self._ready_absent_since is None:
                self._ready_absent_since = t
            if (self._ready_seen and self._ready_absent_since is not None
                    and t - self._ready_absent_since >= BATTLE_CONFIRM
                    and read.pill != "rest" and not read.final_assault
                    # 最终攻势是本局最后一战，之后的过场帧（无胶囊/无准备键）
                    # 不是新战斗；结果页或回主页时才清空
                    and self._final_last_t is None
                    # 作战只能从休整（商店）出发：结算/结果页回主界面的淡入帧
                    # 即使误匹配到准备键/胶囊，也不能报「进入战斗」
                    and self.rest_now
                    # 放大镜打开的「盟约查看」页会盖住准备键/商店，整页标题在场
                    # 就不是进入战斗（休整掉检容忍期内打开查看页不会切走战斗）
                    and not read.core_covenant_page
                    and self._game_active and not self._result_seen
                    and not self.tracker.in_defense
                    and (self._last_end_t is None or t - self._last_end_t > END_DEDUP)):
                self._ready_seen = False
                self._ready_absent_since = None
                self._settle_return = False
                self._search_seen = False
                # 战斗已经打响：休整会话到此为止，下一轮休整重新计时/播报
                # （保留 _rest_t：商店 update 还要靠 3s 掉检容忍维持战斗入场静默）
                self._rest_run_start = None
                self._in_rest = False
                self._rest_voiced = False
                self.shop.on_battle_enter(t)
                self._life_at_battle = self._life_pre
                out += self.reactor.handle({"type": "battle_enter", "t": t})

        # ---- 休整进入（确认后先开休整会话；语音等选择页消失 + 宽限后才播，一轮只播一次）
        if rest_confirmed and not self._in_rest:
            self._in_rest = True
            self._rest_voiced = False
            self._final_active = False
            self.shop.on_rest_enter()
        if (rest_confirmed and self._in_rest and not self._rest_voiced
                and t - self._rest_run_start >= REST_SELECT_GRACE):
            self._rest_voiced = True
            boss = (read.round == 14
                    or (self._final_last_t is not None
                        and t - self._final_last_t <= BOSS_WINDOW))
            out += self.reactor.handle({"type": "rest_enter", "t": t, "boss": boss})
        if not self.rest_now:
            self._in_rest = False
            self._rest_voiced = False
        self._visual_activity(rgb, t, self.rest_now or rest_confirmed or self._in_rest)

        # ---- 最终攻势：红胶囊，或开场「击败敌方首领!」横幅（33/33 金胶囊下也认，
        # 横幅持续多帧）。红胶囊在过场/淡出帧里会只闪一帧，必须连成一段才认，
        # 否则「最终攻势」提前几秒播出，还会把下一局的休整算进 boss 窗口。
        # 两者都解锁首领血量/剩余时间播报；里程碑只在红胶囊下读数，
        # 横幅阶段 clock 还是波次计时，不能拿来做 60/30/10 判断。
        if read.final_assault and self._final_since is None:
            self._final_since = t
        elif not read.final_assault and not read.final_overlay:
            self._final_since = None
        if read.final_overlay and self._final_since is None:
            self._final_since = t
        final_now = read.final_overlay or (
            self._final_since is not None and t - self._final_since >= FINAL_CONFIRM)
        if final_now:
            self._final_last_t = t
            if not self._final_active:
                self._final_active = True
                self._ever_final = True
                self._hp_fired = set()
                self._time_fired = set()
                self._boss_dead = False
                self._boss_dead_t = None
                self._boss_last_hp = None
                self._boss_low_t = None
                self._boss_low_n = 0
                self._boss_dead_n = 0
                self._bar_gone_since = None
                # 最终攻势是独立战斗，生命快照不能沿用几轮前的普通作战
                self._life_at_battle = self._life_pre
                # 最终攻势之后的过场帧没有准备键/胶囊：旧 latch 不能再触发进战斗
                self._ready_seen = False
                self._ready_absent_since = None
                out += self.reactor.handle({"type": "final_start", "t": self._final_since})
            if read.final_assault:
                if read.boss_hp is not None:
                    if not self._boss_dead:
                        crossed = [(thr, key) for thr, key in BOSS_HP_MARKS
                                   if read.boss_hp <= thr and key not in self._hp_fired]
                        if crossed:
                            _, key = crossed[-1]
                            for _, key0 in crossed:
                                self._hp_fired.add(key0)
                            out += self.reactor.handle({"type": "boss_hp", "t": t, "key": key})
                if read.clock is not None:
                    crossed_t = [m for m in TIME_MARKS
                                 if read.clock <= m and m not in self._time_fired]
                    if crossed_t:
                        mark = min(crossed_t)
                        for m in crossed_t:
                            self._time_fired.add(m)
                        out += self.reactor.handle(
                            {"type": "time_warn", "t": t, "key": f"time_{mark}"})

        # ---- 首领击破：血条贴底且连续两帧，或贴底后血条/胶囊消失持续 0.8s。
        # 单帧 ≤15% 就武装会被 10% 的读数/VFX 骗走（实测还有 10% 血就报击破）；
        # 超时结束时血条约 22%，不会进入 6% 武器区。结果页胜利另有兜底。
        if self._final_active and not self._boss_dead:
            if read.final_assault and read.boss_hp is not None:
                hp = read.boss_hp
                self._boss_last_hp = hp
                self._bar_gone_since = None
                if hp <= BOSS_DEAD_HP:
                    self._boss_dead_n += 1
                    if self._boss_dead_n >= BOSS_DEAD_NEED:
                        self._fire_boss(t, out)
                else:
                    self._boss_dead_n = 0
                if hp <= BOSS_ARM_HP:
                    self._boss_low_n += 1
                    if self._boss_low_n >= BOSS_ARM_NEED:
                        self._boss_low_t = t
                else:
                    self._boss_low_n = 0
            elif (self._boss_low_t is not None
                    and t - self._boss_low_t <= BOSS_KILL_WINDOW):
                # 血条读空或红胶囊消失：必须从第一帧消失起持续 BOSS_GONE_HOLD
                self._boss_dead_n = 0
                self._boss_low_n = 0
                if self._bar_gone_since is None:
                    self._bar_gone_since = t
                elif t - self._bar_gone_since >= BOSS_GONE_HOLD:
                    self._fire_boss(t, out)

        for ev in self.shop.update(rgb, t, rest=self.rest_now):
            if ev["type"] == "voice_drop":
                if self.on_drop_voice is not None:
                    self.on_drop_voice(ev["category"])
                continue
            out += self.reactor.handle(ev)

        out += self.reactor.handle({"type": "tick", "t": t, "shop": self.shop.visible,
                                    "active": self._last_visual_t})
        return out
