from dataclasses import dataclass, field


ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
SHOP_GAP = 1.2     # 商店不可见持续这么久才算真的收起了（选卡/面板闪帧不算）


@dataclass
class Voice:
    t: float
    key: str


@dataclass
class Config:
    idle_seconds: float = 15.0


@dataclass
class Reactor:
    config: Config = field(default_factory=Config)
    phase: str = "idle"
    own_perfect: bool = False
    own_damage: int = 0
    own_life_after: int = 0
    prepared: bool = False
    idle_fired: bool = False
    last_activity_t: float = 0.0
    defense_role: str = ""
    came_from_battle: bool = False
    defense_occurred: bool = False
    result_voiced: bool = False
    boss_down: bool = False
    buy_low_fired: bool = False
    buy_high_fired: bool = False
    _shop_gap_since: float | None = None

    def handle(self, event):
        kind = event["type"]
        t = float(event.get("t", 0.0))
        if kind == "mode_enter":
            self._reset_run()
            self.phase = "menu"
            return [Voice(t, "mode_enter")]
        if kind == "match_found":
            return [Voice(t, "match_found")]
        if kind == "strategy_select":
            return [Voice(t, "strategy_select")]
        if kind == "lobby_idle":
            return [Voice(t, "idle_lobby")]
        if kind == "settle_return_home":
            return [Voice(t, "settle_return_home")]
        if kind == "quick_leave":
            return [Voice(t, "quick_leave")]
        if kind in ("freeze", "unfreeze", "same_operator", "triple_reward",
                    "decision_select", "secret_core"):
            return self._activity(t, kind)
        if kind == "refresh":
            return self._activity(t, "refresh")
        if kind == "rest_enter":
            return self._rest_enter(t, bool(event.get("boss", False)))
        if kind == "purchase":
            tier = event.get("tier")
            level = ROMAN.get(tier, tier) if isinstance(tier, str) else tier
            if level is None:
                return self._activity(t, None)
            return self._purchase(t, "buy_low" if int(level) <= 3 else "buy_high")
        if kind == "buy_item":
            return self._activity(t, "buy_item")
        if kind == "level_up":
            return self._activity(t, "level_up")
        if kind == "can_upgrade":
            return self._activity(t, "can_upgrade")
        if kind == "sell":
            return self._activity(t, "sell")
        if kind == "ready":
            self.prepared = True
            self.idle_fired = True
            return []
        if kind == "battle_enter":
            self.phase = "battle"
            self.came_from_battle = False
            self.defense_occurred = False
            self.prepared = False
            self.result_voiced = False
            return [Voice(t, "battle_enter")]
        if kind == "battle_end":
            return self._battle_end(t, event)
        if kind == "defense_enter":
            return self._defense_enter(t, event)
        if kind == "defense_end":
            return self._defense_end(t, event)
        if kind == "final_start":
            self.phase = "final"
            return [Voice(t, "final_start")]
        if kind == "boss_victory":
            self.boss_down = True
            return [Voice(t, "boss_victory")]
        if kind in ("boss_hp", "time_warn"):
            return [Voice(t, event["key"])]
        if kind == "run_result":
            self.phase = "result"
            self.result_voiced = True
            self.came_from_battle = False
            self.defense_occurred = False
            return [Voice(t, "run_victory" if event["result"] == "victory" else "run_defeat")]
        if kind == "tick":
            return self._tick(t, bool(event.get("shop", True)), event.get("active"))
        raise ValueError(f"unknown event: {kind}")

    def _reset_run(self):
        self.phase = "idle"
        self.own_perfect = False
        self.own_damage = 0
        self.own_life_after = 0
        self.prepared = False
        self.idle_fired = False
        self.last_activity_t = 0.0
        self.defense_role = ""
        self.came_from_battle = False
        self.defense_occurred = False
        self.result_voiced = False
        self.boss_down = False
        self.buy_low_fired = False
        self.buy_high_fired = False
        self._shop_gap_since = None

    def _activity(self, t, key):
        self.last_activity_t = t
        # 不重置 idle_fired：闲置语音每轮休整最多一次（用户口径）
        return [] if key is None else [Voice(t, key)]

    def _purchase(self, t, key):
        """低/高阶干员购买每轮休整最多报一次：连续买人时语音会盖住操作。"""
        self.last_activity_t = t
        flag = "buy_low_fired" if key == "buy_low" else "buy_high_fired"
        if getattr(self, flag):
            return []
        setattr(self, flag, True)
        return [Voice(t, key)]

    def _battle_end(self, t, event):
        if self.boss_down:
            # 首领已击破：之后的通用作战结束/战斗总结都不再播，只留结算语音
            self.phase = "result"
            self.came_from_battle = True
            self.defense_occurred = False
            return []
        self.phase = "result"
        self.came_from_battle = True
        self.defense_occurred = False
        self.own_perfect = bool(event["perfect"])
        self.own_damage = int(event.get("damage", 0))
        self.own_life_after = event.get("life_after")
        if event.get("unknown"):
            # 扣血/剩余血量没读全：不判重创/丝血/战败，也不猜扣血量
            return []
        if not self.own_perfect and self.own_damage <= 0:
            self.own_damage = 1
        voices = []
        lethal = (self.own_damage > 0 and self.own_life_after is not None
                  and self.own_life_after <= 0)
        if self.own_perfect:
            voices.append(Voice(t, "battle_perfect"))
        elif lethal:
            voices.append(Voice(t, "battle_lethal"))
        elif self.own_damage >= 4:
            voices.append(Voice(t, "battle_heavy"))
        else:
            voices.append(Voice(t, "battle_small"))
        if (self.own_damage > 0 and self.own_life_after is not None
                and 1 <= self.own_life_after <= 3):
            voices.append(Voice(t, "near_death"))
        return voices

    def _defense_enter(self, t, event):
        self.phase = "defense"
        self.came_from_battle = True
        self.defense_occurred = True
        self.defense_role = event.get("role") or ""
        return [Voice(t, "defense_enter")]

    def _defense_end(self, t, event):
        role = event.get("role") or self.defense_role or "peer"
        loss = event.get("loss")
        self.phase = "post_defense"
        self.defense_role = role
        # 血量没读完（数字缺识别）就不猜结果，宁可少播一句
        if not isinstance(loss, (list, tuple)) or len(loss) != 4:
            return []
        if any(v is None for v in loss):
            return []
        loss = [int(v) for v in loss]
        player_after = event.get("player_after")
        if role == "helped":
            # 队友协防玩家：看玩家自己掉血，且不能进濒死（剩余 <=3）
            if loss[0] <= 0:
                outcome = "full"
            elif loss[0] <= 3 and player_after is not None and int(player_after) > 3:
                outcome = "basic"
            else:
                outcome = "fail"
        else:
            # 玩家协防队友 / 队友协防队友：看队友三人掉血总和
            teammate_loss = sum(loss[1:])
            if teammate_loss <= 0:
                outcome = "full"
            elif teammate_loss <= 3:
                outcome = "basic"
            else:
                outcome = "fail"
        return [Voice(t, f"defense_{role}_{outcome}")]

    def _rest_enter(self, t, boss):
        self.phase = "rest"
        self.prepared = False
        self.idle_fired = False
        self.last_activity_t = t
        self.buy_low_fired = False
        self.buy_high_fired = False
        voices = []
        if (self.came_from_battle and not self.defense_occurred
                and not self.result_voiced and not self.boss_down):
            if self.own_perfect:
                voices.append(Voice(t, "perfect_settlement"))
            else:
                voices.append(Voice(t, "all_fail_encourage"))
        voices.append(Voice(t, "rest_boss" if boss else "rest_normal"))
        self.came_from_battle = False
        self.defense_occurred = False
        self.defense_role = ""
        return voices

    def _tick(self, t, shop=True, active=None):
        if self.phase != "rest" or self.prepared or self.idle_fired:
            return []
        # 画面里真的在动（部署/拖动/买卡/刷新/冻结）就推迟闲置语音；
        # 只靠商店事件不够：布阵、调整站位、看面板都不会发出商店事件。
        if active is not None and float(active) > self.last_activity_t:
            self.last_activity_t = float(active)
        if not shop:
            # 商店不在场：短闪帧（选卡/面板）不算离开；真收起（布阵/战斗）
            # 就重新起算闲置计时，否则「歇够了」会在刚放干员那一刻播出来
            if self._shop_gap_since is None:
                self._shop_gap_since = t
            if t - self._shop_gap_since > SHOP_GAP:
                self.last_activity_t = t
            return []
        self._shop_gap_since = None
        if t - self.last_activity_t >= self.config.idle_seconds:
            self.idle_fired = True
            return [Voice(t, "idle")]
        return []

    def feed(self, events):
        out = []
        for event in events:
            out.extend(self.handle(event))
        return out
