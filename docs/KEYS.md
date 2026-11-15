# 语音 key 列表

语音文件名（不含 `.wav`）就是 key。同一 key 的随机变体用 `_2`、`_3` 后缀，例如 `sell.wav`、`sell_2.wav`。

- 包里缺少某个 key 时，对应状态静默跳过，不会报错
- 打包工具会提示 `known_keys()` 之外的文件名（通常是拼写错误）
- `level_read_1..10` 为预留 key，当前版本不会触发

## 流程与页面

| key | 触发时机 |
|---|---|
| `mode_enter` | 进入卫戍协议（非搜索确认后中断） |
| `quick_leave` | 搜索确认后未匹配就返回主页 |
| `match_found` | 匹配成功，进入确认/策略阶段 |
| `strategy_select` | 策略选择页（含核心盟约标题） |
| `decision_select` | 决策/盟约等页面的「选择时刻」 |
| `secret_core` | 隐秘核心解锁 |
| `settle_return_home` | 结算页之后回到主页 |
| `run_victory` | 模拟完成（本局胜利） |
| `run_defeat` | 模拟中断（本局失败） |

## 战斗

| key | 触发时机 |
|---|---|
| `battle_enter` | 准备就绪后进入战斗 |
| `battle_perfect` | 完美作战（无扣血） |
| `battle_small` | 轻微扣血（1–3）且未濒死 |
| `battle_heavy` | 重创（≥4） |
| `near_death` | 扣血后生命 1–3 |
| `battle_lethal` | 生命 ≤0 |
| `final_start` | 最终攻势开始 |
| `boss_hp_75/50/25/10` | 首领血条里程碑 |
| `time_60/30/10` | 最终攻势剩余时间 |
| `boss_victory` | 击败首领 |
| `defense_enter` | 进入联防阶段 |

## 联防结果

`defense_<role>_<result>`：

- role：`help` = 玩家协防队友；`helped` = 队友协防玩家；`peer` = 队友之间互相联防
- result：`full` = 协防对象无掉血；`basic` = 少量掉血；`fail` = 失败

共 9 个：`defense_help_full/basic/fail`、`defense_helped_full/basic/fail`、`defense_peer_full/basic/fail`。

## 结算与休整

| key | 触发时机 |
|---|---|
| `perfect_settlement` | 完美结算 |
| `all_fail_encourage` | 全员不完美，进入下一轮前鼓励 |
| `rest_normal` | 普通休整开始 |
| `rest_boss` | Boss 前休整开始 |
| `idle` | 休整期长时间无操作 |
| `idle_lobby` | 主页长时间停留 |

## 商店

| key | 触发时机 |
|---|---|
| `buy_low` | 购买低阶干员（徽章 I–III） |
| `buy_high` | 购买高阶干员（徽章 IV–VI） |
| `buy_item` | 购买道具 |
| `sell` | 售出干员（金币上涨且出现售卖 UI） |
| `refresh` | 连续刷新彩蛋 |
| `level_up` | 升级确认完成 |
| `can_upgrade` | 可升级（青面板 + 「升级」） |
| `freeze` | 冻结招募区 |
| `unfreeze` | 解冻招募区 |
| `triple_reward` | 三连奖励 |
| `same_operator` | 商店出现同名干员 |
