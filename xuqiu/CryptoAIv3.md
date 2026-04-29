# CryptoAI v3 需求文档（按当前代码重写）

## 1. 文档定位

本文件不是早期蓝图，也不是未来设想，而是基于当前仓库已实现代码整理出的“现状即规范”版本。

- 项目目标：长期稳定运行于真实环境，最终支持自动真实交易，并以长期正期望和可控回撤为核心目标
- 产品原则：自动化优先、风控优先、可观测优先、纸面指标服从真实可执行性
- 代码主路径：
  - `main.py`
  - `core/engine.py`
  - `config/__init__.py`
  - `execution/*`
  - `analysis/*`
  - `learning/*`
  - `monitor/*`
  - `dashboard.py`

当前文档版本：`v3.3`  
更新时间：`2026-04-07`

### 1.1 当前架构借鉴来源

当前代码虽然是加密交易系统，但架构上已经开始明确吸收几类主流量化平台的设计思想：

- 借鉴 `QMT / PTrade`：
  - 将“策略逻辑”与“运行托管方式”分开
  - 将本地执行、云端托管、纸面验证、实盘守卫视为不同运行画像
- 借鉴 `Qlib`：
  - 强化研究、模型、评估、执行、复盘的分层边界
  - 尽量让特征、预测、评估、学习反馈可重复、可追踪
- 借鉴 `vn.py`：
  - 将核心引擎与策略画像/应用层分离
  - 避免把不同交易风格的规则散落在多个运行时文件中

因此当前版本新增一条架构约束：

- `entry_thesis / entry_thesis_strength / entry_open_bias` 不再只由单个 runtime service 临时拼接
- 系统引入统一的 `strategy profile` 概念，用同一套画像推导同时服务于入场元数据和持仓守护
- 其目标是减少“入场是一套规则、持仓又是一套规则”的漂移风险

---

## 2. 产品目标

### 2.1 核心目标

CryptoAI v3 的目标不是做“看起来很聪明”的 LLM 交易演示，而是做一套能长期运行、自动执行、可验证、可收敛的加密交易系统。

系统目标分三层：

1. 长期稳定运行  
   要求服务能持续轮询、自动调度、自动恢复、自动对账、自动产出监控和报告。

2. 自动完成研究、决策与执行  
   要求系统在不依赖人工逐笔判断的前提下，完成数据采集、研究分析、风险审批、开平仓和复盘。

3. 逐步形成真实正期望  
   要求系统通过历史反思、实时评估、动态阈值和 setup 级别风控，持续压缩低质量交易，优先提升真实可落地收益质量。

### 2.2 非目标

以下内容不属于当前系统目标：

- 不承诺固定收益
- 不追求高频交易
- 不追求复杂多代理 LLM 辩论主链
- 不做 Meme / 土狗币
- 不做高杠杆投机
- 不把回测收益作为放开实盘的唯一依据

---

## 3. 当前系统边界

### 3.1 当前运行模式

系统支持两种运行模式：

- `paper`
- `live`

当前代码逻辑中：

- `RUNTIME_MODE=paper` 使用 `PaperTrader`
- `RUNTIME_MODE=live` 使用 `LiveTrader`
- 只有当 `ALLOW_LIVE_ORDERS=true` 时，`LiveTrader` 才会真正下真实订单
- 即使进入 `live`，系统仍会先执行 live readiness 校验，不满足条件会直接拒绝启动

### 3.2 当前交易方向

当前主策略路径是：

- 开仓只做 `LONG`
- 没有主动做空路径
- `FLAT/HOLD` 是最常见的默认输出

### 3.3 当前交易所与市场数据

当前代码支持：

- 行情采集：`OKX` / `Binance`
- 执行适配器：`OKXExchangeAdapter` / `BinanceExchangeAdapter`
- 默认市场数据提供方：`OKX`

### 3.4 当前持久化

当前系统使用 SQLite 作为运行时数据库，核心表包括：

- `ohlcv`
- `feature_snapshots`
- `prediction_runs`
- `trades`
- `positions`
- `orders`
- `account_snapshots`
- `training_runs`
- `walkforward_runs`
- `execution_events`
- `reconciliation_runs`
- `reflections`
- `scheduler_runs`
- `report_artifacts`
- `system_state`

### 3.5 当前操作界面

当前 dashboard 已被收缩为自动化主流程视图，仅保留 4 个核心页：

- `Overview`
- `Settings`
- `Predictions`
- `Ops`

训练、回测、日志、调度、Watchlist 手工操作等能力仍保留在代码和 CLI 中，但不再作为主界面的默认操作入口。

---

## 4. 当前系统主流程

系统每个分析周期执行的核心流程如下。

### 4.1 周期启动

`run_once()` 启动时会先做以下动作：

1. 刷新学习型 runtime overrides
2. 应用 runtime overrides
3. 读取当前执行池与活跃池
4. 判断是否需要自动重建执行池
5. 写入周期状态和锁文件

### 4.2 自动学习层刷新

系统在每轮周期前都会自动生成学习型运行时参数，来源包括：

- `reflections` 中的已平仓结果
- `prediction_runs` 的已验证方向准确率
- `paper_canary` 平仓/部分平仓后的即时反馈刷新

学习层当前会自动作用于以下参数：

- `xgboost_probability_threshold`
- `final_score_threshold`
- `min_liquidity_ratio`
- `sentiment_weight`
- `blocked_setups`

学习层的目标不是放大交易，而是自动收紧：

- 在 `EXTREME_FEAR` 且方向准确率差时，提高 `xgboost` 和 `final score` 门槛
- 在弱流动性 setup 表现差时，提高 `min_liquidity_ratio`
- 在极端恐慌阶段降低甚至清零 `sentiment_weight`
- 对近期真实负贡献的 setup 直接生成 `pause_open`

当前版本新增一条重要运行时约束：

- 学习层不再只在下一个完整分析周期生效
- `paper_canary` 的部分平仓与全部平仓会立即触发 learning refresh
- refresh 完成后会立刻重算 runtime effective config，并进入后续执行池重建逻辑

### 4.3 Runtime Override 合并规则

系统当前支持三类运行时参数来源：

- `default`
- `manual`
- `learning`

默认策略是：

- 自动优先
- 手动参数默认可以被学习层覆盖
- 只有显式锁定的字段，手动值才会压住学习层

状态通过以下 system state 持久化：

- `runtime_settings_overrides`
- `runtime_settings_learning_overrides`
- `runtime_settings_locked_fields`
- `runtime_settings_override_conflicts`
- `runtime_settings_effective`

### 4.4 自动执行池重建

系统不再把“训练成功”直接等价为“可交易”。

当前执行池会根据以下条件自动重建：

- 模型文件存在且满足最小训练样本
- 最近真实预测评估达到最小样本
- symbol 的近期方向准确率高于最低门槛
- 按准确率、样本数、核心币优先级排序

同时，模型就绪判定已补充以下约束：

- 运行时 predictor 会优先读取该 symbol 最新 `training_runs` 对应的 `model_path`
- `model_ready_symbols` 与运行时 predictor 使用同一模型路径来源
- 已被标记为坏模型且模型文件签名未变化的 symbol，会被临时排除出 ready / active pool

系统当前已支持坏模型状态跟踪：

- `broken_model_symbols`

其作用不是做人工告警展示，而是为了：

- 防止损坏模型继续进入 active pool
- 防止执行池把“有训练记录但模型不可用”的 symbol 误判为可交易
- 为后续训练链自愈提供待修复队列

当前执行池相关状态包括：

- `execution_symbols`
- `model_ready_symbols`
- `edge_qualified_symbols`
- `active_symbols`

系统支持手动命令：

- `execution-pool`
- `execution-set`
- `execution-add`
- `execution-remove`
- `execution-rebuild`

但这类手动池管理已不属于主 UI 的核心操作流。

### 4.5 数据采集与质量校验

每个 symbol 分析前，系统会抓取：

- `1h` / `4h` / `1d` K 线
- 最新市场情绪与 Fear & Greed
- 新闻摘要
- 宏观摘要
- 链上摘要
- 资金费率
- 可选盘口深度

系统会在进入预测前做多层校验：

- 缺失 K 线直接跳过
- 数据质量失败直接跳过
- 资金费率过热直接拦截
- 明显利空新闻可触发新闻冷却
- 市场数据连续失败可触发 circuit breaker

### 4.6 特征、研究、预测

系统的核心分析链是：

1. `FeaturePipeline` 构造特征
2. `ResearchLLMAnalyzer` 生成结构化研究输出
3. `XGBoostPredictor` 生成上涨概率
4. `CrossValidationService` 做多源一致性校验
5. `ResearchManager` 做二次审批
6. `DecisionEngine` 生成开仓/观望决策

当前设计中：

- LLM 负责信息梳理，不单独决定交易
- XGBoost 是主要方向概率模型
- 最终是否开仓由研究、预测、交叉验证、风控、组合评分共同决定

当前版本新增了模型可用性约束：

- 若 XGBoost predictor 实际退回到 `fallback_v2`
- 则该 symbol 本轮不会进入正常分析结果落库
- 不会写入新的 `prediction_runs`
- 不会触发新开仓
- 同时会登记 `model_unavailable` 事件，并把 symbol 放入坏模型自愈队列

该设计的目标是：

- 避免启发式 fallback 污染实时 edge 与准确率统计
- 避免“模型坏了但系统仍继续开仓”的实盘风险
- 让 paper / live 的行为口径尽量接近“真实模型可用时”的决策逻辑

### 4.7 坏模型自愈链

当前系统已经具备坏模型自动修复链，但修复动作不再阻塞交易热路径。

当前流程为：

1. 周期分析时若发现 `fallback_v2`
2. 该 symbol 被标记进 `broken_model_symbols`
3. 当前周期直接跳过该 symbol 的正常预测落库与交易决策
4. 后续训练链会把 `broken_model_symbols` 并入训练候选集
5. 若达到修复间隔条件，则自动重训该 symbol
6. 重训成功后自动清除坏模型状态，symbol 才重新回到 model-ready / active pool

当前设计约束：

- 自愈补训不在 `snapshot` 热路径同步执行
- 避免单个坏模型把整轮 `run_once()` 卡住
- 修复频率受训练调度间隔限制，避免失败模型高频重试

### 4.8 Loop 内模型维护

当前版本新增了 loop 结束后的轻量模型维护逻辑：

1. 每轮 `run_once()` 完成后，会检查是否存在坏模型
2. 若存在坏模型，则自动触发一次非阻塞的训练维护检查
3. 即使没有坏模型，也会按训练调度间隔做节流检查
4. 该维护逻辑不会中断当前周期状态写入

该设计的目标是：

- 避免 `main.py loop` 长期运行但无人手工执行 `train` 时，坏模型永远停留在降级状态
- 让 loop-only 部署也具备最小化的模型自愈能力
- 不把训练动作塞回分析热路径

### 4.9 模型文件原子写入

当前训练链已经补充模型文件原子写入约束：

1. 训练输出先写入带同后缀的临时文件
2. 写入成功后用 `os.replace()` 原子替换正式模型文件
3. metadata 文件也使用原子写入

该设计的目标是：

- 避免训练与推理并发时读到半写入模型文件
- 避免一次训练失败把原有可用模型破坏成坏文件
- 降低 `broken_model_symbols` 的偶发误触发概率

---

## 5. 决策与风控逻辑

### 5.1 Cross Validation

交叉验证当前关注：

- 新闻覆盖是否过薄
- 新闻服务是否不可用
- 恐惧贪婪指数与 LunarCrush 是否冲突
- 链上净流向与 regime 是否冲突
- 价格表现与 regime 是否冲突

当前实现已经从“无新闻直接误杀”调整为：

- `thin news` 只做轻微降权
- 真正的多源冲突才阻断

### 5.2 Research Manager

Research Manager 是当前主决策链的核心二次审批层。

输入包括：

- 原始研究结果
- XGBoost 概率
- 交叉验证结果
- 流动性
- 趋势
- fear & greed
- 新闻风险
- 链上可用性
- 历史经验与 setup 统计
- 学习层 blocked setups

输出包括：

- `reviewed_action`
- `approval_rating`
- `review_score`
- `setup_profile`
- `setup_performance`

### 5.3 自动暂停低质量 Setup

当前系统已经支持自动暂停一类 setup 的开仓，而不是只调阈值。

实现方式：

- 学习层根据最近预测质量生成 `blocked_setups`
- 学习层可根据最近真实已平仓表现生成 `blocked_setups`
- 当前 pause 既支持通用 setup 级规则，也支持按 `symbol + setup_profile` 粒度的近期封禁

Research Manager 命中这类 setup 时会：

- 保留研究结果
- 标记 `setup_auto_pause`
- 将原本可开的 `OPEN_LONG` 压成 `HOLD`

重要约束：

- `setup_auto_pause` 只影响新开仓
- 不会误伤已有仓位的正常退出 / 减仓逻辑
- 正向 `shadow_trade_runs` 结果可以对冲同类被封 setup，触发 `resume_open`

### 5.4 Decision Engine

Decision Engine 当前综合：

- `xgboost_probability_threshold`
- `final_score_threshold`
- `sentiment_weight`
- `min_liquidity_ratio`
- 趋势过滤
- 波动率因子
- 研究是否 fallback
- Research Manager 输出的 `suggested_action`
- 模型是否真实可用

当前系统会大量输出 `FLAT/HOLD`，这是设计结果，不是异常。

当前版本新增一条明确规则：

- `fallback*` 与 `invalid_features` 都视为模型不可用
- 模型不可用时，Decision Engine 不允许新开仓
- 决策原因会写成 `model unavailable`

### 5.4.1 LLM 运行时网络与退避

当前版本补充了两条与 LLM 可用性相关的运行时约束：

1. LLM 客户端显式忽略宿主机 `HTTP_PROXY / HTTPS_PROXY / ALL_PROXY`
2. 若 LLM 在运行时出现 401、限流、上游 5xx 或网络错误，会进入短时 backoff，而不是每个 symbol / 每轮周期都继续重试

该设计的目标是：

- 避免本机代理格式不兼容时直接拖垮研究层初始化
- 避免上游瞬时异常放大成整轮日志风暴和延迟抖动
- 保持 `fallback_research_model` 作为降级路径，而不是作为高频重复失败路径

### 5.5 账户与仓位风险

Risk Manager 当前负责：

- 单笔风险控制
- 单币暴露上限
- 总暴露上限
- 最大持仓数
- 单日/单周亏损限制
- 最大回撤限制
- 连续亏损冷静期
- 连续盈利/亏损动态仓位因子

当前默认约束来自 `config/__init__.py`：

- `single_trade_risk_pct = 0.5%`
- `max_positions = 3`
- `daily_loss_limit_pct = 2%`
- `weekly_loss_limit_pct = 5%`
- `max_drawdown_pct = 12%`

当前版本补充了账户口径上的两条重要约束：

1. `paper` 模式下，权益仍按 `initial_balance + realized + unrealized` 计算
2. `live` 模式下，权益优先使用交易所 quote 余额 + 本系统托管持仓市值，而不再锚定本地固定初始资金

其目标是：

- 避免 live 风控继续使用固定本地初始资金口径
- 让实盘 drawdown、daily loss、仓位预算与真实账户规模一致
- 让 paper / live 在风险计算上保持“同逻辑、不同资金来源”

### 5.6 持仓退出逻辑

当前已有持仓的退出逻辑包括：

- `fixed_stop_loss`
- `take_profit_1`
- `take_profit_2`
- `trend_reversal`
- `time_stop`
- `research_exit`
- `cross_validation_exit`
- `portfolio_de_risk`
- `bearish_news_exit`

其中：

- `research_exit` 倾向直接平仓
- `cross_validation_exit` / `portfolio_de_risk` 倾向减仓或部分平仓
- `setup_auto_pause` 不再触发已有仓位的 `portfolio_de_risk`

当前版本进一步补充：

- 部分平仓的已实现盈亏会进入账户权益、日内盈亏、周内盈亏和日报统计
- 事件层通过 `incremental_pnl` 单独记录每次部分平仓的真实增量盈亏
- 对高质量 `paper_canary` / 强 thesis setup，`research_exit` 不再只靠短周期连续降级立即强平
- `research_exit` 的确认次数只有在满足最短持有时间后才生效
- 持仓复核期间若选择继续观察，会显式写入 `position_review_watch` 事件
- 持仓复核回退路径会自动尝试有/无 `:USDT` 的本地 K 线符号变体，避免因为存储符号口径不一致导致退出判断退化

其目标是：

- 避免账户权益与 performance 报告、日报口径不一致
- 避免部分止盈后系统错误低估真实收益与剩余风险

---

## 6. 执行层要求

### 6.1 Paper 执行

`PaperTrader` 是当前主执行路径，特点：

- 只支持 `LONG`
- 自动记录 `trades`、`positions`、`orders`
- 支持部分平仓
- 自动计算已实现盈亏

### 6.2 Live 执行

`LiveTrader` 已具备真实下单框架，但仍属于 guarded live path。

当前 live 路径具备：

- 余额检查
- 滑点守卫
- 订单超时
- 限价重试
- live order 事件记录
- 失败事件写库

当前 live 路径仍然受以下门控限制：

- `runtime_mode = live`
- `allow_live_orders = true`
- live readiness 通过

### 6.3 对账与人工恢复

系统当前具备对账与人工恢复阻断机制：

- `Reconciler` 定期对账
- mismatch 可触发 circuit breaker
- 部分错误场景会进入 `manual_recovery_required`
- 必须显式 `approve-recovery` 才能恢复

当前版本对对账边界做了收敛，默认更适配共享账户或存在外部持仓/外部挂单的场景：

- 交易所余额高于本系统记录的情况，默认不视为 mismatch
- 交易所存在额外挂单，但本系统没有对应本地挂单时，默认不视为 mismatch
- 真正危险的情况仍然会触发 mismatch：
  - 本地记录有仓位，但交易所余额不足
  - 本地记录有挂单，但交易所缺少对应挂单或挂单数量不足

该设计的目标是：

- 避免共享账户、人工干预、dust 余额导致误熔断
- 保留“系统以为自己有仓位/挂单，但交易所实际没有”的关键风控能力

---

## 7. 学习闭环

### 7.1 Reflection

每笔交易结束后，系统会产出 reflection，内容包括：

- 正确信号
- 错误信号
- lesson
- market regime

当前支持：

- 规则反思
- 可选 LLM 反思

### 7.2 Setup 级别经验

系统当前已经把 setup 信息编码到交易 rationale 中，用于后续统计：

- `symbol`
- `regime`
- `validation`
- `liquidity_bucket`
- `news_bucket`

这使得系统可以在后续做两类学习：

1. `reflection` 驱动的已成交经验学习
2. `prediction_runs` 驱动的 bootstrap 学习

### 7.3 当前学习层实际效果

当前学习层已经能在真实数据库上自动生成：

- `xgboost_probability_threshold`
- `final_score_threshold`
- `min_liquidity_ratio`
- `sentiment_weight`
- `blocked_setups`

它不再只是“提出建议”，而是已经进入 runtime effective config。

当前版本新增两类更强的在线学习闭环：

1. `soft_review` 去噪

- `soft_review` 不再作为广泛试单入口
- 只有满足更强 near-miss 质量门槛的 `soft_review` 才允许真实 `paper_canary`
- 弱 `soft_review` 保留在 shadow / 归因链，不再默认真实占仓

2. 近期负贡献 setup pause

- 学习层会按最近窗口内的真实已平仓结果，快速识别持续负贡献 setup
- 默认按 `symbol + regime + validation + liquidity_bucket + news_bucket` 粒度生成 `pause_open`
- `paper_canary` 平仓后会立即刷新这一结果，不再必须等待下一轮完整分析周期

---

## 8. 监控、报告与可观测性

### 8.1 Dashboard

当前 dashboard 已按自动化流程压缩：

- `Overview`
  - 资金、持仓、回撤、最近周期、准确率趋势
- `Settings`
  - 自动化摘要
  - 当前 effective runtime
  - learning reasons
  - blocked setups
  - 高级手动控制（折叠）
- `Predictions`
  - 最近 prediction runs
  - 决策链
  - 关键理由
  - 最新 intelligence 因子
  - 最近 research reviews
- `Ops`
  - 运维总览
  - reconciliation
  - scheduler / ops artifact

### 8.2 核心报告

系统当前可生成：

- `health`
- `guards`
- `metrics`
- `ops`
- `alpha`
- `attribution`
- `failures`
- `incidents`
- `daily / weekly report`
- `drift`
- `abtest`

### 8.3 关键状态项

系统通过 `system_state` 暴露的关键状态包括：

- `model_degradation_status`
- `model_degradation_reason`
- `last_accuracy_guard_triggered`
- `runtime_settings_effective`
- `runtime_settings_learning_details`
- `runtime_settings_override_conflicts`
- `broken_model_symbols`
- `execution_symbols`
- `active_symbols`
- `model_ready_symbols`

当前版本新增的状态/事件语义包括：

- `last_loop_model_maintenance_at`
- `model_self_heal`
- `paper_canary_open`
- `incremental_pnl`（作为 `close/live_close` 事件 payload 字段）
- `position_review_watch`
- `learning_feedback_refresh`
- `runtime_settings_learning_details.blocked_setups`

---

## 9. CLI 与运维入口

### 9.1 核心自动化命令

推荐保留并重点使用的命令：

- `python main.py once`
- `python main.py loop`
- `python main.py health`
- `python main.py guards`
- `python main.py metrics`
- `python main.py ops`
- `python main.py alpha`
- `python main.py attribution`
- `python main.py reconcile`
- `python main.py report`
- `python main.py validate BTC/USDT,ETH/USDT`

### 9.2 仍然可用但非主流程命令

以下命令仍然实现，但不属于主 UI：

- `train`
- `backtest`
- `walkforward`
- `watchlist-refresh`
- `execution-set`
- `execution-add`
- `execution-remove`
- `execution-rebuild`
- `daemon`
- `schedule`

---

## 10. 当前 live readiness 边界

系统目前已经具备 live gating，但默认仍不建议直接放开真实交易，原因不是功能缺失，而是绩效尚未达标。

当前 live readiness 关注：

- closed trades 数量
- prediction eval 样本数
- XGBoost accuracy
- fusion accuracy
- holdout accuracy
- total realized pnl
- drawdown
- research fallback ratio

换句话说，当前系统已经逐步具备“自动化实盘框架”，但还处于“自动学习收紧 + 纸面验证 + 小步收敛”阶段，而不是“大仓位真实放开”阶段。

---

## 10.1 当前验证样本扩展机制

为避免“长期不开仓 → 无法验证 → 持续不开仓”的死循环，当前代码已补充两条样本扩展机制：

### A. Near-Miss Shadow

系统会把“接近开仓阈值但被拦下”的机会额外记录为 shadow trade，典型场景包括：

- 接近 `xgboost_probability_threshold`
- 接近 `final_score_threshold`
- 接近可开仓区间但最终被 `risk guard` 拦下
- 接近可开仓区间但最终被研究审批压成 `HOLD/CLOSE`

其目标是：

- 即使真实 0 开仓，也能积累“差一点就开”的反事实样本
- 帮助判断当前阈值是在正确过滤，还是在错误杀伤

### B. Paper Canary

系统支持仅在 `paper` 模式开启的 canary 轨道：

- 当前默认开启
- 仅允许较小仓位，并区分 `primary_review / offensive_review / soft_review`
- `soft_review` 已被显著收紧，只保留高质量 near-miss
- 其目标是快速增加可评估样本，同时避免低质量试单持续占用仓位

当前代码中：

- `near_miss_shadow_enabled` 默认开启
- `paper_canary_enabled` 默认开启
- `paper_canary_soft_review_min_score` 已显著收紧
- `paper_canary_soft_position_scale` 已下调，避免 `soft_review` 主导真实样本

---

## 11. 当前主需求结论

基于当前代码，CryptoAI v3 的真实需求可以归纳为：

1. 它首先是一套自动化风控和自动化收敛系统，而不是一套激进追单系统。
2. LLM 的角色是研究辅助，不是最终拍板者。
3. 真正的决策核心是：
   - 预测概率
   - 研究审批
   - setup 级别风控
   - runtime 动态阈值
4. 系统界面应服务于“是否能继续自动跑”这一问题，而不是展示大量低频、低关注度模块。
5. 当前系统已明确转向“alpha 优先于 beta 收缩”的路线，但实现方式不是放松风控，而是：
   - 提升高质量样本密度
   - 压缩 `soft_review` 主导的低质量试单
   - 用更快的在线学习闭环暂停近期负贡献 setup
   - 用更长的 thesis 持有窗口减少超短 `research_exit` 负 churn
   - 在保持风控刚性的前提下逐步接近真实正期望

---

## 12. 后续文档维护规则

今后更新本文件时遵循以下规则：

1. 以代码当前实现为准，不写未实现功能当作现状
2. 动态 runtime 数值写“逻辑来源”，少写易过期的瞬时状态
3. UI 文档只描述主界面保留页，不把隐藏功能写成主流程
4. 对自动学习、风控、执行池、模型自愈、live readiness 的任何重大调整，都必须同步更新本文件
