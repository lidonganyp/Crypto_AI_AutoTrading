# Crypto AI AutoTrading

Crypto AI AutoTrading 是一个面向加密货币市场的 AI 辅助量化交易系统。项目集成了行情采集、特征工程、XGBoost 预测、LLM 辅助研究、风险控制、纸面交易、受保护的实盘执行、回测、走步验证、运行监控和 Streamlit 仪表盘。

本项目更适合作为量化交易系统工程、策略研究、风控流程和自动化运维的开源参考，不应被理解为“自动盈利”工具。

## 风险提示

加密货币交易具有高波动、高杠杆、高滑点和交易所接口不可用等风险。本项目默认以纸面交易模式运行；即使启用实盘模式，也必须同时设置 `RUNTIME_MODE=live` 和 `ALLOW_LIVE_ORDERS=true` 才会尝试真实下单。

使用前请自行审计代码、配置 API 权限、限制交易额度，并先在隔离数据库和纸面环境中验证。项目作者不对任何资金损失负责，本文档和代码均不构成投资建议。

## 核心能力

- 行情采集：通过 `ccxt` 接入 OKX / Binance 等交易所行情，并提供降级和缓存处理。
- 特征工程：构建技术指标、市场状态、流动性、新闻和宏观上下文等多源特征。
- 模型预测：使用 XGBoost 进行方向和质量判断，支持训练、评估、回测和走步验证。
- LLM 研究：支持 DeepSeek、Qwen 及 OpenAI 兼容接口，用于辅助市场研究和交易理由生成。
- 风险控制：包含仓位控制、相关性控制、熔断、滑点检查、资金费率和延迟拦截等保护。
- 纸面交易：内置 paper trader，可在 SQLite 中记录交易、预测、持仓和运行状态。
- 实盘保护：实盘路径默认关闭，需要显式配置才会启用真实订单。
- 策略演化：`nextgen_evolution` 提供候选策略、修复、晋级、组合分配和运行证据管理。
- 运维监控：提供健康检查、风控报告、漂移报告、事故报告、指标报告和每日摘要。
- Web 仪表盘：基于 Streamlit 展示概览、设置、预测和运维状态。

## 项目结构

| 路径 | 说明 |
| --- | --- |
| `main.py` | CLI 入口，负责一次性运行、循环运行、训练、报告和运维命令 |
| `dashboard.py` | Streamlit 仪表盘入口 |
| `config/` | 运行配置和环境变量解析 |
| `core/` | 主运行时、存储、评分、报告、模型生命周期和仪表盘数据服务 |
| `analysis/` | 技术分析、新闻、宏观、链上、研究 LLM 和信号融合 |
| `strategy/` | 风控、仓位、相关性、模式库、XGBoost 预测和策略编排 |
| `execution/` | 交易所适配、纸面交易、实盘交易、订单管理和对账 |
| `backtest/` | 回测、真实交易成本模拟和走步验证 |
| `learning/` | 经验存储、反思和运行参数自适应 |
| `monitor/` | 健康、漂移、失败、事故、归因和运维报告 |
| `nextgen_evolution/` | 策略候选、晋级、修复、组合分配和生命周期管理 |
| `tests/` | `unittest` 测试套件 |
| `deploy/` | systemd 等部署辅助文件 |
| `xuqiu/` | 项目需求和设计说明 |

## 环境要求

- Python 3.10 或更高版本。
- 推荐 Linux / macOS 环境，Windows 可通过 PowerShell 或 WSL 运行。
- 需要可访问交易所公开行情接口；部分地区可能需要代理。
- 如使用 LLM、链上、新闻或告警能力，需要配置对应 API key。

## 快速开始

```bash
git clone https://github.com/lidonganyp/Crypto_AI_AutoTrading.git
cd Crypto_AI_AutoTrading

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
copy .env.example .env
```

建议先保持默认纸面模式：

```env
RUNTIME_MODE=paper
ALLOW_LIVE_ORDERS=false
DB_PATH=data/cryptoai.db
```

运行一次完整分析和纸面执行流程：

```bash
python main.py once
```

使用临时数据库做隔离验证：

```bash
DB_PATH=/tmp/cryptoai-smoke.db RUNTIME_MODE=paper ALLOW_LIVE_ORDERS=false python main.py once
```

## 常用命令

```bash
python main.py once
python main.py loop
python main.py train
python main.py report
python main.py backfill 180
python main.py backtest BTC/USDT
python main.py walkforward BTC/USDT
python main.py validate BTC/USDT,ETH/USDT
python main.py reconcile
python main.py health
python main.py guards
python main.py drift
python main.py metrics
python main.py live-readiness
python main.py ops
python main.py execution-pool
python main.py execution-rebuild
python main.py schedule once
python main.py daemon
```

更多运维入口包括：`positions`、`entries`、`approve-recovery`、`abtest`、`maintenance`、`failures`、`incidents`、`alpha`、`attribution`、`watchlist-refresh`、`execution-set`、`execution-add`、`execution-remove`、`init-system`、`cleanup-data`。

## 仪表盘

```bash
streamlit run dashboard.py
```

仪表盘默认面向本地运维使用，主要页面包括 Overview、Settings、Predictions 和 Ops。不要把 Streamlit 直接暴露到公网；如果需要远程访问，请使用 SSH 隧道或带认证的反向代理。

## Docker 部署

```bash
docker compose up -d --build
```

默认 compose 文件会启动：

- `cryptoai-engine`：调度和交易系统服务。
- `cryptoai-dashboard`：Streamlit 仪表盘，默认绑定 `127.0.0.1:8501`。

轻量云服务器可使用 Tencent Cloud 配置文件：

```bash
docker compose -f docker-compose.tencent-lite.yml up -d --build
docker compose -f docker-compose.tencent-lite.yml --profile dashboard up -d --build
```

## 关键配置

所有本地配置从 `.env` 读取，`.env` 不应提交到 Git。请从 [.env.example](./.env.example) 复制后自行填写。

| 变量 | 说明 |
| --- | --- |
| `DB_PATH` | SQLite 数据库路径 |
| `RUNTIME_MODE` | `paper` 或 `live` |
| `ALLOW_LIVE_ORDERS` | 是否允许真实下单，默认必须为 `false` |
| `EXCHANGE_PROVIDER` | 交易所提供方，默认 `okx` |
| `EXCHANGE_PROXY_URL` | 交易所接口代理地址，可为空 |
| `OKX_API_KEY` / `OKX_API_SECRET` / `OKX_API_PASSPHRASE` | OKX API 凭据，仅实盘或私有接口需要 |
| `DEEPSEEK_API_KEY` / `QWEN_API_KEY` | LLM API 凭据，可选 |
| `CRYPTOPANIC_API_KEY` / `GLASSNODE_API_KEY` / `COINMETRICS_API_KEY` / `LUNARCRUSH_API_KEY` | 新闻、链上和外部数据源凭据，可选 |
| `FEISHU_WEBHOOK_URL` / `FEISHU_WEBHOOK_SECRET` | 飞书告警配置，可选 |

实盘运行前必须完成至少以下检查：

- 使用只读或最小权限 API key 做验证。
- 确认 `python main.py live-readiness` 通过。
- 确认交易所 API 权限、IP 白名单、资金规模和最大仓位限制。
- 确认 `RUNTIME_MODE=live` 与 `ALLOW_LIVE_ORDERS=true` 是明确、有意设置的。

## 测试

```bash
python -m unittest discover -s tests -v
```

本项目当前测试使用 Python 标准库 `unittest`。在提交改动前，建议至少运行完整测试和源码编译检查：

```bash
python -m compileall -q analysis backtest config core execution learning monitor nextgen_evolution scripts strategy tests main.py dashboard.py
```

## 数据与安全

- `.env`、`data/`、`logs/`、`.venv/`、缓存文件、数据库文件默认被 `.gitignore` 排除。
- 不要提交交易所 API key、LLM key、Webhook secret、数据库、交易记录或日志。
- 默认 Docker 忽略规则会排除 `.env`、`data/` 和 `logs/`，避免敏感文件进入 build context。
- 实盘交易建议使用独立子账户、低额度资金、IP 白名单和最小权限 API key。

## 当前边界

- 项目不是高频交易系统，不保证实时性。
- 公共行情、新闻、链上和 LLM 接口可能不可用，系统会尽量降级，但结果质量会受影响。
- XGBoost 和 LLM 输出只是决策输入，不代表确定收益。
- 实盘执行路径虽然带保护，但仍需用户自行审计和承担风险。

## 贡献

欢迎提交 Issue 和 Pull Request。建议贡献前先运行测试，并尽量提供：

- 清晰的问题描述或改动目标。
- 可复现步骤、样例配置或测试数据。
- 对风险控制、实盘行为或数据结构变更的影响说明。
- 对应测试或验证命令结果。

## 许可证

当前仓库尚未添加开源许可证文件。正式开源前建议添加 `LICENSE`，例如 MIT、Apache-2.0 或 GPL-3.0。没有许可证时，默认并不等于允许他人自由复制、修改和分发。

## 免责声明

本项目仅用于技术研究、学习和工程实践。加密资产交易风险极高，任何自动化策略都可能因为市场波动、模型误判、接口异常、滑点、流动性不足、交易所故障或配置错误造成损失。使用者应自行承担全部风险。
