# 贡献指南

感谢你关注 Crypto AI AutoTrading。这个项目涉及交易、风控、数据源、模型和自动化执行，任何改动都应优先保证可解释性和安全边界。

## 贡献方式

- 提交 Issue：报告 bug、提出功能建议、补充运行环境差异或数据源问题。
- 提交 Pull Request：修复问题、补充测试、改进文档、优化工程结构或增加受保护的新能力。
- 参与验证：在纸面交易或隔离数据库中复现实验结果，并提供关键日志和配置说明。

## 开发流程

1. Fork 仓库并创建功能分支。
2. 从 `.env.example` 复制 `.env`，不要提交真实凭据。
3. 尽量使用隔离数据库验证改动，例如 `DB_PATH=/tmp/cryptoai-dev.db`。
4. 为行为变化补充或更新测试。
5. 提交 PR 前运行测试和基本编译检查。

## 推荐检查命令

```bash
python -m unittest discover -s tests -v
python -m compileall -q analysis backtest config core execution learning monitor nextgen_evolution scripts strategy tests main.py dashboard.py
```

## Pull Request 要求

- 说明改动目标和用户可见影响。
- 标明是否影响实盘交易、下单路径、仓位、风控或数据库结构。
- 列出已执行的验证命令和结果。
- 不要把 `.env`、数据库、日志、模型产物、API key 或 webhook secret 放入提交。
- 涉及实盘行为的改动必须默认保持保守，不能绕过 `ALLOW_LIVE_ORDERS`。

## 代码风格

- 保持实现清晰、可测试、可回滚。
- 优先复用现有模块和运行时配置，不引入不必要的全局状态。
- 新增风控或执行逻辑时，应说明失败模式和降级行为。
- 交易相关日志不要输出完整密钥、账户信息或敏感订单上下文。

## 安全边界

任何涉及真实下单、API 凭据、远程访问、告警 webhook 或权限控制的问题，请优先参考 [SECURITY.md](./SECURITY.md)。
