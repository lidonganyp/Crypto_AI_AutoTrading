## 改动概述

请简要说明本次 PR 解决的问题和主要改动。

## 影响范围

- [ ] 文档或注释
- [ ] 测试或工具
- [ ] 数据采集或分析
- [ ] 模型、训练或回测
- [ ] 风控、执行或实盘路径
- [ ] 仪表盘或运维报告

## 风险检查

- [ ] 不包含 `.env`、API key、secret、数据库、日志或交易敏感信息
- [ ] 没有绕过 `ALLOW_LIVE_ORDERS`
- [ ] 如影响实盘路径，已说明安全边界和降级行为

## 验证

请列出已运行的命令和结果。

```bash
python -m unittest discover -s tests -v
python -m compileall -q analysis backtest config core execution learning monitor nextgen_evolution scripts strategy tests main.py dashboard.py
```
