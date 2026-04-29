# Tencent Cloud Linux Deployment

This profile is tuned for a small server such as 2 vCPU / 4 GB RAM.

## Recommended Mode

- run the engine continuously
- keep live trading disabled until paper mode is stable
- start the dashboard only when needed

## Environment

Start from `.env.example` and set at least:

```bash
LOW_RESOURCE_MODE=true
RUNTIME_MODE=paper
ALLOW_LIVE_ORDERS=false
EXCHANGE_PROXY_URL=
```

Notes:

- leave `EXCHANGE_PROXY_URL` empty on the server unless you really use a proxy
- if you need a non-default SQLite path, prefer `DB_PATH=/path/to/cryptoai.db`; `APP_DB_PATH` is only a compatibility alias
- low resource mode limits XGBoost threads, shrinks training dataset windows, and disables heavy scheduled jobs such as walk-forward, drift, and AB test
- in low resource mode, the dashboard also disables heavy manual actions such as training, walk-forward, backtest, drift, and AB test generation

## Docker Compose

Engine only:

```bash
docker compose -f docker-compose.tencent-lite.yml up -d --build
```

Engine plus dashboard:

```bash
docker compose -f docker-compose.tencent-lite.yml --profile dashboard up -d --build
```

The dashboard port is bound to `127.0.0.1:8501` only. Use an SSH tunnel or put it behind an authenticated reverse proxy such as Nginx before exposing it publicly.

## Resource Notes

- engine container budget: about 2.3 GB RAM / 1.5 CPU
- dashboard container budget: about 0.9 GB RAM / 0.5 CPU
- this leaves headroom for the OS and SQLite file cache
- do not expose Streamlit directly to the public internet without authentication

## Recommended First Run

```bash
python main.py health
python main.py once
python main.py backfill 90
python main.py train
```

Run `train` during low-traffic hours. On a 2c4g server, do not run repeated walk-forward and dashboard-heavy workloads at the same time.
