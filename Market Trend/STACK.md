# Final Technology Stack Selection

- Backend: Python 3, FastAPI (served by `uvicorn`)
- Data Access: SQLAlchemy 2.x ORM
- Database (Cold Store): PostgreSQL 15 (`psycopg2-binary` driver)
- Cache (Hot Store): Redis 7
- Market Data Provider: `yfinance` with retry and rate limiting
- Logging: Python `logging` to stdout; adapter diagnostics
- Configuration: Environment-driven (`DATABASE_URL`, `REDIS_URL`, yfinance rate/retry envs)
- Containers: `docker-compose` for local development (Postgres, Redis, API)

These selections are implemented in code and infrastructure:
- `docker-compose.yml` provisions Postgres, Redis, and API service.
- `backend/requirements.txt` includes `sqlalchemy`, `redis`, and `psycopg2-binary`.
- `app/core/config.py` reads `DATABASE_URL`, `REDIS_URL`, and yfinance limits.
- `app/services/yfinance_adapter.py` implements token-bucket rate limiting and exponential backoff retries.
- `app/crud/historical.py` uses `ON CONFLICT DO NOTHING` for efficient Postgres upserts.

