"""Create/upgrade LangGraph checkpoint tables in the `langgraph` schema. Idempotent.

Run after `alembic upgrade head`, from the repo root:
    uv run --project web_api python -m web_api.scripts.setup_checkpointer           # main DB
    uv run --project web_api python -m web_api.scripts.setup_checkpointer --test    # test DB
"""
import argparse
import asyncio

from psycopg import AsyncConnection
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from web_api.core.config import get_settings
from web_api.db.checkpointer import CHECKPOINT_SCHEMA, CONNECTION_KWARGS, to_psycopg_url


async def main(use_test: bool) -> None:
    settings = get_settings()
    url = settings.TEST_MIGRATION_DATABASE_URL if use_test else settings.MIGRATION_DATABASE_URL
    if not url:
        raise SystemExit("MIGRATION_DATABASE_URL / TEST_MIGRATION_DATABASE_URL not set in .env")

    # owner role: it has CREATE on the schema, and default privileges grant DML to synth_app
    async with await AsyncConnection.connect(to_psycopg_url(url), **CONNECTION_KWARGS) as conn:
        await AsyncPostgresSaver(conn).setup()
        cur = await conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = %s ORDER BY 1", (CHECKPOINT_SCHEMA,)
        )
        tables = [row["tablename"] for row in await cur.fetchall()]
    print(f"LangGraph checkpointer ready in schema '{CHECKPOINT_SCHEMA}': {', '.join(tables)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="target TEST_MIGRATION_DATABASE_URL")
    args = parser.parse_args()
    # psycopg async can't use Windows' default ProactorEventLoop
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(main(args.test))
