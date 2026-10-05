"""LangGraph Postgres checkpointer, kept in its own `langgraph` schema.

Tables are created once by `python -m web_api.scripts.setup_checkpointer` (as synth_owner);
at runtime the app connects as synth_app, which only has DML on that schema.
"""
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

CHECKPOINT_SCHEMA = "langgraph"

# what AsyncPostgresSaver expects, plus pinning its unqualified tables to our schema
CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
    "options": f"-c search_path={CHECKPOINT_SCHEMA}",
}


def to_psycopg_url(sqlalchemy_url: str) -> str:
    """`postgresql+psycopg://...` (SQLAlchemy) -> `postgresql://...` (plain psycopg)."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


async def open_checkpointer(sqlalchemy_url: str, max_size: int = 5) -> tuple[AsyncConnectionPool, AsyncPostgresSaver]:
    pool = AsyncConnectionPool(
        to_psycopg_url(sqlalchemy_url),
        kwargs=CONNECTION_KWARGS,
        max_size=max_size,
        open=False,
    )
    await pool.open()
    return pool, AsyncPostgresSaver(pool)
