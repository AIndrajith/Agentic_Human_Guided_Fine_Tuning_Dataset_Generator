import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

from web_api.core.config import get_settings
from web_api.db.base import Base
import web_api.db.models  # noqa: F401  (registers every table on Base.metadata)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Schemas owned by libraries, not by our migrations (LangGraph checkpointer).
EXCLUDED_SCHEMAS = {"langgraph"}


def _database_url() -> str:
    """`alembic -x db=test ...` targets the test database; default is the main one."""
    settings = get_settings()
    use_test = context.get_x_argument(as_dictionary=True).get("db") == "test"
    url = settings.TEST_MIGRATION_DATABASE_URL if use_test else settings.MIGRATION_DATABASE_URL
    if not url:
        name = "TEST_MIGRATION_DATABASE_URL" if use_test else "MIGRATION_DATABASE_URL"
        raise RuntimeError(f"{name} is not set in .env")
    return url


def _include_name(name, type_, parent_names) -> bool:
    if type_ == "schema":
        return name not in EXCLUDED_SCHEMAS
    return True


def _configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_name=_include_name,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """`alembic upgrade head --sql`: emit SQL without connecting."""
    _configure(url=_database_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    # psycopg async can't use Windows' default ProactorEventLoop
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
