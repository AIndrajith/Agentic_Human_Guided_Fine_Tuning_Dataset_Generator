"""Create the first admin (or promote/reset an existing user to admin).

From the repo root:
    uv run --project web_api python -m web_api.scripts.create_admin --email you@example.com --username admin
Prompts for the password (not echoed, not stored in shell history).
"""
import argparse
import asyncio
import getpass

from web_api.data_models.enums import AppRole
from web_api.db.models import User
from web_api.db.session import dispose_engine, get_sessionmaker
from web_api.errors import ValidationError
from web_api.services.SecurityService import SecurityService
from web_api.services.UserService import UserService


def _ask_password(security: SecurityService) -> str:
    while True:
        password = getpass.getpass("Password: ")
        try:
            security.validate_password(password)
        except ValidationError as e:
            print(e.message)
            continue
        if getpass.getpass("Repeat password: ") == password:
            return password
        print("Passwords do not match.")


async def main(email: str, username: str) -> None:
    security = SecurityService()
    password = _ask_password(security)

    async with get_sessionmaker()() as session:
        users = UserService(session)
        taken = await users.find_by_username(username)
        user = await users.find_by_email(email)
        if taken and (not user or taken.id != user.id):
            raise SystemExit(f"Username '{username}' is already taken")

        if user is None:
            user = User(email=users.normalize_email(email))
            session.add(user)
        user.username = username
        user.password_hash = security.hash_password(password)
        user.app_role = AppRole.ADMIN
        user.is_active = True
        user.must_change_password = False
        user.setup_token_hash = None
        user.setup_token_expires_at = None
        await session.commit()
        print(f"Admin ready: {user.email} (id {user.id})")
    await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(main(args.email, args.username))
