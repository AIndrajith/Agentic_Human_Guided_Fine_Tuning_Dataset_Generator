from sqlalchemy import text

from web_api.db.session import get_sessionmaker
from web_api.tests.helpers import PASSWORD, create_user, login


async def test_invite_setup_login_flow(app_client, admin_headers, email):
    r = await app_client.post("/users", json={"email": "Bob@Test.io"}, headers=admin_headers)
    assert r.status_code == 201, r.text
    assert r.json()["email"] == "bob@test.io"          # stored lowercase
    assert r.json()["must_change_password"] is True
    token = email.sent["bob@test.io"]

    r = await app_client.post("/users/setup", json={"token": token, "username": "bob", "password": PASSWORD})
    assert r.status_code == 200, r.text
    assert r.json()["must_change_password"] is False

    # the link is one-time
    r = await app_client.post("/users/setup", json={"token": token, "username": "bob2", "password": PASSWORD})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_token"

    headers = await login(app_client, "BOB@test.io")    # email is case-insensitive
    r = await app_client.get("/users/me", headers=headers)
    assert r.json()["username"] == "bob"


async def test_setup_token_is_stored_hashed(app_client, admin_headers, email):
    await app_client.post("/users", json={"email": "carol@test.io"}, headers=admin_headers)
    token = email.sent["carol@test.io"]
    async with get_sessionmaker()() as session:
        stored = await session.scalar(text("SELECT setup_token_hash FROM users WHERE email='carol@test.io'"))
    assert stored and stored != token and len(stored) == 64


async def test_weak_password_is_422(app_client, admin_headers, email):
    await app_client.post("/users", json={"email": "dave@test.io"}, headers=admin_headers)
    r = await app_client.post(
        "/users/setup", json={"token": email.sent["dave@test.io"], "username": "dave", "password": "weakpass"}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


async def test_duplicate_email_is_409(app_client, admin_headers, email):
    await app_client.post("/users", json={"email": "erin@test.io"}, headers=admin_headers)
    r = await app_client.post("/users", json={"email": "erin@test.io"}, headers=admin_headers)
    assert r.status_code == 409


async def test_resend_invite_invalidates_old_link(app_client, admin_headers, email):
    await app_client.post("/users", json={"email": "fay@test.io"}, headers=admin_headers)
    old = email.sent["fay@test.io"]
    r = await app_client.post("/users/resend-invite", json={"email": "fay@test.io"}, headers=admin_headers)
    assert r.status_code == 200
    new = email.sent["fay@test.io"]
    assert new != old
    r = await app_client.post("/users/setup", json={"token": old, "username": "fay", "password": PASSWORD})
    assert r.status_code == 401
    r = await app_client.post("/users/setup", json={"token": new, "username": "fay", "password": PASSWORD})
    assert r.status_code == 200


async def test_non_admin_cannot_invite(app_client, user_headers, email):
    r = await app_client.post("/users", json={"email": "x@test.io"}, headers=user_headers)
    assert r.status_code == 403


async def test_wrong_password_and_unknown_email_look_the_same(app_client, user_headers):
    wrong = await app_client.post("/users/login", data={"username": "alice@test.io", "password": "Nope!123a"})
    unknown = await app_client.post("/users/login", data={"username": "ghost@test.io", "password": "Nope!123a"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


async def test_deactivated_user_token_stops_working(app_client, admin_headers):
    user = await create_user("gus@test.io")
    headers = await login(app_client, "gus@test.io")
    assert (await app_client.get("/users/me", headers=headers)).status_code == 200

    r = await app_client.patch(f"/users/{user.id}/active", json={"is_active": False}, headers=admin_headers)
    assert r.status_code == 200
    r = await app_client.get("/users/me", headers=headers)
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "account_not_active"


async def test_missing_or_bad_token(app_client):
    assert (await app_client.get("/users/me")).status_code == 401
    r = await app_client.get("/users/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401
