import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from web_api.db.session import get_sessionmaker

SECRET = "sk-test-super-secret"


async def _credential(client, headers, name="OpenAI team", provider="openai", fields=None):
    r = await client.post(
        "/credentials",
        json={"name": name, "provider": provider, "fields": fields or {"api_key": SECRET}},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


async def test_credentials_never_expose_secrets(app_client, admin_headers):
    cred = await _credential(app_client, admin_headers)
    assert cred["fields_present"] == ["api_key"]
    listing = await app_client.get("/credentials", headers=admin_headers)
    assert SECRET not in listing.text

    async with get_sessionmaker()() as session:
        stored = await session.scalar(text("SELECT encrypted_fields::text FROM provider_credentials"))
    assert SECRET not in stored


async def test_credentials_are_admin_only_but_options_are_public(app_client, admin_headers, user_headers):
    await _credential(app_client, admin_headers)
    assert (await app_client.get("/credentials", headers=user_headers)).status_code == 403
    options = await app_client.get("/credentials/options", headers=user_headers)
    assert options.status_code == 200
    assert set(options.json()[0]) == {"id", "name", "provider"}


async def test_field_validation(app_client, admin_headers):
    r = await app_client.post(
        "/credentials",
        json={"name": "Azure", "provider": "azure_openai", "fields": {"api_key": "k", "extra": "x"}},
        headers=admin_headers,
    )
    assert r.status_code == 422
    assert "endpoint" in r.json()["error"]["message"] and "extra" in r.json()["error"]["message"]


async def test_attach_to_project_and_delete_protection(app_client, admin_headers, user_headers):
    openai = await _credential(app_client, admin_headers)
    jina = await _credential(app_client, admin_headers, name="Jina", provider="jina")
    pid = (await app_client.post(
        "/projects", json={"title": "P", "data_type": "fiction"}, headers=user_headers
    )).json()["id"]

    # provider must be able to serve the stage
    r = await app_client.put(
        f"/projects/{pid}/model-config",
        json={"stages": {"meta_agent": {"credential_id": jina["id"], "model_name": "x"}}},
        headers=user_headers,
    )
    assert r.status_code == 422

    # embedder needs a dimension
    r = await app_client.put(
        f"/projects/{pid}/model-config",
        json={"stages": {"embedder": {"credential_id": openai["id"], "model_name": "text-embedding-3-large"}}},
        headers=user_headers,
    )
    assert r.status_code == 422

    r = await app_client.put(
        f"/projects/{pid}/model-config",
        json={"stages": {
            "meta_agent": {"credential_id": openai["id"], "model_name": "gpt-4o-mini"},
            "embedder": {"credential_id": openai["id"], "model_name": "text-embedding-3-large", "embedding_dim": 3072},
        }},
        headers=user_headers,
    )
    assert r.status_code == 200, r.text
    assert {s["stage"] for s in r.json()["stages"]} == {"meta_agent", "embedder"}

    listing = (await app_client.get("/credentials", headers=admin_headers)).json()
    used = {c["name"]: [p["title"] for p in c["used_by_projects"]] for c in listing}
    assert used == {"OpenAI team": ["P"], "Jina": []}

    r = await app_client.delete(f"/credentials/{openai['id']}", headers=admin_headers)
    assert r.status_code == 409 and r.json()["error"]["code"] == "credential_in_use"
    assert (await app_client.delete(f"/credentials/{jina['id']}", headers=admin_headers)).status_code == 204


async def test_app_role_cannot_create_tables(app_client):
    async with get_sessionmaker()() as session:
        with pytest.raises(ProgrammingError, match="permission denied"):
            await session.execute(text("CREATE TABLE should_fail (i int)"))
