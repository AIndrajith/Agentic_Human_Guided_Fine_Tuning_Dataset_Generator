import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from web_api.data_models.enums import ModelCapability
from web_api.db.session import get_sessionmaker
from web_api.services.llm_gateway import ModelInfo

SECRET = "sk-test-super-secret"


async def _credential(client, headers, name="OpenAI team", provider="openai", secrets=None, settings=None):
    r = await client.post(
        "/credentials",
        json={"name": name, "provider": provider, "secrets": {"api_key": SECRET} if secrets is None else secrets,
              "settings": settings or {}},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _project(client, headers):
    return (await client.post("/projects", json={"title": "P", "data_type": "fiction"}, headers=headers)).json()["id"]


async def _set_stages(client, headers, pid, stages):
    return await client.put(f"/projects/{pid}/model-config", json={"stages": stages}, headers=headers)


async def test_credentials_never_expose_secrets(app_client, admin_headers):
    cred = await _credential(app_client, admin_headers)
    assert cred["secrets_present"] == ["api_key"]
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


async def test_schema_describes_forms(app_client, user_headers):
    schema = {p["provider"]: p for p in (await app_client.get("/credentials/schema", headers=user_headers)).json()}
    assert schema["openai"]["config_page"] is False and schema["openai"]["live_listing"] is True
    assert schema["ollama"]["config_page"] is True and schema["ollama"]["model_source"] == "registered"
    assert schema["ollama"]["settings_fields"] == ["base_url"] and schema["ollama"]["secret_fields"] == []
    assert schema["azure_openai"]["settings_fields"] == ["endpoint", "api_version"]


async def test_field_validation(app_client, admin_headers):
    r = await app_client.post("/credentials", headers=admin_headers, json={
        "name": "Azure", "provider": "azure_openai", "secrets": {"api_key": "k", "extra": "x"},
        "settings": {"endpoint": "https://x.openai.azure.com"},
    })
    assert r.status_code == 422 and "extra" in r.json()["error"]["message"]

    r = await app_client.post("/credentials", headers=admin_headers, json={
        "name": "Azure", "provider": "azure_openai", "secrets": {"api_key": "k"},
    })
    assert r.status_code == 422 and "endpoint" in r.json()["error"]["message"]

    r = await app_client.post("/credentials", headers=admin_headers, json={
        "name": "Ollama", "provider": "ollama", "settings": {"base_url": "localhost:11434"},
    })
    assert r.status_code == 422 and "http" in r.json()["error"]["message"]

    # defaults fill in; settings are visible, the optional key is not required
    azure = await _credential(app_client, admin_headers, name="Azure", provider="azure_openai",
                              settings={"endpoint": "https://x.openai.azure.com/"})
    assert azure["settings"] == {"endpoint": "https://x.openai.azure.com", "api_version": "2024-10-21"}
    ollama = await _credential(app_client, admin_headers, name="Ollama", provider="ollama", secrets={})
    assert ollama["settings"] == {"base_url": "http://127.0.0.1:11434"} and ollama["secrets_present"] == []


async def test_attach_to_project_and_delete_protection(app_client, admin_headers, user_headers, gateway):
    openai = await _credential(app_client, admin_headers)
    jina = await _credential(app_client, admin_headers, name="Jina", provider="jina")
    pid = await _project(app_client, user_headers)

    # a catalog model of the wrong kind is rejected before any call
    r = await _set_stages(app_client, user_headers, pid,
                          {"meta_agent": {"credential_id": openai["id"], "model_name": "text-embedding-3-large"}})
    assert r.status_code == 422 and "needs chat" in r.json()["error"]["message"]

    # a failing test call is reported and nothing is saved
    gateway.fail["gpt-nope"] = "model not found"
    r = await _set_stages(app_client, user_headers, pid, {
        "embedder": {"credential_id": openai["id"], "model_name": "text-embedding-3-large"},
        "meta_agent": {"credential_id": openai["id"], "model_name": "gpt-nope"},
    })
    assert r.status_code == 422 and r.json()["error"]["code"] == "model_test_failed"
    assert "model not found" in r.json()["error"]["message"]
    assert (await app_client.get(f"/projects/{pid}/model-config", headers=user_headers)).json()["stages"] == []

    # the embedding size is measured, not typed
    r = await _set_stages(app_client, user_headers, pid, {
        "meta_agent": {"credential_id": openai["id"], "model_name": "gpt-4o-mini"},
        "embedder": {"credential_id": openai["id"], "model_name": "text-embedding-3-large"},
    })
    assert r.status_code == 200, r.text
    stages = {s["stage"]: s for s in r.json()["stages"]}
    assert stages["embedder"]["embedding_dim"] == 3072 and stages["meta_agent"]["embedding_dim"] is None

    listing = (await app_client.get("/credentials", headers=admin_headers)).json()
    used = {c["name"]: [p["title"] for p in c["used_by_projects"]] for c in listing}
    assert used == {"OpenAI team": ["P"], "Jina": []}

    r = await app_client.delete(f"/credentials/{openai['id']}", headers=admin_headers)
    assert r.status_code == 409 and r.json()["error"]["code"] == "credential_in_use"
    assert (await app_client.delete(f"/credentials/{jina['id']}", headers=admin_headers)).status_code == 204


async def test_model_dropdown_from_catalog_and_provider(app_client, admin_headers, user_headers, gateway):
    openai = await _credential(app_client, admin_headers)

    # provider unreachable -> catalog only, with a warning
    r = (await app_client.get(f"/credentials/{openai['id']}/models?stage=embedder", headers=user_headers)).json()
    names = {m["name"]: m for m in r["models"]}
    assert r["provider_checked"] is False and r["warning"] and r["allow_custom"] is True
    assert names["text-embedding-3-large"]["embedding_dim"] == 3072
    assert "gpt-4o-mini" not in names

    # provider reachable -> only models this key can use, plus brand-new ones the catalog doesn't know
    gateway.live = {"text-embedding-3-small", "gpt-4o-mini", "brand-new-model"}
    r = (await app_client.get(f"/credentials/{openai['id']}/models?stage=embedder", headers=user_headers)).json()
    assert r["provider_checked"] is True and r["warning"] is None
    assert {m["name"] for m in r["models"]} == {"text-embedding-3-small", "brand-new-model"}
    assert next(m for m in r["models"] if m["name"] == "brand-new-model")["capabilities"] == []

    # a provider without a live list: catalog only, no warning
    voyage = await _credential(app_client, admin_headers, name="Voyage", provider="voyageai")
    r = (await app_client.get(f"/credentials/{voyage['id']}/models?stage=embedder", headers=user_headers)).json()
    assert r["provider_checked"] is False and r["warning"] is None
    assert "voyage-3" in {m["name"] for m in r["models"]}


async def test_ollama_configuration_flow(app_client, admin_headers, user_headers, gateway):
    ollama = await _credential(app_client, admin_headers, name="Office Ollama", provider="ollama", secrets={})
    cid = ollama["id"]
    gateway.ollama = [
        ModelInfo(name="llama3.1:8b", capabilities={ModelCapability.CHAT}, context_window=131072),
        ModelInfo(name="nomic-embed-text", capabilities={ModelCapability.EMBEDDING}),
    ]

    r = await app_client.post(f"/credentials/{cid}/registered-models/sync", headers=admin_headers)
    assert r.status_code == 200, r.text
    models = {m["name"]: m for m in r.json()}
    assert set(models) == {"llama3.1:8b", "nomic-embed-text"}
    assert models["llama3.1:8b"]["status"] == "untested" and models["llama3.1:8b"]["context_window"] == 131072

    # the dropdown shows only registered models that fit the stage
    r = (await app_client.get(f"/credentials/{cid}/models?stage=embedder", headers=user_headers)).json()
    assert [m["name"] for m in r["models"]] == ["nomic-embed-text"] and r["allow_custom"] is False

    # a model that isn't on the server can't be registered; an unregistered one can't be used
    r = await app_client.post(f"/credentials/{cid}/registered-models", json={"name": "mistral"}, headers=admin_headers)
    assert r.status_code == 422 and "ollama pull" in r.json()["error"]["message"]
    pid = await _project(app_client, user_headers)
    r = await _set_stages(app_client, user_headers, pid, {"embedder": {"credential_id": cid, "model_name": "mistral"}})
    assert r.status_code == 422 and "not registered" in r.json()["error"]["message"]

    # using a model tests it and records the result
    gateway.embedding_dim = 768
    r = await _set_stages(app_client, user_headers, pid,
                          {"embedder": {"credential_id": cid, "model_name": "nomic-embed-text"}})
    assert r.status_code == 200 and r.json()["stages"][0]["embedding_dim"] == 768
    listed = {m["name"]: m for m in
              (await app_client.get(f"/credentials/{cid}/registered-models", headers=admin_headers)).json()}
    assert listed["nomic-embed-text"]["status"] == "ok" and listed["nomic-embed-text"]["embedding_dim"] == 768

    # a model in use can't be removed; one removed from the server is marked failed
    r = await app_client.delete(f"/credentials/{cid}/registered-models/{listed['nomic-embed-text']['id']}",
                                headers=admin_headers)
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_in_use"
    gateway.ollama = gateway.ollama[1:]
    models = {m["name"]: m for m in
              (await app_client.post(f"/credentials/{cid}/registered-models/sync", headers=admin_headers)).json()}
    assert models["llama3.1:8b"]["status"] == "failed"
    assert (await app_client.delete(f"/credentials/{cid}/registered-models/{models['llama3.1:8b']['id']}",
                                    headers=admin_headers)).status_code == 204


async def test_azure_deployments(app_client, admin_headers, gateway):
    azure = await _credential(app_client, admin_headers, name="Azure Prod", provider="azure_openai",
                              settings={"endpoint": "https://x.openai.azure.com"})
    url = f"/credentials/{azure['id']}/registered-models"

    r = await app_client.post(url, json={"name": "company-gpt"}, headers=admin_headers)
    assert r.status_code == 422   # nothing tells us what it does

    # the base model fills in capabilities from the catalog; the deployment is tested right away
    r = await app_client.post(url, json={"name": "company-gpt4o", "base_model": "gpt-4o"}, headers=admin_headers)
    assert r.status_code == 201, r.text
    assert r.json()["capabilities"] == ["chat", "vision"] and r.json()["status"] == "ok"
    assert {(m, c.value) for _, m, c in gateway.probes} =={("company-gpt4o", "chat"), ("company-gpt4o", "vision")}

    # a failing deployment is saved as failed, with the reason
    gateway.fail["typo-deploy"] = "DeploymentNotFound"
    r = await app_client.post(url, json={"name": "typo-deploy", "capabilities": ["embedding"]}, headers=admin_headers)
    assert r.status_code == 201 and r.json()["status"] == "failed" and "DeploymentNotFound" in r.json()["last_error"]

    # catalog providers have nothing to register
    openai = await _credential(app_client, admin_headers)
    r = await app_client.post(f"/credentials/{openai['id']}/registered-models", json={"name": "x"},
                              headers=admin_headers)
    assert r.status_code == 422


async def test_app_role_cannot_create_tables(app_client):
    async with get_sessionmaker()() as session:
        with pytest.raises(ProgrammingError, match="permission denied"):
            await session.execute(text("CREATE TABLE should_fail (i int)"))
