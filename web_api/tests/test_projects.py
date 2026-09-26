from sqlalchemy import text

from web_api.db.session import get_sessionmaker
from web_api.tests.helpers import create_user, login

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


async def _create_project(client, headers, title="Books"):
    r = await client.post("/projects", json={"title": title, "data_type": "fiction"}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


async def test_creator_becomes_owner(app_client, user_headers):
    project = await _create_project(app_client, user_headers)
    r = await app_client.get(f"/projects/{project['id']}/members", headers=user_headers)
    assert [(m["email"], m["role"]) for m in r.json()] == [("alice@test.io", "owner")]


async def test_non_member_gets_404_and_list_is_scoped(app_client, user_headers):
    project = await _create_project(app_client, user_headers)
    await create_user("bob@test.io")
    bob = await login(app_client, "bob@test.io")

    assert (await app_client.get(f"/projects/{project['id']}", headers=bob)).status_code == 404
    assert (await app_client.get("/projects", headers=bob)).json() == []


async def test_admin_sees_every_project(app_client, user_headers, admin_headers):
    project = await _create_project(app_client, user_headers)
    assert (await app_client.get(f"/projects/{project['id']}", headers=admin_headers)).status_code == 200


async def test_member_roles(app_client, user_headers):
    project = await _create_project(app_client, user_headers)
    pid = project["id"]
    bob = await create_user("bob@test.io")
    bob_headers = await login(app_client, "bob@test.io")

    r = await app_client.post(f"/projects/{pid}/members", json={"email": "bob@test.io"}, headers=user_headers)
    assert r.status_code == 201 and r.json()["role"] == "worker"

    # workers can read but not manage
    assert (await app_client.get(f"/projects/{pid}", headers=bob_headers)).status_code == 200
    assert (await app_client.delete(f"/projects/{pid}", headers=bob_headers)).status_code == 403

    # last owner can't be removed or demoted
    alice_id = (await app_client.get("/users/me", headers=user_headers)).json()["id"]
    assert (await app_client.delete(f"/projects/{pid}/members/{alice_id}", headers=user_headers)).status_code == 409

    # after promoting bob, alice can step down
    r = await app_client.patch(f"/projects/{pid}/members/{bob.id}", json={"role": "owner"}, headers=user_headers)
    assert r.status_code == 200
    assert (await app_client.delete(f"/projects/{pid}/members/{alice_id}", headers=user_headers)).status_code == 204


async def test_upload_validates_content(app_client, user_headers, minio):
    pid = (await _create_project(app_client, user_headers))["id"]

    r = await app_client.post(
        f"/projects/{pid}/documents",
        files=[("files", ("book.pdf", PDF, "application/pdf")), ("files", ("cover.png", PNG, "image/png"))],
        headers=user_headers,
    )
    assert r.status_code == 201, r.text
    docs = r.json()
    assert [d["file_type"] for d in docs] == ["pdf", "images"]
    assert all(d["status"] == "uploaded" for d in docs)
    assert len(minio.objects) == 2
    assert all(k.startswith(f"projects/{pid}/documents/") for k in minio.objects)

    # a .pdf that isn't a PDF is rejected, and nothing from the batch is kept
    r = await app_client.post(
        f"/projects/{pid}/documents",
        files=[("files", ("ok.pdf", PDF, "application/pdf")), ("files", ("fake.pdf", b"MZ\x90", "application/pdf"))],
        headers=user_headers,
    )
    assert r.status_code == 400
    assert len(minio.objects) == 2
    assert len((await app_client.get(f"/projects/{pid}/documents", headers=user_headers)).json()) == 2

    r = await app_client.post(
        f"/projects/{pid}/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))],
        headers=user_headers,
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_file_type"


async def test_document_access_is_checked(app_client, user_headers, minio):
    pid = (await _create_project(app_client, user_headers))["id"]
    r = await app_client.post(
        f"/projects/{pid}/documents", files=[("files", ("a.pdf", PDF, "application/pdf"))], headers=user_headers
    )
    doc_id = r.json()[0]["id"]
    await create_user("bob@test.io")
    bob = await login(app_client, "bob@test.io")
    assert (await app_client.get(f"/documents/{doc_id}", headers=bob)).status_code == 404
    assert (await app_client.delete(f"/documents/{doc_id}", headers=bob)).status_code == 404

    assert (await app_client.delete(f"/documents/{doc_id}", headers=user_headers)).status_code == 204
    assert minio.objects == {}


async def test_project_delete_cascades(app_client, user_headers, minio):
    pid = (await _create_project(app_client, user_headers))["id"]
    await app_client.post(
        f"/projects/{pid}/documents", files=[("files", ("a.pdf", PDF, "application/pdf"))], headers=user_headers
    )
    assert (await app_client.delete(f"/projects/{pid}", headers=user_headers)).status_code == 204

    async with get_sessionmaker()() as session:
        counts = (await session.execute(text(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM project_members)"
        ))).one()
    assert tuple(counts) == (0, 0)
    assert minio.objects == {}


async def test_data_type_locked_once_documents_exist(app_client, user_headers, minio):
    pid = (await _create_project(app_client, user_headers))["id"]
    assert (await app_client.patch(f"/projects/{pid}", json={"data_type": "academic"}, headers=user_headers)).status_code == 200
    await app_client.post(
        f"/projects/{pid}/documents", files=[("files", ("a.pdf", PDF, "application/pdf"))], headers=user_headers
    )
    assert (await app_client.patch(f"/projects/{pid}", json={"data_type": "fiction"}, headers=user_headers)).status_code == 409
