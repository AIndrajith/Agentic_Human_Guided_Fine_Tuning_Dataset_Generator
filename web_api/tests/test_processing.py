import uuid

from sqlalchemy import text

from web_api.db.session import get_sessionmaker
from web_api.tests.helpers import API_KEY, configure_stages, create_user, login

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


async def _project_with_docs(client, headers, *files, data_type="fiction", configured=True):
    pid = (await client.post("/projects", json={"title": "P", "data_type": data_type}, headers=headers)).json()["id"]
    r = await client.post(f"/projects/{pid}/documents", files=[("files", f) for f in files], headers=headers)
    assert r.status_code == 201, r.text
    if configured:
        await configure_stages(pid, academic=data_type == "academic")
    return pid, [d["id"] for d in r.json()]


def _complete(job_id, doc_id, pid, n_chunks=2, status="completed"):
    return {
        "task_id": job_id, "document_id": doc_id, "project_id": pid, "status": status,
        "chunks_processed": n_chunks, "total_chunks": n_chunks,
        "chunks_data": [{"chunk_index": i, "qdrant_point_id": str(uuid.uuid4()), "metadata": {"k": i}}
                        for i in range(n_chunks)],
    }


async def _doc_status(client, headers, doc_id):
    return (await client.get(f"/documents/{doc_id}", headers=headers)).json()["status"]


async def test_internal_routes_require_token(app_client, user_headers, minio):
    pid, (doc,) = await _project_with_docs(app_client, user_headers, ("a.pdf", PDF, "application/pdf"))
    assert (await app_client.get(f"/internal/files/{doc}/metadata")).status_code == 401
    r = await app_client.get(f"/internal/files/{doc}/metadata", headers={"X-Internal-Token": "wrong"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_internal_token"
    # a user JWT is not enough either
    assert (await app_client.post("/webhooks/processing-failed", json={}, headers=user_headers)).status_code == 401


async def test_full_processing_flow(app_client, user_headers, minio, celery, internal_headers):
    pid, docs = await _project_with_docs(
        app_client, user_headers, ("a.pdf", PDF, "application/pdf"), ("b.pdf", PDF, "application/pdf")
    )

    r = await app_client.post(f"/projects/{pid}/processing-jobs", headers=user_headers)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued" and {d["status"] for d in job["documents"]} == {"queued"}

    # the worker receives the same payload shape as before, with ids as strings
    (sent,) = celery.sent
    assert sent["name"] == "workers.tasks.process_documents"
    assert sent["task_id"] == job["id"]
    assert sent["payload"]["data_type"] == "fiction"
    assert {d["id"] for d in sent["payload"]["documents"]} == set(docs)

    # can't queue the same documents twice
    assert (await app_client.post(f"/projects/{pid}/processing-jobs", headers=user_headers)).status_code == 422
    r = await app_client.post(f"/projects/{pid}/processing-jobs", json={"document_ids": docs}, headers=user_headers)
    assert r.status_code == 409

    # worker: fetch metadata (-> processing) and the file
    r = await app_client.get(f"/internal/files/{docs[0]}/metadata", headers=internal_headers)
    assert r.status_code == 200 and r.json()["data_category"] == "fiction" and r.json()["should_stream"] is False
    assert await _doc_status(app_client, user_headers, docs[0]) == "processing"
    r = await app_client.get(f"/internal/files/{docs[0]}/base64", headers=internal_headers)
    assert r.status_code == 200 and r.json()["file_data"]

    # worker: store extracted text (goes to MinIO, not the DB)
    r = await app_client.post("/internal/extracted/fiction", headers=internal_headers, json={
        "document_id": docs[0], "project_id": pid, "extracted_text": "Once upon a time.",
        "extraction_metadata": {"page_count": 1},
    })
    assert r.status_code == 200, r.text
    assert f"projects/{pid}/extracted/{docs[0]}/text.txt" in minio.objects

    # worker: doc 0 done, doc 1 failed
    r = await app_client.post("/webhooks/processing-complete", json=_complete(job["id"], docs[0], pid, 3),
                              headers=internal_headers)
    assert r.status_code == 200 and r.json()["chunks_created"] == 3
    r = await app_client.get(f"/processing-jobs/{job['id']}", headers=user_headers)
    assert r.json()["status"] == "running"

    r = await app_client.post("/webhooks/processing-failed", headers=internal_headers, json={
        "task_id": job["id"], "document_id": docs[1], "error_message": "boom", "stage": "chunking",
    })
    assert r.status_code == 200

    job = (await app_client.get(f"/processing-jobs/{job['id']}", headers=user_headers)).json()
    assert job["status"] == "partial" and job["finished_at"]
    by_doc = {d["document_id"]: d for d in job["documents"]}
    assert by_doc[docs[0]]["status"] == "completed"
    assert by_doc[docs[1]]["status"] == "failed" and by_doc[docs[1]]["error"] == "[chunking] boom"

    # embedder is now locked; extracted text readable by members
    project = (await app_client.get(f"/projects/{pid}", headers=user_headers)).json()
    assert project["embedding_locked"] is True
    r = await app_client.get(f"/documents/{docs[0]}/extraction", headers=user_headers)
    assert r.status_code == 200 and r.json()["text"] == "Once upon a time."

    # re-running processes only the failed document
    r = await app_client.post(f"/projects/{pid}/processing-jobs", headers=user_headers)
    assert r.status_code == 202
    assert [d["document_id"] for d in r.json()["documents"]] == [docs[1]]


async def test_retry_replaces_chunks(app_client, user_headers, minio, celery, internal_headers):
    pid, (doc,) = await _project_with_docs(app_client, user_headers, ("a.pdf", PDF, "application/pdf"))
    job = (await app_client.post(f"/projects/{pid}/processing-jobs", headers=user_headers)).json()
    for n in (5, 2):   # worker retried the document
        r = await app_client.post("/webhooks/processing-complete", json=_complete(job["id"], doc, pid, n),
                                  headers=internal_headers)
        assert r.status_code == 200
    async with get_sessionmaker()() as session:
        count = await session.scalar(text("SELECT count(*) FROM chunks"))
    assert count == 2


async def test_academic_images_and_filename_safety(app_client, user_headers, minio, internal_headers):
    pid, (doc,) = await _project_with_docs(
        app_client, user_headers, ("paper.pdf", PDF, "application/pdf"), data_type="academic"
    )
    r = await app_client.post(
        f"/internal/extracted/academic/images/{doc}", headers=internal_headers,
        files=[("images", ("fig_1.png", PNG, "image/*"))],
    )
    assert r.status_code == 200 and r.json()["images_saved"] == 1
    assert f"projects/{pid}/extracted/{doc}/images/fig_1.png" in minio.objects

    # a path can't escape the document's folder
    r = await app_client.post(
        f"/internal/extracted/academic/images/{doc}", headers=internal_headers,
        files=[("images", ("../../other/evil.png", PNG, "image/*"))],
    )
    assert r.status_code == 200
    assert f"projects/{pid}/extracted/{doc}/images/evil.png" in minio.objects
    assert not any("other" in k for k in minio.objects)

    r = await app_client.post("/internal/extracted/academic", headers=internal_headers, json={
        "document_id": doc, "project_id": pid, "markdown_text": "# T\n![](fig_1.png)",
        "enriched_markdown": "# T\n**[IMAGE]** a chart", "extraction_metadata": {},
        "images": [{"filename": "fig_1.png", "description": "a chart", "position_in_markdown": 4}],
    })
    assert r.status_code == 200, r.text
    detail = (await app_client.get(f"/documents/{doc}/extraction", headers=user_headers)).json()
    assert detail["enriched_text"].endswith("a chart")
    assert detail["images"][0]["description"] == "a chart"


async def test_processing_access_and_validation(app_client, user_headers, minio, celery):
    pid, (pdf, png) = await _project_with_docs(
        app_client, user_headers, ("a.pdf", PDF, "application/pdf"), ("c.png", PNG, "image/png")
    )
    r = await app_client.post(f"/projects/{pid}/processing-jobs", json={"document_ids": [png]}, headers=user_headers)
    assert r.status_code == 422   # images aren't processable yet

    await create_user("bob@test.io")
    bob = await login(app_client, "bob@test.io")
    assert (await app_client.post(f"/projects/{pid}/processing-jobs", headers=bob)).status_code == 404

    job = (await app_client.post(f"/projects/{pid}/processing-jobs", json={"document_ids": [pdf]},
                                 headers=user_headers)).json()
    assert (await app_client.get(f"/processing-jobs/{job['id']}", headers=bob)).status_code == 404
    assert celery.sent[0]["payload"]["documents"] == [{"id": pdf, "file_size": len(PDF)}]


async def test_model_steps_required_before_processing(app_client, user_headers, minio, celery):
    pid, _ = await _project_with_docs(app_client, user_headers, ("a.pdf", PDF, "application/pdf"), configured=False)
    r = await app_client.post(f"/projects/{pid}/processing-jobs", headers=user_headers)
    assert r.status_code == 422 and "meta_agent, embedder" in r.json()["error"]["message"]
    assert celery.sent == []


async def test_keys_fetched_by_worker_not_sent_through_redis(app_client, user_headers, minio, celery, internal_headers):
    pid, _ = await _project_with_docs(app_client, user_headers, ("a.pdf", PDF, "application/pdf"))
    await app_client.post(f"/projects/{pid}/processing-jobs", headers=user_headers)
    payload = celery.sent[0]["payload"]
    assert "credentials" not in payload and API_KEY not in str(payload)

    url = f"/internal/projects/{pid}/processing-config"
    assert (await app_client.get(url)).status_code == 401
    config = (await app_client.get(url, headers=internal_headers)).json()
    assert config["qdrant_collection"] == f"project_{uuid.UUID(pid).hex}"
    assert config["llm"]["model"] == "openai/gpt-4o-mini" and config["llm"]["api_key"] == API_KEY
    assert config["llm"]["supports_json_schema"] is True and config["llm"]["context_window"] == 128000
    assert config["embedder"]["model"] == "openai/text-embedding-3-large" and config["embedder"]["dimension"] == 3072
    assert config["vision"] is None


async def test_deleting_removes_vectors(app_client, user_headers, minio, qdrant):
    pid, (doc_a, doc_b) = await _project_with_docs(
        app_client, user_headers, ("a.pdf", PDF, "application/pdf"), ("b.pdf", PDF, "application/pdf")
    )
    assert (await app_client.delete(f"/documents/{doc_a}", headers=user_headers)).status_code == 204
    assert qdrant.dropped_documents == [(uuid.UUID(pid), uuid.UUID(doc_a))]
    assert (await app_client.delete(f"/projects/{pid}", headers=user_headers)).status_code == 204
    assert qdrant.dropped_projects == [uuid.UUID(pid)]
