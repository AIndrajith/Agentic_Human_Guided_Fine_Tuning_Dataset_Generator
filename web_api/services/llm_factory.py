import asyncio
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai
from openai import AsyncOpenAI, AsyncAzureOpenAI
from anthropic import AsyncAnthropic
import cohere
import httpx

from web_api.data_models.enums import ModelProvider, ModelStage
from web_api.data_models.ModelConfigModels import ProjectModelConfigModel, StageModelConfig
from web_api.services.credential_service import CredentialService
from beanie import PydanticObjectId
from fastapi import HTTPException


STAGE_CAPABILITIES: dict[ModelStage, set[ModelProvider]] = {
    ModelStage.QUESTION_GENERATOR: {
        ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE,
        ModelProvider.GROQ, ModelProvider.MISTRAL, ModelProvider.COHERE,
        ModelProvider.TOGETHER, ModelProvider.OPENROUTER,
        ModelProvider.AZURE_OPENAI, ModelProvider.OLLAMA,
    },
    ModelStage.ANSWER_GENERATOR: {
        ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE,
        ModelProvider.GROQ, ModelProvider.MISTRAL, ModelProvider.COHERE,
        ModelProvider.TOGETHER, ModelProvider.OPENROUTER,
        ModelProvider.AZURE_OPENAI, ModelProvider.OLLAMA,
    },
    ModelStage.VALIDATOR: {
        ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE,
        ModelProvider.GROQ, ModelProvider.MISTRAL, ModelProvider.COHERE,
        ModelProvider.TOGETHER, ModelProvider.OPENROUTER,
        ModelProvider.AZURE_OPENAI, ModelProvider.OLLAMA,
    },
    ModelStage.META_AGENT: {
        ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE,
        ModelProvider.GROQ, ModelProvider.MISTRAL, ModelProvider.COHERE,
        ModelProvider.TOGETHER, ModelProvider.OPENROUTER,
        ModelProvider.AZURE_OPENAI, ModelProvider.OLLAMA,
    },
    ModelStage.EMBEDDER: {
        ModelProvider.OPENAI, ModelProvider.GOOGLE, ModelProvider.OLLAMA,
        ModelProvider.COHERE, ModelProvider.VOYAGEAI, ModelProvider.JINA,
    },
    ModelStage.RERANKER: {
        ModelProvider.COHERE, ModelProvider.JINA, ModelProvider.OLLAMA,
    },
}

_OPENAI_COMPAT_BASE_URLS = {
    ModelProvider.GROQ:      "https://api.groq.com/openai/v1",
    ModelProvider.TOGETHER:  "https://api.together.xyz/v1",
    ModelProvider.OPENROUTER:"https://openrouter.ai/api/v1",
    ModelProvider.MISTRAL:   "https://api.mistral.ai/v1",
}


@dataclass
class LLMClientWrapper:
    provider:   ModelProvider
    client:     Any
    model_name: str
    extra:      dict = field(default_factory=dict)


async def _build_client(provider: ModelProvider, model_name: str, stage_config: StageModelConfig = None) -> LLMClientWrapper:
    creds = await CredentialService.get_decrypted_fields(provider)

    match provider:
        case ModelProvider.OPENAI:
            client = AsyncOpenAI(api_key=creds["api_key"])
            return LLMClientWrapper(provider, client, model_name)

        case ModelProvider.ANTHROPIC:
            client = AsyncAnthropic(api_key=creds["api_key"])
            return LLMClientWrapper(provider, client, model_name)

        case ModelProvider.GOOGLE:
            genai.configure(api_key=creds["api_key"])
            return LLMClientWrapper(provider, genai, model_name)

        case ModelProvider.COHERE:
            client = cohere.AsyncClientV2(api_key=creds["api_key"])
            return LLMClientWrapper(provider, client, model_name)

        case ModelProvider.AZURE_OPENAI:
            client = AsyncAzureOpenAI(
                api_key=creds["api_key"],
                azure_endpoint=creds["endpoint"],
                api_version=creds["api_version"],
            )
            return LLMClientWrapper(provider, client, model_name, extra={"api_version": creds["api_version"]})

        case ModelProvider.OLLAMA:
            base_url = (stage_config.base_url if stage_config else None) or creds.get("base_url", "http://localhost:11434/v1")
            client = AsyncOpenAI(api_key="ollama", base_url=base_url)
            return LLMClientWrapper(provider, client, model_name)

        case ModelProvider.VOYAGEAI:
            return LLMClientWrapper(provider, None, model_name, extra={"api_key": creds["api_key"]})

        case ModelProvider.JINA:
            return LLMClientWrapper(provider, None, model_name, extra={"api_key": creds["api_key"]})

        case _:
            # Groq, Together, OpenRouter, Mistral — all OpenAI-compatible
            base_url = _OPENAI_COMPAT_BASE_URLS[provider]
            client = AsyncOpenAI(api_key=creds["api_key"], base_url=base_url)
            return LLMClientWrapper(provider, client, model_name)


async def get_llm_client(project_id: str, stage: ModelStage) -> LLMClientWrapper:
    obj_id = PydanticObjectId(project_id)
    config_doc = await ProjectModelConfigModel.find_one(
        ProjectModelConfigModel.project_id == obj_id
    )
    if not config_doc:
        raise HTTPException(status_code=404, detail="No model config set for this project")

    stage_config = config_doc.stages.get(stage.value)
    if not stage_config:
        raise HTTPException(status_code=404, detail=f"No config for stage: {stage.value}")

    return await _build_client(stage_config.provider, stage_config.model_name, stage_config)


async def call_chat(
    wrapper: LLMClientWrapper,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    match wrapper.provider:
        case (ModelProvider.OPENAI | ModelProvider.GROQ | ModelProvider.TOGETHER |
              ModelProvider.OPENROUTER | ModelProvider.MISTRAL |
              ModelProvider.AZURE_OPENAI | ModelProvider.OLLAMA):
            response = await wrapper.client.chat.completions.create(
                model=wrapper.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        case ModelProvider.ANTHROPIC:
            system = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_messages = [m for m in messages if m["role"] != "system"]
            response = await wrapper.client.messages.create(
                model=wrapper.model_name,
                system=system,
                messages=user_messages,
                max_tokens=max_tokens,
            )
            return response.content[0].text

        case ModelProvider.GOOGLE:
            system = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_content = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
            model = genai.GenerativeModel(model_name=wrapper.model_name, system_instruction=system)
            response = await model.generate_content_async(user_content)
            return response.text

        case ModelProvider.COHERE:
            cohere_messages = [
                {"role": "system" if m["role"] == "system" else "user", "content": m["content"]}
                for m in messages
            ]
            response = await wrapper.client.chat(model=wrapper.model_name, messages=cohere_messages)
            return response.message.content[0].text


async def call_embed(wrapper: LLMClientWrapper, text: str) -> list[float]:
    match wrapper.provider:
        case ModelProvider.OPENAI | ModelProvider.OLLAMA:
            response = await wrapper.client.embeddings.create(model=wrapper.model_name, input=text)
            return response.data[0].embedding

        case ModelProvider.GOOGLE:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=wrapper.model_name,
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]

        case ModelProvider.COHERE:
            response = await wrapper.client.embed(
                texts=[text],
                model=wrapper.model_name,
                input_type="search_document",
                embedding_types=["float"],
            )
            return response.embeddings.float[0]

        case ModelProvider.VOYAGEAI:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.voyageai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {wrapper.extra['api_key']}"},
                    json={"input": [text], "model": wrapper.model_name},
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]

        case ModelProvider.JINA:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {wrapper.extra['api_key']}"},
                    json={"input": [text], "model": wrapper.model_name},
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]


async def call_rerank(
    wrapper: LLMClientWrapper,
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[dict]:
    """Returns list of {index, relevance_score} sorted by score descending."""
    match wrapper.provider:
        case ModelProvider.COHERE:
            response = await wrapper.client.rerank(
                query=query,
                documents=documents,
                model=wrapper.model_name,
                top_n=top_n,
            )
            return [{"index": r.index, "relevance_score": r.relevance_score} for r in response.results]

        case ModelProvider.JINA:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.jina.ai/v1/rerank",
                    headers={"Authorization": f"Bearer {wrapper.extra['api_key']}"},
                    json={"query": query, "documents": documents, "model": wrapper.model_name, "top_n": top_n},
                )
                response.raise_for_status()
                results = response.json()["results"]
                return [{"index": r["index"], "relevance_score": r["relevance_score"]} for r in results]

        case ModelProvider.OLLAMA:
            query_emb = await call_embed(wrapper, query)
            scores = []
            for i, doc in enumerate(documents):
                doc_emb = await call_embed(wrapper, doc)
                score = _cosine_similarity(query_emb, doc_emb)
                scores.append({"index": i, "relevance_score": score})
            scores.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scores[:top_n]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def ping_stage(provider: ModelProvider, model_name: str, stage: ModelStage, stage_config=None) -> None:
    """Raises if provider is unreachable or credentials are invalid."""
    wrapper = await _build_client(provider, model_name, stage_config)

    if stage == ModelStage.EMBEDDER:
        await call_embed(wrapper, "test")
    elif stage == ModelStage.RERANKER:
        await call_rerank(wrapper, "test query", ["test document"], top_n=1)
    else:
        await call_chat(wrapper, [{"role": "user", "content": "hi"}], max_tokens=1)
