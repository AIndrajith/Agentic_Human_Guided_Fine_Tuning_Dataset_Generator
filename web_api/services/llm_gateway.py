"""The one door to every model provider, built on LiteLLM.

- catalog:  LiteLLM's model list (type, context size, vision, embedding size). Updated by LiteLLM
            at startup unless LITELLM_LOCAL_MODEL_COST_MAP=True.
- live:     ask the provider which models a key can use (see providers.LiveListing).
- ollama:   read the models downloaded on an Ollama server, with their capabilities.
- probe:    a tiny real call that proves a key + model work for one capability
            (and measures the embedding size).
"""
import asyncio
import base64
import hashlib
import logging
import struct
import time
import zlib
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import httpx
import litellm

from web_api.data_models.enums import ModelCapability, ModelProvider
from web_api.errors import ModelTestFailed
from web_api.services.providers import FIELD_TO_LITELLM_ARG, LiveListing, ProviderSpec, get_spec

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.drop_params = True   # silently drop params a provider doesn't support (e.g. temperature)

LIVE_LIST_TTL_SECONDS = 600
PROBE_TIMEOUT_SECONDS = 120   # an Ollama model may need to load into memory first
HTTP_TIMEOUT_SECONDS = 15


@dataclass
class ModelInfo:
    name: str
    capabilities: set[ModelCapability] = field(default_factory=set)
    context_window: int | None = None
    embedding_dim: int | None = None
    supports_json_schema: bool = False


@dataclass
class ProbeResult:
    embedding_dim: int | None = None


@dataclass
class Connection:
    """A credential's decrypted secrets + plain settings, ready for LiteLLM calls."""
    provider: ModelProvider
    secrets: dict[str, str]
    settings: dict[str, str]

    @property
    def spec(self) -> ProviderSpec:
        return get_spec(self.provider)

    def litellm_kwargs(self) -> dict[str, str]:
        kwargs = {}
        for name, value in {**self.settings, **self.secrets}.items():
            if value and (arg := FIELD_TO_LITELLM_ARG.get(name)):
                kwargs[arg] = value
        return kwargs

    def model_id(self, model_name: str, capability: ModelCapability) -> str:
        """LiteLLM model string, e.g. 'anthropic/claude-sonnet-4-5'."""
        spec = self.spec
        prefix = spec.litellm_provider
        if spec.chat_litellm_provider and capability in (ModelCapability.CHAT, ModelCapability.VISION):
            prefix = spec.chat_litellm_provider
        return f"{prefix}/{model_name}"


# ---------- catalog ----------

_MODE_TO_CAPABILITY = {
    "chat": ModelCapability.CHAT,
    "embedding": ModelCapability.EMBEDDING,
    "rerank": ModelCapability.RERANK,
}


def _strip_prefix(name: str, spec: ProviderSpec) -> str:
    prefixes = [*spec.catalog_keys, spec.litellm_provider, spec.chat_litellm_provider, "models"]
    for prefix in filter(None, prefixes):
        if name.startswith(f"{prefix}/"):
            return name[len(prefix) + 1:]
    return name


def _is_deprecated(entry: dict) -> bool:
    try:
        return date.fromisoformat(entry["deprecation_date"]) <= date.today()
    except (KeyError, TypeError, ValueError):
        return False


def _catalog_info(name: str, entry: dict) -> ModelInfo:
    """Models of other kinds (image generation, speech, ...) get no capabilities: known, but never offered."""
    capability = _MODE_TO_CAPABILITY.get(entry.get("mode"))
    capabilities = {capability} if capability else set()
    if capability == ModelCapability.CHAT and entry.get("supports_vision"):
        capabilities.add(ModelCapability.VISION)
    return ModelInfo(
        name=name,
        capabilities=capabilities,
        context_window=entry.get("max_input_tokens"),
        embedding_dim=entry.get("output_vector_size"),
        supports_json_schema=bool(entry.get("supports_response_schema")),
    )


@lru_cache
def _catalog_index(provider: ModelProvider) -> dict[str, ModelInfo]:
    spec = get_spec(provider)
    index: dict[str, ModelInfo] = {}
    for key, entry in litellm.model_cost.items():
        if not isinstance(entry, dict) or entry.get("litellm_provider") not in spec.catalog_keys:
            continue
        if _is_deprecated(entry):
            continue
        info = _catalog_info(_strip_prefix(key, spec), entry)
        if info.name not in index:
            index[info.name] = info
    return index


def catalog_models(provider: ModelProvider) -> list[ModelInfo]:
    """Chat / embedding / rerank models the catalog knows for this provider."""
    return sorted((m for m in _catalog_index(provider).values() if m.capabilities), key=lambda m: m.name)


def catalog_lookup(provider: ModelProvider, model_name: str) -> ModelInfo | None:
    """Any catalog entry, including kinds we never offer (empty capabilities)."""
    return _catalog_index(provider).get(model_name)


# ---------- live listing ----------

_live_cache: dict[tuple[ModelProvider, str], tuple[float, set[str]]] = {}


def _cache_key(connection: Connection) -> tuple[ModelProvider, str]:
    secret = connection.secrets.get("api_key", "")
    return connection.provider, hashlib.sha256(secret.encode()).hexdigest()[:16]


async def _fetch_live_names(connection: Connection) -> list[str]:
    spec = connection.spec
    api_key = connection.secrets.get("api_key")
    if spec.live_listing == LiveListing.LITELLM:
        # blocking HTTP inside; LiteLLM returns [] instead of raising on errors
        names = await asyncio.to_thread(
            litellm.get_valid_models,
            check_provider_endpoint=True,
            custom_llm_provider=spec.litellm_provider,
            api_key=api_key,
        )
        if not names:
            raise RuntimeError("provider returned no models")
        return names
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{spec.api_base}/models", headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        body = response.json()
    items = body.get("data", []) if isinstance(body, dict) else body
    return [item["id"] for item in items if isinstance(item, dict) and item.get("id")]


async def live_model_names(connection: Connection, use_cache: bool = True) -> set[str] | None:
    """Model names this key can use, or None when the provider has no listing / couldn't be reached."""
    if connection.spec.live_listing is None:
        return None
    key = _cache_key(connection)
    if use_cache and (cached := _live_cache.get(key)) and time.monotonic() - cached[0] < LIVE_LIST_TTL_SECONDS:
        return cached[1]
    try:
        names = {_strip_prefix(n, connection.spec) for n in await _fetch_live_names(connection)}
    except Exception as e:
        logger.warning("Live model listing failed for %s: %s", connection.provider.value, e)
        return None
    _live_cache[key] = (time.monotonic(), names)
    return names


# ---------- Ollama ----------

_OLLAMA_CAPABILITIES = {
    "completion": ModelCapability.CHAT,
    "vision": ModelCapability.VISION,
    "embedding": ModelCapability.EMBEDDING,
}


def _ollama_request(connection: Connection) -> tuple[str, dict[str, str]]:
    base = connection.settings["base_url"].rstrip("/")
    api_key = connection.secrets.get("api_key")
    return base, ({"Authorization": f"Bearer {api_key}"} if api_key else {})


async def ollama_models(connection: Connection) -> list[ModelInfo]:
    """Models downloaded on the Ollama server. Capabilities are empty on old Ollama versions."""
    base, headers = _ollama_request(connection)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{base}/api/tags", headers=headers)
        response.raise_for_status()
        names = [m.get("name") or m.get("model") for m in response.json().get("models", [])]

        models = []
        for name in filter(None, names):
            info = ModelInfo(name=name)
            try:
                show = await client.post(f"{base}/api/show", json={"model": name}, headers=headers)
                show.raise_for_status()
                data = show.json()
                info.capabilities = {
                    _OLLAMA_CAPABILITIES[c] for c in data.get("capabilities", []) if c in _OLLAMA_CAPABILITIES
                }
                info.context_window = next(
                    (v for k, v in (data.get("model_info") or {}).items() if k.endswith(".context_length")), None
                )
            except httpx.HTTPError as e:
                logger.warning("Ollama /api/show failed for %s: %s", name, e)
            models.append(info)
    return models


# ---------- connection test ----------

async def test_connection(connection: Connection) -> tuple[bool | None, str]:
    """(ok, message). ok=None means the provider can't be tested without a model."""
    try:
        if connection.provider == ModelProvider.OLLAMA:
            base, headers = _ollama_request(connection)
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{base}/api/version", headers=headers)
                response.raise_for_status()
            return True, f"Connected, Ollama {response.json().get('version', '')}".strip()

        if connection.provider == ModelProvider.AZURE_OPENAI:
            endpoint = connection.settings["endpoint"].rstrip("/")
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    f"{endpoint}/openai/models",
                    params={"api-version": connection.settings["api_version"]},
                    headers={"api-key": connection.secrets.get("api_key", "")},
                )
                response.raise_for_status()
            return True, "Connected to Azure OpenAI"
    except httpx.HTTPStatusError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except httpx.HTTPError as e:
        return False, f"Could not connect: {e}"

    if connection.spec.live_listing:
        names = await live_model_names(connection, use_cache=False)
        if names is None:
            return False, "Could not list models with this key. Check the key."
        return True, f"Connected, {len(names)} models available"
    return None, "This provider can't be tested without a model. Each model is tested when you save it."


# ---------- probe (tiny real call) ----------

def _tiny_png(size: int = 16) -> str:
    """A small white PNG as a data URI (for vision probes)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff\xff\xff" * size for _ in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _clean_error(error: Exception) -> str:
    message = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    parts: list[str] = []
    for part in message.removeprefix("litellm.").split(": "):   # "X: X: detail" -> "X: detail"
        if not parts or parts[-1] != part.removeprefix("litellm."):
            parts.append(part.removeprefix("litellm."))
    return ": ".join(parts)[:300]


def _embedding_length(item) -> int:
    vector = item["embedding"] if isinstance(item, dict) else item.embedding
    return len(vector)


async def probe(connection: Connection, model_name: str, capability: ModelCapability) -> ProbeResult:
    """Make one tiny call. Raises ModelTestFailed with the provider's message on failure."""
    model = connection.model_id(model_name, capability)
    kwargs = {**connection.litellm_kwargs(), "timeout": PROBE_TIMEOUT_SECONDS}
    try:
        match capability:
            case ModelCapability.CHAT:
                await litellm.acompletion(
                    model=model, messages=[{"role": "user", "content": "Reply with OK."}], max_tokens=16, **kwargs
                )
            case ModelCapability.VISION:
                await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "What colour is this image? One word."},
                        {"type": "image_url", "image_url": {"url": _tiny_png()}},
                    ]}],
                    max_tokens=16,
                    **kwargs,
                )
            case ModelCapability.EMBEDDING:
                response = await litellm.aembedding(model=model, input=["hello"], **kwargs)
                return ProbeResult(embedding_dim=_embedding_length(response.data[0]))
            case ModelCapability.RERANK:
                await litellm.arerank(model=model, query="test", documents=["alpha", "beta"], top_n=1, **kwargs)
    except Exception as e:
        raise ModelTestFailed(f"{model_name} ({capability.value}): {_clean_error(e)}") from e
    return ProbeResult()
