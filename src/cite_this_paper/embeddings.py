"""Configurable dense-embedding backends and index compatibility metadata."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .corpus import CorpusError


DEFAULT_BGE_MODEL = "BAAI/bge-m3"
DEFAULT_E5_MODEL = "intfloat/e5-large-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-large"


def _collect_model_memory() -> None:
    """Best-effort release for CPU-only and CUDA-enabled installations."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class EmbeddingBackend(Protocol):
    """A model that can encode corpus passages and retrieval queries separately."""

    name: str
    fingerprint: str

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EmbeddingSpec:
    """JSON-safe, desired embedding configuration for one corpus."""

    provider: str
    model: str
    revision: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    adapter: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_DEFAULTS:
            raise CorpusError(f"Unknown embedding provider: {self.provider}")
        if not self.model.strip():
            raise CorpusError("An embedding model must be specified.")
        if self.provider == "local-file" and not self.adapter:
            raise CorpusError("The local-file embedding provider requires --adapter PATH:CLASS.")
        if self.provider != "local-file" and self.adapter:
            raise CorpusError("Only the local-file embedding provider accepts an adapter path.")
        normalized_options = dict(PROVIDER_DEFAULT_OPTIONS[self.provider])
        normalized_options.update(self.options)
        try:
            json.dumps(normalized_options, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise CorpusError("Embedding options must be JSON-serializable.") from error
        object.__setattr__(self, "options", normalized_options)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "options": dict(self.options),
        }
        if self.revision:
            result["revision"] = self.revision
        if self.adapter:
            result["adapter"] = self.adapter
        return result


PROVIDER_DEFAULTS = {
    "bge-m3": DEFAULT_BGE_MODEL,
    "e5": DEFAULT_E5_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
    "local-file": "custom",
}

PROVIDER_DEFAULT_OPTIONS = {
    "bge-m3": {"batch_size": 32, "max_length": 512},
    "e5": {"batch_size": 32},
    "openai": {"batch_size": 128},
    "local-file": {},
}


def default_embedding_config() -> dict[str, Any]:
    return {
        "provider": "bge-m3",
        "model": DEFAULT_BGE_MODEL,
        "options": {"batch_size": 32, "max_length": 512},
    }


def embedding_spec_from_config(config: Mapping[str, Any]) -> EmbeddingSpec:
    """Read structured configuration, accepting the previous BGE-only format."""
    raw = config.get("embedding")
    if raw is None:
        return EmbeddingSpec(
            "bge-m3",
            str(config.get("embedding_model", DEFAULT_BGE_MODEL)),
            options={"batch_size": 32, "max_length": 512},
        )
    if not isinstance(raw, Mapping):
        raise CorpusError("The corpus embedding configuration must be an object.")
    provider = raw.get("provider")
    if not isinstance(provider, str):
        raise CorpusError("The corpus embedding configuration needs a provider.")
    model = raw.get("model", PROVIDER_DEFAULTS.get(provider))
    if not isinstance(model, str):
        raise CorpusError("The corpus embedding configuration needs a model.")
    revision = raw.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise CorpusError("The embedding revision must be text.")
    options = raw.get("options", {})
    if not isinstance(options, Mapping):
        raise CorpusError("Embedding options must be an object.")
    adapter = raw.get("adapter")
    if adapter is not None and not isinstance(adapter, str):
        raise CorpusError("The embedding adapter reference must be text.")
    return EmbeddingSpec(provider, model, revision, dict(options), adapter)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _adapter_path(spec: EmbeddingSpec, corpus_root: Path) -> tuple[Path, str]:
    assert spec.adapter is not None
    try:
        relative_path, class_name = spec.adapter.rsplit(":", 1)
    except ValueError as error:
        raise CorpusError("Local embedding adapters must use PATH:CLASS.") from error
    if not relative_path or not class_name:
        raise CorpusError("Local embedding adapters must use PATH:CLASS.")
    root = corpus_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CorpusError("A local embedding adapter must be inside its corpus directory.") from error
    if not path.is_file():
        raise CorpusError(f"Local embedding adapter does not exist: {relative_path}")
    return path, class_name


def descriptor_for_spec(spec: EmbeddingSpec, corpus_root: Path) -> dict[str, Any]:
    """Describe all vector-affecting settings, including local adapter source."""
    descriptor: dict[str, Any] = {"spec": spec.as_dict()}
    if spec.provider == "local-file":
        path, _ = _adapter_path(spec, corpus_root)
        descriptor["adapter_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["fingerprint"] = hashlib.sha256(_canonical_json(descriptor).encode()).hexdigest()
    return descriptor


def descriptor_for_legacy_model(model: object) -> dict[str, Any]:
    """Identify explicitly injected test/application models without guessing config."""
    name = str(getattr(model, "name", type(model).__name__))
    identity = f"{type(model).__module__}.{type(model).__qualname__}"
    descriptor: dict[str, Any] = {
        "spec": {"provider": "injected", "model": name, "options": {"class": identity}},
    }
    descriptor["fingerprint"] = hashlib.sha256(_canonical_json(descriptor).encode()).hexdigest()
    return descriptor


def active_descriptor(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read new metadata or reconstruct an equivalent descriptor for BGE legacy indexes."""
    raw = state.get("embedding_config_json") or "{}"
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        data = {}
    if isinstance(data, Mapping) and isinstance(data.get("fingerprint"), str) and isinstance(data.get("spec"), Mapping):
        return dict(data)
    model = state.get("embedding_model")
    if not model:
        return None
    options = data if isinstance(data, Mapping) else {}
    legacy_options = {key: value for key, value in options.items() if value is not None}
    if not legacy_options:
        legacy_options = {"batch_size": 32, "max_length": 512}
    spec = EmbeddingSpec("bge-m3", str(model), options=legacy_options)
    # No adapter path is involved in the legacy BGE descriptor.
    descriptor: dict[str, Any] = {"spec": spec.as_dict()}
    descriptor["fingerprint"] = hashlib.sha256(_canonical_json(descriptor).encode()).hexdigest()
    return descriptor


def assert_active_descriptor(corpus_root: Path, state: Mapping[str, Any], desired: dict[str, Any]) -> None:
    active = active_descriptor(state)
    if active is None or active.get("fingerprint") != desired["fingerprint"]:
        raise CorpusError(
            "The configured embedding backend differs from the active index. "
            "Run rebuild-index before verifying claims."
        )


def active_descriptor_matches(state: Mapping[str, Any], desired: Mapping[str, Any]) -> bool:
    active = active_descriptor(state)
    return active is not None and active.get("fingerprint") == desired.get("fingerprint")


def _option_int(spec: EmbeddingSpec, name: str, default: int | None = None) -> int | None:
    value = spec.options.get(name, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusError(f"Embedding option {name} must be a positive integer.")
    return value


def validate_embedding_spec(spec: EmbeddingSpec) -> None:
    """Reject options a selected built-in backend cannot honor."""
    allowed_options = {
        "bge-m3": {"batch_size", "max_length"},
        "e5": {"batch_size", "device"},
        "openai": {"batch_size", "dimensions"},
        "local-file": None,
    }
    allowed = allowed_options[spec.provider]
    if allowed is not None:
        unknown = set(spec.options) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise CorpusError(f"Embedding provider {spec.provider} does not support: {names}.")
    for option in ("batch_size", "max_length", "dimensions"):
        if option in spec.options:
            _option_int(spec, option)
    if "device" in spec.options and not isinstance(spec.options["device"], str):
        raise CorpusError("Embedding option device must be text.")
    if spec.provider == "openai" and spec.revision:
        raise CorpusError("The openai embedding backend does not support model revisions.")


class BGEEmbeddingBackend:
    """Local BGE-M3 adapter retaining the package's existing encoding behavior."""

    def __init__(self, spec: EmbeddingSpec, descriptor: Mapping[str, Any]):
        self.spec = spec
        self.name = spec.label
        self.fingerprint = str(descriptor["fingerprint"])
        self.batch_size = _option_int(spec, "batch_size", 32) or 32
        self.max_length = _option_int(spec, "max_length", 512) or 512
        self._model = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as error:
                raise CorpusError("BGE-M3 requires FlagEmbedding. Install cite-this-paper with its default dependencies.") from error
            keyword_arguments = {"use_fp16": True}
            if self.spec.revision:
                keyword_arguments["revision"] = self.spec.revision
            self._model = BGEM3FlagModel(self.spec.model, **keyword_arguments)
        output = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(output["dense_vecs"], dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def close(self) -> None:
        had_model = self._model is not None
        self._model = None
        if had_model:
            _collect_model_memory()


class E5EmbeddingBackend:
    """Local E5 adapter with the required asymmetric retrieval prefixes."""

    def __init__(self, spec: EmbeddingSpec, descriptor: Mapping[str, Any]):
        self.spec = spec
        self.name = spec.label
        self.fingerprint = str(descriptor["fingerprint"])
        self.batch_size = _option_int(spec, "batch_size", 32) or 32
        self.device = spec.options.get("device")
        if self.device is not None and not isinstance(self.device, str):
            raise CorpusError("Embedding option device must be text.")
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise CorpusError("The e5 embedding backend requires `pip install cite-this-paper[e5]`.") from error
            keyword_arguments = {"device": self.device} if self.device else {}
            if self.spec.revision:
                keyword_arguments["revision"] = self.spec.revision
            self._model = SentenceTransformer(self.spec.model, **keyword_arguments)
        return self._model

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        return np.asarray(
            self._load().encode(
                [prefix + text for text in texts],
                batch_size=self.batch_size,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query: ")

    def close(self) -> None:
        self._model = None


class OpenAIEmbeddingBackend:
    """Hosted embedding backend; credentials stay outside corpus configuration."""

    def __init__(self, spec: EmbeddingSpec, descriptor: Mapping[str, Any]):
        self.spec = spec
        self.name = spec.label
        self.fingerprint = str(descriptor["fingerprint"])
        self.batch_size = _option_int(spec, "batch_size", 128) or 128
        self.dimensions = _option_int(spec, "dimensions")
        self._client = None

    def _load(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise CorpusError("The openai embedding backend requires OPENAI_API_KEY in the environment.")
            try:
                from openai import OpenAI
            except ImportError as error:
                raise CorpusError("The openai embedding backend requires `pip install cite-this-paper[openai]`.") from error
            self._client = OpenAI()
        return self._client

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            request: dict[str, Any] = {"model": self.spec.model, "input": list(texts[start : start + self.batch_size])}
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            response = self._load().embeddings.create(**request)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def close(self) -> None:
        self._client = None


def _load_local_adapter(spec: EmbeddingSpec, descriptor: Mapping[str, Any], corpus_root: Path) -> EmbeddingBackend:
    path, class_name = _adapter_path(spec, corpus_root)
    module_name = f"cite_this_paper_local_embedding_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise CorpusError(f"Could not load local embedding adapter: {path.name}")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
        adapter_class = getattr(module, class_name)
        backend = adapter_class(spec)
    except Exception as error:
        raise CorpusError(f"Could not initialize local embedding adapter {spec.adapter}: {error}") from error
    for attribute in ("name", "encode_passages", "encode_queries"):
        if not hasattr(backend, attribute):
            raise CorpusError(f"Local embedding adapter {spec.adapter} is missing {attribute}.")
    if not hasattr(backend, "fingerprint"):
        setattr(backend, "fingerprint", str(descriptor["fingerprint"]))
    return backend


ProviderFactory = Callable[[EmbeddingSpec, Mapping[str, Any], Path], EmbeddingBackend]


def _built_in(factory: Callable[[EmbeddingSpec, Mapping[str, Any]], EmbeddingBackend]) -> ProviderFactory:
    return lambda spec, descriptor, root: factory(spec, descriptor)


PROVIDERS: Mapping[str, ProviderFactory] = {
    "bge-m3": _built_in(BGEEmbeddingBackend),
    "e5": _built_in(E5EmbeddingBackend),
    "openai": _built_in(OpenAIEmbeddingBackend),
    "local-file": _load_local_adapter,
}


def create_embedding_backend(spec: EmbeddingSpec, corpus_root: Path) -> tuple[EmbeddingBackend, dict[str, Any]]:
    validate_embedding_spec(spec)
    descriptor = descriptor_for_spec(spec, corpus_root)
    return PROVIDERS[spec.provider](spec, descriptor, corpus_root), descriptor


def encode_passages(model: object, texts: Sequence[str]) -> np.ndarray:
    """Bridge legacy injected models while providers use the explicit contract."""
    method = getattr(model, "encode_passages", None) or getattr(model, "encode", None)
    if not callable(method):
        raise CorpusError("The embedding backend does not implement encode_passages.")
    return np.asarray(method(texts), dtype=np.float32)


def encode_queries(model: object, texts: Sequence[str]) -> np.ndarray:
    method = getattr(model, "encode_queries", None) or getattr(model, "encode", None)
    if not callable(method):
        raise CorpusError("The embedding backend does not implement encode_queries.")
    return np.asarray(method(texts), dtype=np.float32)
