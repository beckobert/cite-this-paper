from __future__ import annotations

import os
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np

from cite_this_paper import cli
from cite_this_paper.corpus import CorpusError
from cite_this_paper.embeddings import (
    BGEEmbeddingBackend,
    E5EmbeddingBackend,
    EmbeddingSpec,
    OpenAIEmbeddingBackend,
    descriptor_for_spec,
    embedding_spec_from_config,
)
from cite_this_paper.indexing import rebuild_index
from cite_this_paper.ingest import ingest_pdf
from cite_this_paper.retrieval import verify_claim

from test_support import CorpusTestCase, FakeEmbeddingModel, FakeReranker, FakeVerifier


class EmbeddingConfigurationTests(CorpusTestCase):
    def test_legacy_bge_configuration_is_read_as_the_default_backend(self):
        self.corpus.config_path.write_text('{"embedding_model": "BAAI/bge-m3"}', encoding="utf-8")
        spec = embedding_spec_from_config(self.corpus.config())
        self.assertEqual((spec.provider, spec.model), ("bge-m3", "BAAI/bge-m3"))
        self.assertEqual(spec.options, {"batch_size": 32, "max_length": 512})

    def test_configuring_a_different_backend_requires_rebuild_and_blocks_verification(self):
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus, FakeEmbeddingModel())
        changed = self.corpus.configure_embedding(EmbeddingSpec("e5", "intfloat/e5-large-v2"))
        self.assertTrue(changed)
        self.assertEqual(self.corpus.state()["index_status"], "rebuild_required")

        with self.assertRaisesRegex(CorpusError, "differs from the active index"):
            verify_claim(
                self.corpus,
                "scientific evidence",
                reranker=FakeReranker(),
                verifier=FakeVerifier(),
                candidate_k=10,
                rerank_k=5,
                verify_k=1,
            )

    def test_configure_embedding_cli_persists_selected_provider_and_options(self):
        output = StringIO()
        with redirect_stdout(output):
            result = cli.main([
                "configure-embedding",
                "--database", str(self.corpus.root),
                "--provider", "e5",
                "--model", "intfloat/e5-large-v2",
                "--device", "cpu",
                "--batch-size", "8",
            ])
        self.assertEqual(result, 0)
        spec = self.corpus.embedding_spec()
        self.assertEqual((spec.provider, spec.model), ("e5", "intfloat/e5-large-v2"))
        self.assertEqual(spec.options, {"device": "cpu", "batch_size": 8})
        self.assertIn("Configured embedding backend", output.getvalue())

    def test_e5_backend_applies_passage_and_query_prefixes_without_loading_a_real_model(self):
        calls: list[list[str]] = []

        class SentenceTransformer:
            def __init__(self, model, **kwargs):
                self.model = model

            def encode(self, texts, **kwargs):
                calls.append(list(texts))
                return [[float(len(text)), 1.0] for text in texts]

        spec = EmbeddingSpec("e5", "intfloat/e5-large-v2")
        descriptor = descriptor_for_spec(spec, self.corpus.root)
        with patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=SentenceTransformer)}):
            backend = E5EmbeddingBackend(spec, descriptor)
            backend.encode_passages(["paper text"])
            backend.encode_queries(["claim text"])
        self.assertEqual(calls, [["passage: paper text"], ["query: claim text"]])

    def test_bge_close_releases_cuda_memory_after_unloading_a_model(self):
        spec = EmbeddingSpec("bge-m3", "BAAI/bge-m3")
        backend = BGEEmbeddingBackend(spec, descriptor_for_spec(spec, self.corpus.root))
        backend._model = object()
        cuda = types.SimpleNamespace(is_available=Mock(return_value=True), empty_cache=Mock())
        with patch("cite_this_paper.embeddings.gc.collect") as collect, patch.dict(
            sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}
        ):
            backend.close()

        self.assertIsNone(backend._model)
        collect.assert_called_once_with()
        cuda.empty_cache.assert_called_once_with()

    def test_openai_backend_batches_fake_api_requests_and_never_reads_configuration_for_credentials(self):
        requests: list[dict] = []

        class Client:
            class embeddings:
                @staticmethod
                def create(**request):
                    requests.append(request)
                    data = [types.SimpleNamespace(index=index, embedding=[float(index), 1.0]) for index, _ in enumerate(request["input"])]
                    return types.SimpleNamespace(data=data)

        spec = EmbeddingSpec("openai", "text-embedding-3-large", options={"batch_size": 2, "dimensions": 256})
        descriptor = descriptor_for_spec(spec, self.corpus.root)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch.dict(
            sys.modules, {"openai": types.SimpleNamespace(OpenAI=Client)}
        ):
            vectors = OpenAIEmbeddingBackend(spec, descriptor).encode_passages(["one", "two", "three"])
        self.assertEqual(vectors.shape, (3, 2))
        self.assertEqual([request["input"] for request in requests], [["one", "two"], ["three"]])
        self.assertTrue(all(request["dimensions"] == 256 for request in requests))

    def test_openai_backend_requires_an_environment_key(self):
        spec = EmbeddingSpec("openai", "text-embedding-3-large")
        descriptor = descriptor_for_spec(spec, self.corpus.root)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(CorpusError, "OPENAI_API_KEY"):
            OpenAIEmbeddingBackend(spec, descriptor).encode_queries(["claim"])

    def test_corpus_local_adapter_is_loaded_and_source_changes_invalidate_the_index(self):
        provider_directory = self.corpus.root / "providers"
        provider_directory.mkdir()
        adapter = provider_directory / "simple.py"
        adapter.write_text(
            "import numpy as np\n"
            "class SimpleEmbeddings:\n"
            "    name = 'simple-local'\n"
            "    def __init__(self, spec): self.spec = spec\n"
            "    def encode_passages(self, texts): return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)\n"
            "    def encode_queries(self, texts): return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)\n",
            encoding="utf-8",
        )
        spec = EmbeddingSpec("local-file", "simple", adapter="providers/simple.py:SimpleEmbeddings")
        self.assertFalse(self.corpus.configure_embedding(spec))
        ingest_pdf(self.corpus, self.pdf, on_duplicate="discard")
        rebuild_index(self.corpus)
        self.assertTrue(self.corpus.configure_embedding(EmbeddingSpec("e5", "intfloat/e5-large-v2")))
        self.assertEqual(self.corpus.state()["index_status"], "rebuild_required")
        self.assertFalse(self.corpus.configure_embedding(spec))
        self.assertEqual(self.corpus.state()["index_status"], "ready")
        _, _, results = verify_claim(
            self.corpus,
            "scientific evidence",
            reranker=FakeReranker(),
            verifier=FakeVerifier(),
            candidate_k=10,
            rerank_k=5,
            verify_k=1,
        )
        self.assertTrue(results)

        adapter.write_text(adapter.read_text(encoding="utf-8") + "\n# revised\n", encoding="utf-8")
        with self.assertRaisesRegex(CorpusError, "differs from the active index"):
            verify_claim(
                self.corpus,
                "scientific evidence",
                reranker=FakeReranker(),
                verifier=FakeVerifier(),
                candidate_k=10,
                rerank_k=5,
                verify_k=1,
            )
        self.assertEqual(self.corpus.state()["index_status"], "rebuild_required")
