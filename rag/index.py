"""Retrieval indexes: lexical (BM25), dense vector (fastembed), and a
reciprocal-rank-fusion hybrid of the two."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from rag.models import Chunk

#: Keeps IDs like "s-009" and "sp-001" intact as single tokens: a run of
#: alphanumerics that may contain internal hyphens, but never starts or
#: ends with one.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str, stopwords: frozenset[str] | None = None) -> list[str]:
    """Tokenizes text, keeping IDs like "s-009" intact as single tokens.
    If `stopwords` is given, tokens exactly matching a stopword are dropped
    (an ID never collides with an English stopword, so IDs always survive)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


class VectorIndex(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class BM25Index:
    """Lexical retrieval over chunk text with rank_bm25's Okapi BM25."""

    def __init__(self, tokenizer=tokenize) -> None:
        self.tokenizer = tokenizer
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([self.tokenizer(c.text) for c in self.chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        assert self._bm25 is not None, "call add() before search()"
        scores = self._bm25.get_scores(self.tokenizer(query))
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class _EmbeddingModelCache:
    """Shared fastembed model loading, L2 normalization, and on-disk
    embedding cache helpers, for any index built on the same embedding
    model (dense passage vectors, multi-representation vectors, ...)."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.MODEL_NAME)
        return self._model

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.npz"

    def _load_cached(self, key: str) -> np.ndarray | None:
        if self.cache_dir is None:
            return None
        path = self._cache_path(key)
        return np.load(path)["embeddings"] if path.exists() else None

    def _save_cache(self, key: str, vecs: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._cache_path(key), embeddings=vecs)


class NumpyVectorIndex(_EmbeddingModelCache):
    """Dense retrieval using fastembed's BAAI/bge-small-en-v1.5, with exact
    cosine similarity (L2-normalized embeddings, matrix multiply). Uses the
    model's dedicated query_embed/passage_embed methods so BGE's query
    instruction prefix is applied correctly (and only to queries)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def _cache_key(self, chunks: list[Chunk]) -> str:
        h = hashlib.sha256()
        h.update(self.MODEL_NAME.encode("utf-8"))
        for c in chunks:
            h.update(c.chunk_id.encode("utf-8"))
            h.update(b"\0")
            h.update(c.text.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        key = self._cache_key(self.chunks)

        cached = self._load_cached(key)
        if cached is not None:
            self.embeddings = cached
            return

        vecs = np.array(list(self._get_model().passage_embed([c.text for c in self.chunks])))
        self.embeddings = self._normalize(vecs)
        self._save_cache(key, self.embeddings)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        assert self.embeddings is not None, "call add() before search()"
        model = self._get_model()
        qvec = self._normalize(np.array(list(model.query_embed([query])))[0])
        scores = self.embeddings @ qvec
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class HybridRetriever:
    """Reciprocal Rank Fusion over a BM25Index and a vector index."""

    def __init__(self, bm25: BM25Index, vector: VectorIndex, rrf_k: int = 60) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        pool_size = len(self.bm25.chunks)
        bm25_results = self.bm25.search(query, k=pool_size)
        vector_results = self.vector.search(query, k=pool_size)

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_by_id: dict[str, Chunk] = {}
        for results in (bm25_results, vector_results):
            for rank, (chunk, _score) in enumerate(results, start=1):
                rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)
                chunk_by_id[chunk.chunk_id] = chunk

        ranked_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])[:k]
        return [(chunk_by_id[cid], rrf_scores[cid]) for cid in ranked_ids]
