from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .schema import Chunk, ScoredChunk
from .text import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lens = np.array([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        df: Counter[str] = Counter()
        for freqs in self.term_freqs:
            df.update(freqs.keys())
        n_docs = len(self.chunks)
        return {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_n: int = 50) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if not query_terms or not self.chunks:
            return []

        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_lens[idx] / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom

        top_idx = np.argsort(-scores)[:top_n]
        results: list[ScoredChunk] = []
        rank = 1
        for idx in top_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(ScoredChunk(self.chunks[int(idx)].chunk_id, score, rank))
            rank += 1
        return results
