from __future__ import annotations

from collections import defaultdict

from .schema import ScoredChunk


def minmax_normalize(results: list[ScoredChunk]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {r.chunk_id: 1.0 for r in results}
    return {r.chunk_id: (r.score - lo) / (hi - lo) for r in results}


def linear_fusion(
    bm25_results: list[ScoredChunk],
    dense_results: list[ScoredChunk],
    bm25_weight: float,
    top_n: int,
) -> list[ScoredChunk]:
    bm25_scores = minmax_normalize(bm25_results)
    dense_scores = minmax_normalize(dense_results)
    ids = set(bm25_scores) | set(dense_scores)
    fused = {
        cid: bm25_weight * bm25_scores.get(cid, 0.0) + (1.0 - bm25_weight) * dense_scores.get(cid, 0.0)
        for cid in ids
    }
    return _rank(fused, top_n)


def rrf_fusion(
    bm25_results: list[ScoredChunk],
    dense_results: list[ScoredChunk],
    top_n: int,
    k: int = 60,
) -> list[ScoredChunk]:
    scores: defaultdict[str, float] = defaultdict(float)
    for result_set in (bm25_results, dense_results):
        for result in result_set:
            scores[result.chunk_id] += 1.0 / (k + result.rank)
    return _rank(dict(scores), top_n)


def _rank(scores: dict[str, float], top_n: int) -> list[ScoredChunk]:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [ScoredChunk(chunk_id=cid, score=float(score), rank=rank) for rank, (cid, score) in enumerate(items, 1)]
