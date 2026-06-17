from __future__ import annotations

from .schema import Chunk, ScoredChunk


def hits_evidence_page(
    results: list[ScoredChunk],
    chunk_by_id: dict[str, Chunk],
    positive_pages: set[tuple[str, int]],
    k: int,
) -> bool:
    for result in results[:k]:
        chunk = chunk_by_id[result.chunk_id]
        if (chunk.doc_name, chunk.page_num) in positive_pages:
            return True
        if (chunk.doc_name, -1) in positive_pages:
            return True
    return False


def reciprocal_rank(
    results: list[ScoredChunk],
    chunk_by_id: dict[str, Chunk],
    positive_pages: set[tuple[str, int]],
) -> float:
    for result in results:
        chunk = chunk_by_id[result.chunk_id]
        if (chunk.doc_name, chunk.page_num) in positive_pages or (chunk.doc_name, -1) in positive_pages:
            return 1.0 / result.rank
    return 0.0


def aggregate(rows: list[dict], top_ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {"num_queries": float(len(rows))}
    if not rows:
        for k in top_ks:
            out[f"recall@{k}"] = 0.0
        out["mrr"] = 0.0
        return out
    for k in top_ks:
        out[f"recall@{k}"] = sum(float(row[f"hit@{k}"]) for row in rows) / len(rows)
    out["mrr"] = sum(float(row["rr"]) for row in rows) / len(rows)
    return out
