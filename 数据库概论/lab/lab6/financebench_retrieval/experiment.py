from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .bm25 import BM25Index
from .chunking import chunk_pages
from .data import evidence_pages, load_questions, write_jsonl
from .dense import DenseIndex
from .fusion import linear_fusion, rrf_fusion
from .metrics import aggregate, hits_evidence_page, reciprocal_rank
from .pdf import load_or_extract_pages
from .schema import Chunk, ScoredChunk


def build_chunks(
    questions_path: str | Path,
    pdf_dir: str | Path,
    cache_dir: str | Path,
    subset: str = "OPEN_SOURCE",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    force_extract: bool = False,
) -> tuple[pd.DataFrame, list[Chunk]]:
    questions = load_questions(questions_path, subset=subset)
    doc_names = sorted(questions["doc_name"].astype(str).unique())
    pages = load_or_extract_pages(doc_names, pdf_dir=pdf_dir, cache_dir=cache_dir, force=force_extract)
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were produced. Check PDF extraction and chunk settings.")
    return questions, chunks


def run_experiment(
    questions_path: str | Path,
    pdf_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    subset: str = "OPEN_SOURCE",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    weights: list[float] | None = None,
    fusion: str = "linear",
    top_ks: list[int] | None = None,
    retrieval_pool: int = 50,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    dense_batch_size: int = 32,
    force_extract: bool = False,
) -> pd.DataFrame:
    weights = weights if weights is not None else [i / 10 for i in range(11)]
    top_ks = top_ks if top_ks is not None else [5, 10]
    max_k = max(max(top_ks), retrieval_pool)

    questions, chunks = build_chunks(
        questions_path=questions_path,
        pdf_dir=pdf_dir,
        cache_dir=cache_dir,
        subset=subset,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_extract=force_extract,
    )
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    bm25 = BM25Index(chunks)
    dense = DenseIndex(
        chunks,
        model_name=embedding_model,
        cache_dir=cache_dir,
        batch_size=dense_batch_size,
    )

    metric_rows: list[dict] = []
    per_query_rows: list[dict] = []

    method_rows: dict[str, list[dict]] = {"bm25": [], "dense": []}
    if fusion == "linear":
        method_rows.update({f"hybrid_linear_w{w:.2f}": [] for w in weights})
    elif fusion == "rrf":
        method_rows["hybrid_rrf"] = []
    else:
        raise ValueError("--fusion must be 'linear' or 'rrf'")

    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Evaluating queries"):
        query = str(row["question"])
        positives = evidence_pages(row)
        bm25_results = bm25.search(query, top_n=max_k)
        dense_results = dense.search(query, top_n=max_k)

        candidates: dict[str, list[ScoredChunk]] = {
            "bm25": bm25_results,
            "dense": dense_results,
        }
        if fusion == "linear":
            for w in weights:
                candidates[f"hybrid_linear_w{w:.2f}"] = linear_fusion(bm25_results, dense_results, w, top_n=max_k)
        else:
            candidates["hybrid_rrf"] = rrf_fusion(bm25_results, dense_results, top_n=max_k)

        for method, results in candidates.items():
            eval_row = {
                "financebench_id": row["financebench_id"],
                "method": method,
                "rr": reciprocal_rank(results, chunk_by_id, positives),
            }
            for k in top_ks:
                eval_row[f"hit@{k}"] = hits_evidence_page(results, chunk_by_id, positives, k)
            method_rows[method].append(eval_row)
            per_query_rows.append(
                {
                    "financebench_id": row["financebench_id"],
                    "question": query,
                    "doc_name": row["doc_name"],
                    "positive_pages": sorted(list(positives)),
                    "method": method,
                    "retrieved": [_serialize_result(r, chunk_by_id[r.chunk_id]) for r in results[: max(top_ks)]],
                    **{f"hit@{k}": eval_row[f"hit@{k}"] for k in top_ks},
                    "rr": eval_row["rr"],
                }
            )

    for method, rows in method_rows.items():
        metrics = aggregate(rows, top_ks)
        metrics["method"] = method
        if method.startswith("hybrid_linear_w"):
            metrics["bm25_weight"] = float(method.rsplit("w", 1)[1])
        elif method == "bm25":
            metrics["bm25_weight"] = 1.0
        elif method == "dense":
            metrics["bm25_weight"] = 0.0
        else:
            metrics["bm25_weight"] = None
        metric_rows.append(metrics)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows).sort_values(["method"]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    write_jsonl(output_dir / "per_query_results.jsonl", per_query_rows)
    return metrics_df


def _serialize_result(result: ScoredChunk, chunk: Chunk) -> dict:
    return {
        "rank": result.rank,
        "score": result.score,
        "chunk_id": result.chunk_id,
        "doc_name": chunk.doc_name,
        "page_num": chunk.page_num,
        "chunk_index": chunk.chunk_index,
        "text_preview": chunk.text[:300],
    }
