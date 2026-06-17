from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_questions(path: str | Path, subset: str = "OPEN_SOURCE") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FinanceBench questions file not found: {path}. "
            "Place financebench_open_source.jsonl under financebench-main/data or pass --questions."
        )
    df = pd.read_json(path, lines=True)
    if subset and subset.upper() != "ALL" and "dataset_subset_label" in df.columns:
        df = df.loc[df["dataset_subset_label"] == subset].copy()
    required = {"financebench_id", "question", "doc_name", "evidence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Questions file is missing required columns: {sorted(missing)}")
    return df.sort_values(["doc_name", "financebench_id"]).reset_index(drop=True)


def evidence_pages(row: pd.Series) -> set[tuple[str, int]]:
    pages: set[tuple[str, int]] = set()
    for ev in row.get("evidence", []) or []:
        doc = ev.get("evidence_doc_name") or row.get("doc_name")
        page = ev.get("evidence_page_num")
        if doc is not None and page is not None:
            pages.add((str(doc), int(page)))
    if not pages and row.get("doc_name") is not None:
        pages.add((str(row["doc_name"]), -1))
    return pages


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
