from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    doc_name: str
    page_num: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float
    rank: int
