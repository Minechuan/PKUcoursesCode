from __future__ import annotations

from .schema import Chunk, Page


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chars: int = 80,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        text = page.text.strip()
        if len(text) < min_chars:
            continue
        starts = list(range(0, len(text), chunk_size - chunk_overlap))
        for chunk_index, start in enumerate(starts):
            piece = text[start : start + chunk_size].strip()
            if len(piece) < min_chars:
                continue
            chunk_id = f"{page.doc_name}::p{page.page_num}::c{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_name=page.doc_name,
                    page_num=page.page_num,
                    chunk_index=chunk_index,
                    text=piece,
                )
            )
            if start + chunk_size >= len(text):
                break
    return chunks
