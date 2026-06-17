from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .schema import Page


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(" ".join(line.split()))
    return "\n".join(cleaned).strip()


def extract_pdf_pages(pdf_path: str | Path, doc_name: str | None = None) -> list[Page]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF extraction. Install code/requirements.txt.") from exc

    pdf_path = Path(pdf_path)
    doc_name = doc_name or pdf_path.stem
    pages: list[Page] = []
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf):
            text = clean_text(page.get_text("text"))
            pages.append(Page(doc_name=doc_name, page_num=page_num, text=text))
    return pages


def load_or_extract_pages(
    doc_names: Iterable[str],
    pdf_dir: str | Path,
    cache_dir: str | Path,
    force: bool = False,
) -> list[Page]:
    pdf_dir = Path(pdf_dir)
    cache_path = Path(cache_dir) / "pages.jsonl"
    doc_names = sorted(set(map(str, doc_names)))

    if cache_path.exists() and not force:
        cached = _read_pages(cache_path)
        cached_docs = {p.doc_name for p in cached}
        if set(doc_names).issubset(cached_docs):
            return [p for p in cached if p.doc_name in set(doc_names)]

    pages: list[Page] = []
    for doc_name in tqdm(doc_names, desc="Extracting PDFs"):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF for doc_name={doc_name}: {pdf_path}")
        pages.extend(extract_pdf_pages(pdf_path, doc_name=doc_name))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page.__dict__, ensure_ascii=False) + "\n")
    return pages


def _read_pages(path: Path) -> list[Page]:
    pages: list[Page] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(Page(**json.loads(line)))
    return pages
