from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .schema import Chunk, ScoredChunk


class DenseIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "BAAI/bge-base-en-v1.5",
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = self._load_model(model_name)
        self.embeddings = self._load_or_encode()

    @staticmethod
    def _load_model(model_name: str):
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install code/requirements.txt."
            ) from exc
        return SentenceTransformer(model_name)

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_model = self.model_name.replace("/", "__")
        digest = hashlib.sha1("\n".join(c.chunk_id for c in self.chunks).encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"dense_embeddings_{safe_model}_{len(self.chunks)}_{digest}.npz"

    def _load_or_encode(self) -> np.ndarray:
        cache_path = self._cache_path()
        chunk_ids = np.array([c.chunk_id for c in self.chunks])
        if cache_path and cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            if np.array_equal(data["chunk_ids"], chunk_ids):
                return data["embeddings"].astype(np.float32)

        embeddings = self._encode_resumable(chunk_ids, cache_path)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, chunk_ids=chunk_ids, embeddings=embeddings)
        return embeddings

    def _encode_resumable(self, chunk_ids: np.ndarray, cache_path: Path | None) -> np.ndarray:
        texts = [chunk.text for chunk in self.chunks]
        if not cache_path:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stem = cache_path.with_suffix("")
        partial_path = Path(f"{stem}.partial.npy")
        done_path = Path(f"{stem}.done.npy")
        ids_path = Path(f"{stem}.ids.npy")

        if ids_path.exists() and not np.array_equal(np.load(ids_path, allow_pickle=False), chunk_ids):
            partial_path.unlink(missing_ok=True)
            done_path.unlink(missing_ok=True)
            ids_path.unlink(missing_ok=True)

        if not ids_path.exists():
            np.save(ids_path, chunk_ids)

        done = np.load(done_path, allow_pickle=False) if done_path.exists() else np.zeros(len(texts), dtype=bool)
        embeddings: np.memmap | None = None
        dim: int | None = None

        with tqdm(total=len(texts), initial=int(done.sum()), desc="Encoding chunks") as pbar:
            for start in range(0, len(texts), self.batch_size):
                end = min(start + self.batch_size, len(texts))
                if bool(done[start:end].all()):
                    continue

                batch_embeddings = self.model.encode(
                    texts[start:end],
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)

                if embeddings is None:
                    dim = int(batch_embeddings.shape[1])
                    if not partial_path.exists():
                        embeddings = np.lib.format.open_memmap(
                            partial_path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(len(texts), dim),
                        )
                    else:
                        embeddings = np.lib.format.open_memmap(partial_path, mode="r+")
                        dim = int(embeddings.shape[1])

                embeddings[start:end] = batch_embeddings
                embeddings.flush()
                newly_done = ~done[start:end]
                done[start:end] = True
                np.save(done_path, done)
                pbar.update(int(newly_done.sum()))

        if embeddings is None:
            embeddings = np.lib.format.open_memmap(partial_path, mode="r")
        else:
            embeddings.flush()

        completed = np.asarray(embeddings, dtype=np.float32).copy()
        partial_path.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
        ids_path.unlink(missing_ok=True)
        return completed

    def search(self, query: str, top_n: int = 50) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        q = self.model.encode(
            [self.query_prefix + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]
        scores = self.embeddings @ q
        top_idx = np.argsort(-scores)[:top_n]
        return [
            ScoredChunk(self.chunks[int(idx)].chunk_id, float(scores[idx]), rank)
            for rank, idx in enumerate(top_idx, start=1)
        ]
