from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .models import RetrievedClause
from .ingestion import extract_text


DEFAULT_COLLECTION = "fssai_regulations"


@dataclass(frozen=True)
class VectorStoreConfig:
    persist_dir: Path = Path("chroma")
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 100


_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and cache the embedding model for the current process."""
    if model_name not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Ensure the model is available locally or your network allows Hugging Face downloads."
            ) from exc
    return _MODEL_CACHE[model_name]


def _stable_id(source: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:32]


def build_index_from_folder(folder: Path, config: VectorStoreConfig | None = None) -> int:
    config = config or VectorStoreConfig()
    folder = folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    client = chromadb.PersistentClient(path=str(config.persist_dir))
    collection = client.get_or_create_collection(name=config.collection)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    model = get_embedding_model(config.embedding_model)

    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    supported = {".txt", ".md", ".pdf", ".docx"}

    for path in sorted(folder.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in supported:
            continue

        file_bytes = path.read_bytes()
        try:
            raw = extract_text(path.name, file_bytes).text
        except Exception:
            # Fall back to best-effort text read for markdown/text files.
            raw = path.read_text(encoding="utf-8", errors="replace")

        if not raw.strip():
            continue
        chunks = splitter.split_text(raw)
        rel_source = str(path.relative_to(folder))

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            ids.append(_stable_id(rel_source, i, chunk))
            docs.append(chunk)
            metadatas.append({"source": rel_source, "chunk_index": i})

    if not docs:
        return 0

    embeddings = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Upsert lets you rebuild the index without manually wiping `chroma/`.
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings.tolist())
    return len(docs)


def load_collection(config: VectorStoreConfig | None = None):
    config = config or VectorStoreConfig()
    client = chromadb.PersistentClient(path=str(config.persist_dir))
    return client.get_or_create_collection(name=config.collection)


def query_clauses(query: str, top_k: int = 4, config: VectorStoreConfig | None = None) -> list[RetrievedClause]:
    config = config or VectorStoreConfig()
    collection = load_collection(config)

    model = get_embedding_model(config.embedding_model)
    q_emb = model.encode([query], normalize_embeddings=True)

    res = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=top_k,
        # Note: Some Chroma versions do not accept "ids" in `include`.
        include=["documents", "metadatas", "distances"],
    )

    clauses: list[RetrievedClause] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0] if isinstance(res, dict) else []

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else None
        dist = dists[i] if i < len(dists) else None
        cid = ids[i] if i < len(ids) else None

        source = (meta or {}).get("source", "unknown")
        if not cid:
            chunk_index = (meta or {}).get("chunk_index")
            cid = f"{source}:{chunk_index}" if chunk_index is not None else _stable_id(source, i, doc)

        # Convert distance to a rough similarity score if possible.
        score = None
        try:
            score = float(1.0 - float(dist))
        except Exception:
            score = None

        clauses.append(RetrievedClause(text=doc, source=source, chunk_id=cid, score=score))

    return clauses


def index_exists(config: VectorStoreConfig | None = None) -> bool:
    config = config or VectorStoreConfig()
    # Chroma uses sqlite + parquet; any non-empty persist dir is a good signal.
    return config.persist_dir.exists() and any(config.persist_dir.iterdir())
