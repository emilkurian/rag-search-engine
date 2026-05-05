import os
import json
import re
import numpy as np
from lib.semantic_search import SemanticSearch
from lib.semantic_search import cosine_similarity

DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    # 1. Strip input
    text = text.strip()
    if not text:
        return []

    # 2. Split sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 3. Handle no-punctuation case (single "sentence")
    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
        sentences = [text]

    # 4. Clean sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # 5. Chunking with overlap
    chunks = []
    i = 0
    n_sentences = len(sentences)

    step = max_chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_chunk_size")

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]

        chunk = " ".join(chunk_sentences).strip()
        if chunk:  # 6. Only keep non-empty chunks
            chunks.append(chunk)

        i += step

    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__()  # ← remove argument
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        # same setup as base class
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        all_chunks = []
        chunk_metadata = []

        for doc_idx, doc in enumerate(documents):
            text = doc.get("description", "").strip()
            if not text:
                continue

            # semantic chunking: 4 sentences, overlap 1
            chunks = semantic_chunk(text, max_chunk_size=4, overlap=1)

            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })

        # encode all chunks
        embeddings = self.model.encode(all_chunks)

        self.chunk_embeddings = embeddings
        self.chunk_metadata = chunk_metadata

        # ensure cache dir exists
        os.makedirs("cache", exist_ok=True)

        # save embeddings
        np.save("cache/chunk_embeddings.npy", embeddings)

        # save metadata
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2
            )

        return embeddings

    def load_or_create_chunk_embeddings(self, documents):
        # same setup as base class
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        emb_path = "cache/chunk_embeddings.npy"
        meta_path = "cache/chunk_metadata.json"

        if os.path.exists(emb_path) and os.path.exists(meta_path):
            self.chunk_embeddings = np.load(emb_path)

            with open(meta_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings

        # otherwise rebuild
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("Chunk embeddings not loaded.")

        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        # score each chunk
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)

            metadata = self.chunk_metadata[i]

            chunk_scores.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": score
            })

        # aggregate best score per movie
        movie_scores = {}

        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            score = cs["score"]

            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        # sort movies by score
        sorted_movies = sorted(
            movie_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_movies = sorted_movies[:limit]

        results = []

        SCORE_PRECISION = 4

        for movie_idx, score in top_movies:
            doc = self.documents[movie_idx]

            results.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "document": doc.get("description", "")[:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": doc.get("metadata", {})
            })

        return results