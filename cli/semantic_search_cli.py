#!/usr/bin/env python3

import argparse
import re
import json
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    run_search
)
from lib.chunked_semantic_search import ChunkedSemanticSearch


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []

    step = chunk_size - overlap if overlap > 0 else chunk_size
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

    return chunks


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


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser(
        "verify", help="Verify embedding model setup"
    )
    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for input text"
    )
    embed_parser.add_argument(
        "text", type=str, help="Text to embed"
    )
    subparsers.add_parser(
        "verify_embeddings", help="Verify document embeddings"
    )
    query_parser = subparsers.add_parser(
        "embed_query", help="Generate embedding for a search query"
    )
    query_parser.add_argument(
        "query", type=str, help="Query text to embed"
    )
    search_parser = subparsers.add_parser(
        "search", help="Semantic search movies"
    )
    search_parser.add_argument(
        "query", type=str, help="Search query"
    )
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words")
    sem_parser = subparsers.add_parser("semantic_chunk", help="Split text into semantic chunks")
    sem_parser.add_argument("text", type=str, help="Text to chunk")
    sem_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    sem_parser.add_argument("--overlap", type=int, default=0, help="Overlapping sentences")
    embed_parser = subparsers.add_parser("embed_chunks", help="Generate chunk embeddings")
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5)


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            run_search(args.query, args.limit)
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                documents = data["movies"] if isinstance(data, dict) else data

            search = ChunkedSemanticSearch()
            embeddings = search.load_or_create_chunk_embeddings(documents)

            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                documents = data["movies"] if isinstance(data, dict) else data

            search = ChunkedSemanticSearch()
            search.load_or_create_chunk_embeddings(documents)

            results = search.search_chunks(args.query, args.limit)

            for i, r in enumerate(results, start=1):
                print(f"\n{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"   {r['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()