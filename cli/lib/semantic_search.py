from sentence_transformers import SentenceTransformer
import os
import numpy as np
import json

class SemanticSearch:
    def __init__(self):
        # load pre-trained model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Input text must not be empty or whitespace.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        # Create text representations
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]

        # Generate embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        # Ensure cache directory exists
        os.makedirs("cache", exist_ok=True)

        # Save to disk
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        path = "cache/movie_embeddings.npy"

        if os.path.exists(path):
            self.embeddings = np.load(path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        # Otherwise rebuild
        return self.build_embeddings(documents)
    
    def search(self, query, limit=5):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        results = []

        for i, doc in enumerate(self.documents):
            score = cosine_similarity(query_embedding, self.embeddings[i])

            results.append((score, doc))

        # sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        top_results = results[:limit]

        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            }
            for score, doc in top_results
        ]


def verify_model():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def verify_embeddings():
    search = SemanticSearch()

    with open("data/movies.json", "r") as f:
        data = json.load(f)
        documents = data["movies"]

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def run_search(query, limit):
    search_engine = SemanticSearch()

    with open("data/movies.json", "r") as f:
        data = json.load(f)
        documents = data["movies"] if isinstance(data, dict) else data

    search_engine.load_or_create_embeddings(documents)

    results = search_engine.search(query, limit)

    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"  {r['description'][:200]}...\n")