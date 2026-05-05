import argparse
import json
import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import Counter
import math
from cli.lib.semantic_search import verify_model
from cli.lib.keyword_search import InvertedIndex


stemmer = PorterStemmer()
BM25_K1 = 1.5
BM25_B = 0.75

translator = str.maketrans("", "", string.punctuation)

with open("data/stopwords.txt", "r") as stop_words:
    stop_word_list = stop_words.read().splitlines()


def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()

    index.load()

    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)
         


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check")
    search_parser.add_argument("query", type=str, help="Search query")
    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term to check")
    tfidf_parser = subparsers.add_parser("tfidf", help="Compute TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int)
    bm25_tf_parser.add_argument("term", type=str)
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1)
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B)
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    verify_parser = subparsers.add_parser(
        "verify", help="Verify embedding model setup"
    )

    args = parser.parse_args()

    

    # movie_query_list = [title for title in movie_data_dict.keys() if args.query in title.lower()]

    match args.command:
        case "search":
            index = InvertedIndex()

            # try loading index
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            print(f"Searching for: {args.query}")

            # preprocess query (same pipeline as indexing)
            query_tokens = args.query.lower().translate(translator).split()

            results = []
            seen = set()

            for token in query_tokens:
                if token in stop_word_list:
                    continue

                stemmed = stemmer.stem(token)

                doc_ids = index.get_documents(stemmed)

                for doc_id in doc_ids:
                    if doc_id not in seen:
                        seen.add(doc_id)
                        results.append(doc_id)

                    if len(results) == 5:
                        break

                if len(results) == 5:
                    break

            # print results
            for i, doc_id in enumerate(results, 1):
                movie = index.docmap[doc_id]
                print(f"{i}. {movie['title']} (ID: {doc_id})")

        case "build":
            # load movie data
            with open("data/movies.json", "r") as movie_data:
                movie_data_dict = json.load(movie_data)

            movies = movie_data_dict["movies"]

            # build index
            index = InvertedIndex()
            index.build(movies)
            index.save()

        case "tf":
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            tf = index.get_tf(args.doc_id, args.term)
            print(tf)

        case "idf":
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            term = args.term.lower().translate(translator)

            if term in stop_word_list:
                print("0.00")
                return

            stemmed = stemmer.stem(term)

            N = len(index.docmap)
            df = len(index.index.get(stemmed, []))

            # smoothed IDF
            idf = math.log((N + 1) / (df + 1))

            print(f"{idf:.2f}")

        case "tfidf":
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            # normalize term
            term = args.term.lower().translate(translator)

            if term in stop_word_list:
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': 0.00")
                return

            stemmed = stemmer.stem(term)

            # --- TF ---
            tf = index.get_tf(args.doc_id, term)

            # --- IDF ---
            N = len(index.docmap)
            df = len(index.index.get(stemmed, []))
            idf = math.log((N + 1) / (df + 1))

            # --- TF-IDF ---
            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except FileNotFoundError as e:
                print(e)

        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except FileNotFoundError as e:
                print(e)

        case "bm25search":
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            results = index.bm25_search(args.query, args.limit)

            for i, (doc_id, score) in enumerate(results, 1):
                movie = index.docmap[doc_id]
                print(f"{i}. ({doc_id}) {movie['title']} - Score: {score:.2f}")

        case "verify":
            verify_model()
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()