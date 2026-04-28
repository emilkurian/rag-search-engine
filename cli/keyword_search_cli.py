import argparse
import json
import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import Counter
import math

stemmer = PorterStemmer()


translator = str.maketrans("", "", string.punctuation)

with open("data/stopwords.txt", "r") as stop_words:
    stop_word_list = stop_words.read().splitlines()


class InvertedIndex:

    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()

    def __add_document(self, doc_id, text):
        """
        Tokenize text and add tokens to the index + term frequencies
        """
        # normalize text
        text = text.lower().translate(translator)
        tokens = text.split()

        # init counter for this doc
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token in stop_word_list:
                continue

            stemmed = stemmer.stem(token)

            # update inverted index
            if stemmed not in self.index:
                self.index[stemmed] = set()
            self.index[stemmed].add(doc_id)

            # update term frequency
            self.term_frequencies[doc_id][stemmed] += 1

    def get_documents(self, term):
        """
        Return sorted list of document IDs for a term
        """
        # apply same preprocessing as indexing
        term = term.lower().translate(translator)

        if term in stop_word_list:
            return []

        stemmed = stemmer.stem(term)

        if stemmed not in self.index:
            return []

        return sorted(self.index[stemmed])

    def build(self, movies):
        """
        Build index from all movie documents
        """
        for m in movies:
            doc_id = m["id"]

            # combine title + description
            text = f"{m['title']} {m['description']}"

            # store full movie object (overwrites text stored earlier, which is fine)
            self.docmap[doc_id] = m

            # add to index
            self.__add_document(doc_id, text)

    def save(self):
        """
        Save index and docmap to disk using pickle
        """
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        """
        Load index and docmap from disk using pickle
        """
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        tf_path = "cache/term_frequencies.pkl"

        # ensure both files exist
        if not os.path.exists(index_path) or not os.path.exists(docmap_path) or not os.path.exists(tf_path):
            raise FileNotFoundError("Index files not found. Please run the 'build' command first.")

        # load index
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        # load docmap
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
    
    def get_tf(self, doc_id, term):
        """
        Return term frequency for a term in a document
        """
        # normalize
        term = term.lower().translate(translator)
        tokens = term.split()

        if len(tokens) != 1:
            raise ValueError("Only single terms are allowed")

        token = tokens[0]

        if token in stop_word_list:
            return 0

        stemmed = stemmer.stem(token)

        if doc_id not in self.term_frequencies:
            return 0

        return self.term_frequencies[doc_id].get(stemmed, 0)
         


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
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()