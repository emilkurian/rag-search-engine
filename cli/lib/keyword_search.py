import argparse
import json
import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import Counter
import math


stemmer = PorterStemmer()
BM25_K1 = 1.5
BM25_B = 0.75

translator = str.maketrans("", "", string.punctuation)

with open("data/stopwords.txt", "r") as stop_words:
    stop_word_list = stop_words.read().splitlines()

class InvertedIndex:

    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        # normalize text
        text = text.lower().translate(translator)
        tokens = text.split()

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        filtered_tokens = []

        for token in tokens:
            if token in stop_word_list:
                continue

            stemmed = stemmer.stem(token)
            filtered_tokens.append(stemmed)

            # update inverted index
            if stemmed not in self.index:
                self.index[stemmed] = set()
            self.index[stemmed].add(doc_id)

            # update term frequency
            self.term_frequencies[doc_id][stemmed] += 1

        # ✅ FIX: length based on filtered + stemmed tokens
        self.doc_lengths[doc_id] = len(filtered_tokens)

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

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        """
        Load index and docmap from disk using pickle
        """
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        tf_path = "cache/term_frequencies.pkl"
        dl_path = self.doc_lengths_path

        # ensure both files exist
        if not os.path.exists(index_path) or not os.path.exists(docmap_path) or not os.path.exists(tf_path) or not os.path.exists(dl_path):
            raise FileNotFoundError("Index files not found. Please run the 'build' command first.")

        # load index
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        # load docmap
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(dl_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
    
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
    
    def get_bm25_idf(self, term: str) -> float:
        """
        Calculate BM25 IDF score for a term
        """
        # normalize
        term = term.lower().translate(translator)
        tokens = term.split()

        if len(tokens) != 1:
            raise ValueError("Only single terms are allowed")

        token = tokens[0]

        if token in stop_word_list:
            return 0.0

        stemmed = stemmer.stem(token)

        N = len(self.docmap)
        df = len(self.index.get(stemmed, []))

        # BM25 IDF formula
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)

        if tf == 0:
            return 0.0

        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length == 0:
            return 0.0

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)
    
    def bm25(self, doc_id, term):
        """
        Compute BM25 score for a single term in a document
        """
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf
    
    def bm25_search(self, query, limit=5):
        """
        Perform BM25 search over all documents
        """
        # preprocess query
        query_tokens = query.lower().translate(translator).split()

        scores = {}

        for doc_id in self.docmap:
            total_score = 0.0

            for token in query_tokens:
                if token in stop_word_list:
                    continue

                total_score += self.bm25(doc_id, token)

            if total_score > 0:
                scores[doc_id] = total_score

        # sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked[:limit]