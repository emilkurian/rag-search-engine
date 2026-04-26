import argparse
import json
import string
from nltk.stem import PorterStemmer
import os
import pickle

stemmer = PorterStemmer()


translator = str.maketrans("", "", string.punctuation)

with open("data/stopwords.txt", "r") as stop_words:
    stop_word_list = stop_words.read().splitlines()


class InvertedIndex:

    def __init__(self):
        self.index = dict()
        self.docmap = dict()

    def __add_document(self, doc_id, text):
        """
        Tokenize text and add tokens to the index
        """
        # store full document
        self.docmap[doc_id] = text

        # normalize text
        text = text.lower().translate(translator)

        # tokenize
        tokens = text.split()

        for token in tokens:
            # remove stopwords
            if token in stop_word_list:
                continue

            # stem token
            stemmed = stemmer.stem(token)

            # add to inverted index
            if stemmed not in self.index:
                self.index[stemmed] = set()

            self.index[stemmed].add(doc_id)

    def get_documents(self, term):
        """
        Return sorted list of document IDs for a term
        """
        pass

    def build(self, movies):
        """
        Build index from all movie documents
        """
        pass

    def save(self):
        """
        Save index and docmap to disk using pickle
        """
        pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # movie_query_list = [title for title in movie_data_dict.keys() if args.query in title.lower()]

    match args.command:
        case "search":
            with open("data/movies.json", "r") as movie_data:
                movie_data_dict = json.load(movie_data)


            # print(movie_data_dict)
            print(f"Searching for: {args.query}")
            #print(stop_word_list)
            query = args.query.lower().translate(translator).split()
            movie_query_list = [movie["title"] 
                                for movie in movie_data_dict["movies"] 
                                if any(
                                    stemmer.stem(q) in stemmer.stem(t)
                                    for q in query if q not in stop_word_list
                                    for t in movie["title"].lower().translate(translator).split() 
                                    if t.lower() not in stop_word_list
                                    if q and t
                                )]
            # print(movie_query_list)

            for i, title in enumerate(movie_query_list[:5], 1):
                print(f"{i}. {title}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()