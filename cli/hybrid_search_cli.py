import argparse
import json
from lib.hybrid_search import HybridSearch


def load_movies():
    with open("data/movies.json", "r") as f:
        data = json.load(f)
        return data["movies"] if isinstance(data, dict) else data


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # stub command (you’ll expand later)
    search_parser = subparsers.add_parser("search", help="Hybrid search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "search":
            documents = load_movies()
            hybrid = HybridSearch(documents)

            # stub output for now
            print(f"Hybrid search initialized for query: {args.query}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()