import argparse
import sys

from dotenv import load_dotenv

from ingest import load_documents, split_documents
from vectorstore import CHROMA_DIR, build_vectorstore

load_dotenv()

# Windows-Konsolen nutzen oft cp1252; Ausgabe daher explizit auf UTF-8 stellen.
sys.stdout.reconfigure(encoding="utf-8")


def cmd_ingest(_args: argparse.Namespace) -> None:
    """Dokumente laden, in Chunks teilen und den Vektorindex neu aufbauen."""
    documents = load_documents()
    print(f"{len(documents)} Seiten/Dokumente geladen")

    chunks = split_documents(documents)
    print(f"{len(chunks)} Chunks erzeugt")

    print("Erzeuge Embeddings (beim ersten Lauf wird das Modell geladen) ...")
    build_vectorstore(chunks)
    print(f"Index gespeichert in {CHROMA_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG-Pipeline: Fragen zu den Dokumenten in data/ beantworten."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index aus data/ neu aufbauen")
    ingest.set_defaults(func=cmd_ingest)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
