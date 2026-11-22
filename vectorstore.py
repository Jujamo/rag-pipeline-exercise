from pathlib import Path

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "documents"


class E5Embeddings(Embeddings):
    """Lokales Embedding über fastembed (ONNX, ohne PyTorch).

    e5-Modelle wurden mit Präfixen trainiert und liefern ohne sie schlechtere
    Ergebnisse: "passage: " für zu indexierende Texte, "query: " für Fragen.
    fastembed setzt diese Präfixe für e5 nicht automatisch, daher hier explizit.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return [v.tolist() for v in self._model.embed(prefixed)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([f"query: {text}"]))).tolist()


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Baut den Index neu auf und speichert ihn persistent auf der Platte.

    Eine bestehende Collection wird vorher gelöscht, damit ein erneuter Lauf
    keine doppelten Chunks erzeugt.
    """
    embeddings = E5Embeddings()
    Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    ).delete_collection()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        # Kosinus statt Chroma-Standard (L2): passt zu e5 und liefert Scores in [0, 2].
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks)
    return store


def load_vectorstore() -> Chroma:
    """Öffnet den bereits gebauten Index."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=E5Embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
