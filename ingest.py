from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"

# Chunking-Parameter: groß genug für zusammenhängenden Kontext, klein genug
# für präzise Treffer. Der Overlap verhindert, dass Sätze an Grenzen zerrissen
# werden und Information verloren geht.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Zuordnung Dateiendung -> Loader. Neue Formate lassen sich hier ergänzen.
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lädt alle unterstützten Dateien aus data_dir als LangChain-Dokumente.

    PDFs ergeben ein Dokument pro Seite, DOCX-Dateien ein Dokument pro Datei.
    Der Dateiname steht in den Metadaten unter "source_file".
    """
    documents: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        docs = loader_cls(str(path)).load()
        for doc in docs:
            doc.metadata["source_file"] = path.name
        documents.extend(docs)
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Teilt Dokumente in überlappende Chunks.

    Der Splitter trennt bevorzugt an Absätzen, dann an Zeilen, Sätzen und
    Wörtern, damit Chunks möglichst an natürlichen Grenzen enden.
    Die Metadaten (Quelle, Seite) bleiben pro Chunk erhalten.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
