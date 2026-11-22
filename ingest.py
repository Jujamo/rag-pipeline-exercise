from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent / "data"

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
