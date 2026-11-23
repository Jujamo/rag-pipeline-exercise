import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

from vectorstore import load_vectorstore

# Anzahl der Chunks, die dem LLM als Kontext übergeben werden.
TOP_K = 4

DEFAULT_MODEL = "openrouter/free"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Du beantwortest Fragen ausschließlich anhand des bereitgestellten "
            "Kontexts aus Dokumenten. Antworte auf Deutsch, auch wenn der Kontext "
            "englisch ist. Wenn der Kontext die Antwort nicht enthält, sage "
            "ausdrücklich, dass du dazu in den Dokumenten nichts findest, und "
            "erfinde nichts. Nutze kein Wissen außerhalb des Kontexts.",
        ),
        ("human", "Kontext:\n{context}\n\nFrage: {question}"),
    ]
)


def get_llm() -> ChatOpenRouter:
    """Chat-Modell über OpenRouter. Das Modell ist über OPENROUTER_MODEL einstellbar.

    Kostenlose Modelle können überlastet sein, daher Timeout und Wiederholungen.
    """
    return ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        temperature=0,
        request_timeout=120_000,  # Millisekunden
        max_retries=2,
    )


def format_context(docs: list[Document]) -> str:
    """Fügt die Chunks mit Quellenangabe zu einem Kontext-Text zusammen."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[{i}] ({format_source(doc)})\n{doc.page_content}")
    return "\n\n".join(parts)


def format_source(doc: Document) -> str:
    """Lesbare Quellenangabe: Dateiname und, falls vorhanden, Seitenzahl."""
    name = doc.metadata.get("source_file", "unbekannt")
    page = doc.metadata.get("page_label")
    return f"{name}, Seite {page}" if page else name


def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    """Holt die k ähnlichsten Chunks zur Frage aus dem Vektorindex."""
    return load_vectorstore().similarity_search(question, k=k)


def answer(question: str, k: int = TOP_K) -> tuple[str, list[Document]]:
    """Beantwortet eine Frage per RAG. Gibt Antwort und verwendete Chunks zurück."""
    docs = retrieve(question, k=k)
    chain = PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"context": format_context(docs), "question": question})
    return text, docs
