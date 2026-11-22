import os
import sys

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# Windows-Konsolen nutzen oft cp1252; Ausgabe daher explizit auf UTF-8 stellen.
sys.stdout.reconfigure(encoding="utf-8")

# Modell ist über .env konfigurierbar, da kostenlose Modelle wechseln können.
model = ChatOpenRouter(
    model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
)


def main():
    messages = [
        (
            "system",
            "Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
        ),
        ("human", "Was ist ein RAG-System? Erkläre es in einem Satz."),
    ]
    ai_msg = model.invoke(messages)
    print(ai_msg.content)


if __name__ == "__main__":
    main()
