# RAG-Pipeline

Kleine Retrieval-Augmented-Generation-Pipeline: Sie liest die Dokumente in `data/` ein, indexiert sie in einer lokalen Vektordatenbank und beantwortet Fragen dazu mit einem LLM. Die Antworten enthalten Quellenangaben (Datei und Seite).

Aufgabenstellung: [lukasflaig/rag-pipeline-exercise](https://github.com/lukasflaig/rag-pipeline-exercise)

## Funktionsweise

```
ingest:  data/*.pdf, *.docx -> Text laden -> Chunks (1000/150) -> Embeddings (e5-large, lokal) -> Chroma
ask:     Frage -> Embedding -> Top-4-Chunks aus Chroma -> Prompt (Kontext + Frage) -> LLM (OpenRouter) -> Antwort + Quellen
```

Es gibt zwei Modelle: 
- ein lokales Embedding-Modell (Text zu Vektor)
- ein Chat-Modell über OpenRouter, das die Antwort formuliert

## Setup

Voraussetzungen: Python 3.13+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
copy .env.example .env      # Linux/macOS: cp .env.example .env
```

In `.env` den OpenRouter-Key eintragen (`OPENROUTER_API_KEY`, erzeugen unter https://openrouter.ai/settings/keys). Optional kann man mit `OPENROUTER_MODEL` ein festes Modell wählen, Standard ist `openrouter/free`.

Alternativ mit pip: `pip install -r requirements.txt`.

## Nutzung

```bash
uv run main.py ingest                                  # Index aufbauen (einmalig)
uv run main.py ask "Was ist Azure AI Foundry?"         # Frage stellen
```

Hinweise zum ersten Lauf:
- `ingest` lädt beim ersten Mal das Embedding-Modell (ca. 2,2 GB) und braucht danach auf der CPU etwa 5 Minuten für die 201 Chunks.
- Der Index liegt in `chroma_db/` (nicht versioniert). Ein erneutes `ingest` ersetzt ihn vollständig und es entstehen keine Duplikate.

## Projektstruktur

```
main.py         CLI (ingest, ask)
ingest.py       Dokumente laden und in Chunks teilen
vectorstore.py  Embedding-Modell und Chroma-Index
rag.py          Retrieval, Prompt, LLM-Aufruf
data/           Quelldokumente
```

## Designentscheidungen

**Laden und Chunking:** Ich lade die Dokumente mit `PyPDFLoader` und `Docx2txtLoader`. Docling hätte Struktur und Tabellen besser erhalten, wäre mit seinen schweren Abhängigkeiten für vier Dokumente aber zu groß gewesen. Geteilt wird mit dem`RecursiveCharacterTextSplitter` (1000 Zeichen, 150 Overlap), der an Absätzen und Sätzen trennt. Andere Größen habe ich ausprobiert (siehe Experimente).

**Embeddings:** Die Dokumente sind englisch, die Fragen deutsch, deshalb nutze ich `intfloat/multilingual-e5-large`. Es ist für Retrieval trainiert, verarbeitet bis zu 512 Tokens und läuft lokal über fastembed, ohne API und PyTorch.
Der Nachteil ist ein großer Download (2,2 GB) und ein langsames Embedding auf der CPU.

**Vektordatenbank und Retrieval:**  Ich nutze Chroma lokal mit Konsinus-Distanz (statt Standard-Distanz L2, die schlechter zu e5 passt) und hole pro Frage die vier ähnlichsten Chunks. MMR und ein höheres k habe ich getestet, beides war nicht besser.

**Antwort:** Eine einfache LangChain-Kette schickt die Chunks und die Frage an ein Modell über OpenRouter. Der Prompt erlaubt nur Antworten aus dem Kontext und verlangt, offen zu sagen, wenn die Antwort fehlt. Das Modell lässt sich über `OPENROUTER_MODEL`austauschen, weil kostenlose Modelle ausfallen oder überlastet sein können.

## Experimente

Gemessen habe ich mit den vier Beispielfragen. Als ungefährer Maßstab hat gedient, wie viele erwartete Schlüsselbegriffe im abgerufenen Kontext vorkommen. Durchgeführt als Näherungswert für die Retrieval-Qualität, aber kein Test der Antwortqualität.

**Retrieval-Verfahren (Begriffe im Kontext)**

| Frage | sim k=4 | sim k=6 | MMR k=4 | MMR k=6 |
|---|---|---|---|---|
| FastAPI | 3/3 | 3/3 | 0/3 | 2/3 |
| RAG-Paradigmen | 3/3 | 3/3 | 3/3 | 3/3 |
| Azure AI Foundry | 3/3 | 3/3 | 2/3 | 2/3 |
| Docling-Formate | 2/6 | 2/6 | 2/6 | 2/6 |

Ergebnis: Ich hatte erwartet, dass MMR hier besser abschneiden könnte. Es holt aber zusätzliche Seiten aus anderen Dokumenten in den Kontext, obwohl die Antwort in einem Dokument steht. Deshalb bleibe ich bei der einfachen Ähnlichkeitssuche mit k=4

**Chunk-Größe**

| Variante | Chunks | FastAPI | RAG | Azure | Docling (Begriffe, bester Rang) |
|---|---|---|---|---|---|
| 1000/150 | 201 | 3/3 | 3/3 | 3/3 | 2/6, Rang 11 |
| 600/100 | 338 | 2/3 | 3/3 | 3/3 | 2/6, Rang 6 |
| 400/80 | 541 | 2/3 | 3/3 | 3/3 | 2/6, Rang 15 |

Ergebnis: Kleinere Chunks verschlechtern die FastAPI-Frage und lösen das Docling-Problem nicht. Ich habe 1000/150 gewählt.

**Sprache der Anfrage:** Die Docling-Frage habe ich zusätzlich auf Englisch gestellt. Der Rang des relevanten Chunks blieb beinahe gleich (18 bis 19 gegenüber 18 bei der deutschen Frage). Eine Übersetzung der Frage vor der Suche bringt also keine Verbesserung.

##  Grenzen und nächste Schritte

- **Docling-Formatfrage:** Die Antwort bleibt unvollständig. Der Chunk mit der Formatliste (Word, PowerPoint, Excel, AsciiDoc) landet im Ranking auf Platz 18 und damit nicht in den Top 4. An der Sprache, an k oder an der Chunk-Größe liegt es nicht, die genaue Ursache kenne ich noch nicht. Als Nächstes würde ich eine hybride Suche (BM25 plus Vektorsuche) oder einen Reranker ausprobieren. Auch Docling als Loader könnte sauberere Chunks liefern.
- **Quellanzeige:** Auch wenn das LLM sagt, dass es in den Dokumenten nichts findet, werden die abgerufenen Chunks als Quellen aufgelistet. Das ist irreführend und könnte durch einen Schwellenwert für die Distanz behoben werden.
- **Qualität und Messung:** `openrouter/free` wählt bei jeder Anfrage ein anderes kostenloses Modell, Stil und Ausführlichkeit können deshalb abweichen. Ein festes Modell wäre reproduzierbar. Meine Experimente beruhen nur auf 4 Fragen. Für belastbare Aussagen bräuchte ich ein größeres Set mit Referenzantworten und eine automatische Bewertung.


## Entstehung

Die Umsetzung entstand in Zusammenarbeit mit dem KI-Assistenten Claude (Claude Code). Die Designoptionen habe ich mit Claude abgewogen, die Designentscheidungen wurden eigenständig getroffen, die Experimente habe ich nachvollzogen und ausgewertet.

## Hinweis zu Dateinamen unter Windows

Die Datei `data/What is FastAPI?.docx` enthielt ein `?` im Namen, das Windows nicht erlaubt. Sie heißt im Repository jetzt `What is FastAPI.docx`.
