# Northwind RAG — interview prep project

A working RAG system over a fake consultancy's mixed-format knowledge base,
built end-to-end across five phases. See CHEATSHEET.md for interview
answers grounded in this project.

## One-time setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # paste your Groq key
```

## Run each phase

```bash
source venv/bin/activate

# Phase 2 — load + chunk
python loader.py

# Phase 3 — embed + dense retrieval
python index.py
python index.py --query "..."

# Phase 4 — hybrid + reranking
python retriever.py --query "..." --compare

# Phase 5a — end-to-end RAG (retrieval + generation)
python rag.py --query "..."

# Phase 5b — evaluation
python evaluate.py --compare

# Phase 5c — failure-mode tests
python failures.py
python failures.py --test hallucination
```

## Folder layout

```
rag-project/
├── corpus/                # the knowledge base
├── chroma_db/             # vector index (auto-created)
├── loader.py              # Phase 2
├── index.py               # Phase 3
├── retriever.py           # Phase 4
├── rag.py                 # Phase 5a
├── eval_set.py            # Phase 5b — labeled set
├── evaluate.py            # Phase 5b — eval runner
├── failures.py            # Phase 5c — failure tests
├── CHEATSHEET.md          # interview answers grounded in this project
├── requirements.txt
├── .env.example
└── README.md
```
