"""
rag.py

Phase 5a — generation. Wires the retriever into Groq's LLM API to produce
end-to-end answers from question -> retrieved chunks -> grounded answer.

CLI:
  python rag.py --query "What's the refund policy?"
  python rag.py --query "..." --mode rerank        # pick retrieval mode
  python rag.py --query "..." -k 5                 # number of chunks to retrieve
  python rag.py --query "..." --show-context       # see what was sent to the LLM
"""

import argparse
import os
from dotenv import load_dotenv
from groq import Groq

from retriever import dense_search, hybrid_search, rerank_search

# Load GROQ_API_KEY from .env in current dir.
load_dotenv()


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

LLM_MODEL = "llama-3.3-70b-versatile"   # production-grade open-weight model on Groq
DEFAULT_K = 5                            # how many chunks to retrieve for context

# Lazy Groq client
_groq = None
def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq()    # reads GROQ_API_KEY from env
    return _groq


# -----------------------------------------------------------------------------
# PROMPT CONSTRUCTION
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledge assistant for Northwind Advisory, a consultancy.
Your job is to answer questions using ONLY the context provided below.

Rules:
- Answer using ONLY the information in the numbered context chunks.
- When you use information from a chunk, cite it inline like [1], [2], etc.
- If the context does not contain the answer, say exactly: "I don't know based on the available documents."
- Do not invent facts, names, numbers, dates, or quotes that aren't in the context.
- Keep answers concise and direct. No preamble.
"""


def build_user_prompt(question: str, hits: list) -> str:
    """Format chunks + question into the user message sent to the LLM."""
    context_blocks = []
    for i, h in enumerate(hits, 1):
        source = h["metadata"].get("source", "unknown")
        text = h["text"].strip()
        context_blocks.append(f"[{i}] ({source})\n{text}")
    context = "\n\n".join(context_blocks)

    return f"""Context:
{context}

Question: {question}

Answer:"""


# -----------------------------------------------------------------------------
# RETRIEVAL DISPATCH
# -----------------------------------------------------------------------------

RETRIEVAL_MODES = {
    "dense": dense_search,
    "hybrid": hybrid_search,
    "rerank": rerank_search,
}


# -----------------------------------------------------------------------------
# MAIN ASK FUNCTION
# -----------------------------------------------------------------------------

def ask(question: str, mode: str = "hybrid", k: int = DEFAULT_K):
    """
    End-to-end RAG: retrieve -> build prompt -> call LLM -> return answer + sources.

    Returns a dict:
      {
        "answer": str,
        "sources": [{"id", "source", "source_type", "text"}],
        "mode": str,
      }
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    # Step 1: retrieve
    retriever_fn = RETRIEVAL_MODES[mode]
    hits = retriever_fn(question, k=k)

    # Step 2: build the prompt
    user_prompt = build_user_prompt(question, hits)

    # Step 3: call the LLM
    client = get_groq()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,   # deterministic, factual
    )
    answer = response.choices[0].message.content.strip()

    # Step 4: return answer + sources
    sources = [
        {
            "citation": i + 1,
            "id": h["id"],
            "source": h["metadata"].get("source", "unknown"),
            "source_type": h["metadata"].get("source_type", "unknown"),
            "text": h["text"],
        }
        for i, h in enumerate(hits)
    ]
    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,
        "user_prompt": user_prompt,   # so --show-context can display it
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def print_response(result, show_context: bool = False):
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print(f"SOURCES (retrieval mode: {result['mode']})")
    print("=" * 70)
    for src in result["sources"]:
        print(f"\n[{src['citation']}] [{src['source_type']}] {src['source']}")
        preview = src["text"][:200].replace("\n", " ")
        print(f"    {preview}{'...' if len(src['text']) > 200 else ''}")

    if show_context:
        print("\n" + "=" * 70)
        print("PROMPT SENT TO LLM")
        print("=" * 70)
        print(result["user_prompt"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mode", choices=list(RETRIEVAL_MODES.keys()),
                        default="hybrid")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--show-context", action="store_true",
                        help="Print the full prompt sent to the LLM")
    args = parser.parse_args()

    print(f"Question: {args.query}")
    print(f"Retrieval mode: {args.mode}, k={args.k}")

    result = ask(args.query, mode=args.mode, k=args.k)
    print_response(result, show_context=args.show_context)


if __name__ == "__main__":
    main()
