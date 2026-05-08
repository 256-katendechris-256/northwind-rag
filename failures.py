"""
failures.py

Phase 5c — deliberate failure-mode tests.

Triggers each canonical RAG failure mode on purpose and reports observations.
Use these directly in interview answers for Q7, Q8, and Q10.

Run with:
  python failures.py
  python failures.py --test out_of_domain
  python failures.py --test retrieval_miss
  python failures.py --test hallucination
  python failures.py --test stale_data
  python failures.py --test context_overflow
"""

import argparse
from groq import Groq
from dotenv import load_dotenv

from rag import ask, get_groq, LLM_MODEL, SYSTEM_PROMPT
from retriever import dense_search

load_dotenv()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def hr(label):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)


def truncate(s, n=200):
    s = s.replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


def looks_like_refusal(answer):
    a = answer.lower()
    return ("i don't know" in a) or ("i do not know" in a) or \
           ("not in the context" in a) or ("based on the available documents" in a)


# -----------------------------------------------------------------------------
# FAILURE MODE 1 — OUT OF DOMAIN
# Question has no answer in the corpus. Correct behavior: refuse.
# -----------------------------------------------------------------------------

def test_out_of_domain():
    hr("FAILURE 1 — OUT OF DOMAIN")
    print("Question is unanswerable from the corpus. Correct behavior = refuse.\n")

    queries = [
        "Who is the current CEO of Microsoft?",
        "What is the boiling point of water?",
        "Who won the FIFA World Cup in 2022?",
    ]

    for q in queries:
        result = ask(q, mode="hybrid", k=5)
        print(f"Q: {q}")
        print(f"  retrieved sources: {[s['source'] for s in result['sources']]}")
        print(f"  answer: {truncate(result['answer'], 250)}")
        print(f"  refused? {looks_like_refusal(result['answer'])}\n")

    print("Interview point: with a strict system prompt + temperature 0, the model")
    print("should refuse. If it doesn't, the system prompt is too permissive.")


# -----------------------------------------------------------------------------
# FAILURE MODE 2 — RETRIEVAL MISS
# The answer IS in the corpus but retrieval pulls the wrong chunks.
# Demonstrates with the comparison-query failure pattern.
# -----------------------------------------------------------------------------

def test_retrieval_miss():
    hr("FAILURE 2 — RETRIEVAL MISS")
    print("Answer is in the corpus, but retrieval doesn't surface it.")
    print("Comparison queries are the canonical failure pattern.\n")

    q = "What are Maya and James's official titles?"
    result = ask(q, mode="hybrid", k=5)
    print(f"Q: {q}")
    print(f"  retrieved sources: {[s['source'] for s in result['sources']]}")
    print(f"  answer: {truncate(result['answer'], 350)}")

    # Reveal the ground truth so you can see the gap.
    print("\nGround truth from corpus headers:")
    print("  - James Aldridge: Senior Partner")
    print("  - Maya Okonkwo: Managing Partner")

    print("\nInterview point: the role info lives in transcript headers that get")
    print("buried by character-based chunking. The fix is structural — extract")
    print("metadata at ingest, attach to every chunk from that file. Retrieval")
    print("alone can't fix this.")


# -----------------------------------------------------------------------------
# FAILURE MODE 3 — HALLUCINATION ON TOP OF IRRELEVANT CONTEXT
# Force-feed unrelated chunks, ask a fact question.
# A weak prompt lets the LLM fall back on training data.
# A strict prompt forces refusal.
# -----------------------------------------------------------------------------

def test_hallucination():
    hr("FAILURE 3 — HALLUCINATION ON TOP OF IRRELEVANT CONTEXT")
    print("Same question, two prompts — strict vs permissive.")
    print("Same retrieved chunks (deliberately irrelevant in both cases).\n")

    question = "What's the capital of France?"

    # Pull 3 deliberately-irrelevant chunks from our corpus.
    irrelevant_hits = dense_search("Northwind methodology stages", k=3)
    irrelevant_context = "\n\n".join(
        f"[{i+1}] ({h['metadata'].get('source', '?')}):\n{h['text']}"
        for i, h in enumerate(irrelevant_hits)
    )

    # Build two prompts — same context, same question, different system messages.
    permissive_system = (
        "You are a helpful assistant. Answer the user's question."
    )

    user_msg = f"""Context:\n{irrelevant_context}\n\nQuestion: {question}\n\nAnswer:"""

    client = get_groq()

    # Strict (our actual system prompt)
    strict = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    ).choices[0].message.content.strip()

    # Permissive
    permissive = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": permissive_system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    ).choices[0].message.content.strip()

    print(f"Q: {question}")
    print(f"  retrieved (irrelevant) sources: "
          f"{[h['metadata'].get('source','?') for h in irrelevant_hits]}\n")
    print(f"  STRICT system prompt    -> {truncate(strict, 250)}")
    print(f"  PERMISSIVE system prompt -> {truncate(permissive, 250)}")

    print("\nInterview point: with a permissive prompt, the LLM ignores the (irrelevant)")
    print("context and answers from training data. With a strict prompt that says")
    print("\"answer using ONLY the context\", it refuses. The system prompt is the")
    print("cheapest place to enforce groundedness.")


# -----------------------------------------------------------------------------
# FAILURE MODE 4 — STALE DATA
# Simulate a doc update that the index hasn't seen.
# This is more architectural — we describe the failure rather than trigger it.
# -----------------------------------------------------------------------------

def test_stale_data():
    hr("FAILURE 4 — STALE DATA / FRESHNESS")
    print("Simulated: corpus says Discovery is £25,000, but suppose pricing")
    print("changed yesterday to £30,000.")
    print("If the index hasn't been re-embedded, the system answers with the old price.\n")

    q = "How much does Discovery cost?"
    result = ask(q, mode="hybrid", k=3)
    print(f"Q: {q}")
    print(f"  answer: {truncate(result['answer'], 250)}")
    print(f"  retrieved from: {[s['source'] for s in result['sources']]}")

    print("\nInterview point: this is the freshness problem (Q10).")
    print("Three production strategies:")
    print("  1. Per-doc hashing — re-embed only when a doc's content hash changes.")
    print("  2. Scheduled re-ingestion — full or incremental, on a cron.")
    print("  3. Webhook-driven — source systems (Notion, Drive) push change events.")
    print("Naive approach: full re-ingest nightly. Production: incremental + hashes.")


# -----------------------------------------------------------------------------
# FAILURE MODE 5 — CONTEXT WINDOW OVERFLOW
# Stuff way too many chunks into the prompt and watch it strain.
# -----------------------------------------------------------------------------

def test_context_overflow():
    hr("FAILURE 5 — CONTEXT WINDOW OVERFLOW")
    print("Retrieval pulls k=50 chunks. The prompt balloons.")
    print("Modern long-context models survive this; smaller ones truncate or error.\n")

    q = "What's the refund policy?"

    # Use k=50 — much larger than sensible. Most of these chunks are irrelevant.
    big_result = ask(q, mode="dense", k=50)
    prompt_chars = len(big_result["user_prompt"])
    # very rough: ~4 chars per token in English
    approx_tokens = prompt_chars // 4

    print(f"Q: {q}, k=50")
    print(f"  prompt length: {prompt_chars:,} chars (~{approx_tokens:,} tokens)")
    print(f"  answer: {truncate(big_result['answer'], 250)}\n")

    # Compare with a sensible k.
    small = ask(q, mode="dense", k=3)
    print(f"  Same question with k=3:")
    print(f"  answer: {truncate(small['answer'], 250)}")

    print("\nInterview point: more context isn't always better.")
    print("  - 'Lost in the middle' — LLMs attend less to the middle of long prompts.")
    print("  - Cost scales linearly with input tokens.")
    print("  - Latency rises with input size.")
    print("Reranking lets you cast a wide net (k=20) AND keep the prompt small")
    print("(top 3-5 reranked chunks).")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

ALL_TESTS = {
    "out_of_domain": test_out_of_domain,
    "retrieval_miss": test_retrieval_miss,
    "hallucination": test_hallucination,
    "stale_data": test_stale_data,
    "context_overflow": test_context_overflow,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=list(ALL_TESTS.keys()),
                        default=None,
                        help="Run a single test. Default: run all.")
    args = parser.parse_args()

    if args.test:
        ALL_TESTS[args.test]()
    else:
        for fn in ALL_TESTS.values():
            fn()


if __name__ == "__main__":
    main()
