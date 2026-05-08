"""
evaluate.py

Phase 5b — measure how good the RAG system is.

Runs every question in eval_set.py through the RAG pipeline and computes:
  Retrieval metrics:
    - Context Recall:     Did at least one expected_source appear in retrieved chunks?
    - Context Precision:  What fraction of retrieved chunks were from expected sources?
  Generation metrics (LLM-as-judge):
    - Faithfulness:       Are all claims in the answer supported by retrieved context?
    - Answer Relevance:   Does the answer actually address the question?
  Special:
    - Refusal Accuracy:   For adversarial items, did the system correctly say "I don't know"?

CLI:
  python evaluate.py                       # default mode = hybrid
  python evaluate.py --mode dense
  python evaluate.py --mode rerank
  python evaluate.py --compare             # run all 3 modes, print a comparison table
"""

import argparse
import json
import re
from groq import Groq
from dotenv import load_dotenv

from eval_set import EVAL_SET
from rag import ask, get_groq, LLM_MODEL

load_dotenv()


# -----------------------------------------------------------------------------
# RETRIEVAL METRICS (deterministic — no LLM needed)
# -----------------------------------------------------------------------------

def context_recall(retrieved_sources, expected_sources):
    """
    Did at least one expected source make it into retrieval?
    Returns 1.0 if any expected source is in retrieved, 0.0 otherwise.
    For adversarial items (expected_sources is empty), returns 1.0 (trivially).
    """
    if not expected_sources:
        return 1.0
    expected_set = set(expected_sources)
    retrieved_set = set(retrieved_sources)
    hits = expected_set & retrieved_set
    return len(hits) / len(expected_set)


def context_precision(retrieved_sources, expected_sources):
    """
    Of the retrieved chunks, what fraction came from expected sources?
    For adversarial items, undefined — return None.
    """
    if not expected_sources:
        return None
    if not retrieved_sources:
        return 0.0
    expected_set = set(expected_sources)
    relevant = sum(1 for s in retrieved_sources if s in expected_set)
    return relevant / len(retrieved_sources)


# -----------------------------------------------------------------------------
# GENERATION METRICS (LLM-as-judge)
# -----------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an evaluation assistant. You score RAG system outputs.
Output ONLY a JSON object — no preamble, no explanation outside the JSON.
Scores are integers 1 to 5 unless otherwise specified.
"""

FAITHFULNESS_RUBRIC = """Score the answer's faithfulness to the context.

5 = Every factual claim in the answer is directly supported by the context.
4 = Almost all claims are supported; one minor unsupported phrasing.
3 = Mostly supported, but contains a noticeable claim not in context.
2 = Several unsupported claims, looks like the model is filling gaps.
1 = Major hallucinations — claims that contradict or aren't in the context.

If the answer is "I don't know" or refuses, score 5 (vacuously faithful)."""

RELEVANCE_RUBRIC = """Score how directly the answer addresses the question.

5 = Directly and completely answers the question.
4 = Answers the question but includes some off-topic content.
3 = Partially answers; misses key aspect of the question.
2 = Tangentially related; doesn't really answer.
1 = Off-topic or non-responsive.

If the answer is "I don't know" because the context lacks the answer, that's
a 5 if the question is genuinely unanswerable from the context, else lower."""


def parse_score(text):
    """Pull the first integer 1-5 out of an LLM response. Robust to extra text."""
    try:
        obj = json.loads(text)
        return int(obj.get("score"))
    except Exception:
        m = re.search(r'"score"\s*:\s*([1-5])', text)
        if m:
            return int(m.group(1))
        m = re.search(r'\b([1-5])\b', text)
        if m:
            return int(m.group(1))
    return None


def judge_faithfulness(question, context, answer):
    user_msg = f"""{FAITHFULNESS_RUBRIC}

Context provided to the system:
{context}

Question: {question}

System's answer: {answer}

Respond with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""
    resp = get_groq().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    return parse_score(resp.choices[0].message.content)


def judge_relevance(question, answer, context_was_relevant):
    user_msg = f"""{RELEVANCE_RUBRIC}

Question: {question}

System's answer: {answer}

(Note: the retrieved context {"did" if context_was_relevant else "did not"} contain the answer.)

Respond with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""
    resp = get_groq().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    return parse_score(resp.choices[0].message.content)


# -----------------------------------------------------------------------------
# REFUSAL DETECTION FOR ADVERSARIAL ITEMS
# -----------------------------------------------------------------------------

REFUSAL_MARKERS = [
    "i don't know",
    "i do not know",
    "based on the available documents",
    "not in the context",
    "context does not contain",
    "no information",
]

def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)


# -----------------------------------------------------------------------------
# RUN EVAL
# -----------------------------------------------------------------------------

def run_eval(mode: str, k: int = 5, verbose: bool = True):
    """Run the full eval set in `mode`. Returns aggregate metrics + per-item rows."""
    rows = []

    for item in EVAL_SET:
        q = item["question"]
        if verbose:
            print(f"\n[{item['difficulty']}] {q}")

        result = ask(q, mode=mode, k=k)
        retrieved_sources = [s["source"] for s in result["sources"]]
        answer = result["answer"]
        context = "\n\n".join(s["text"] for s in result["sources"])

        # Retrieval metrics
        recall = context_recall(retrieved_sources, item["expected_sources"])
        precision = context_precision(retrieved_sources, item["expected_sources"])

        # Adversarial: did the system refuse?
        if item["difficulty"] == "adversarial":
            refused = is_refusal(answer)
            faith = relev = None   # rubrics handle refusals as 5 anyway, but skip the call
            if verbose:
                print(f"  refused: {refused}  (answer: {answer[:120]}...)")
        else:
            refused = None
            faith = judge_faithfulness(q, context, answer)
            relev = judge_relevance(q, answer, context_was_relevant=(recall > 0))
            if verbose:
                print(f"  recall: {recall:.2f}  precision: {precision:.2f}  "
                      f"faith: {faith}  relevance: {relev}")

        rows.append({
            "difficulty": item["difficulty"],
            "question": q,
            "recall": recall,
            "precision": precision,
            "faithfulness": faith,
            "relevance": relev,
            "refused": refused,
            "answer": answer,
        })

    # Aggregate
    nonadv = [r for r in rows if r["difficulty"] != "adversarial"]
    adv = [r for r in rows if r["difficulty"] == "adversarial"]
    summary = {
        "mode": mode,
        "n_total": len(rows),
        "n_nonadversarial": len(nonadv),
        "n_adversarial": len(adv),
        "avg_recall": _avg(r["recall"] for r in nonadv),
        "avg_precision": _avg(r["precision"] for r in nonadv if r["precision"] is not None),
        "avg_faithfulness": _avg(r["faithfulness"] for r in nonadv if r["faithfulness"] is not None),
        "avg_relevance": _avg(r["relevance"] for r in nonadv if r["relevance"] is not None),
        "refusal_accuracy": (
            _avg(1.0 if r["refused"] else 0.0 for r in adv) if adv else None
        ),
    }
    return summary, rows


def _avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


# -----------------------------------------------------------------------------
# REPORTING
# -----------------------------------------------------------------------------

def print_summary(summary):
    print("\n" + "=" * 70)
    print(f"SUMMARY — mode: {summary['mode']}")
    print("=" * 70)
    print(f"  Items evaluated: {summary['n_total']} "
          f"({summary['n_nonadversarial']} non-adversarial, "
          f"{summary['n_adversarial']} adversarial)")
    print()
    print(f"  RETRIEVAL:")
    print(f"    avg context recall:     {_fmt(summary['avg_recall'])}")
    print(f"    avg context precision:  {_fmt(summary['avg_precision'])}")
    print()
    print(f"  GENERATION (1-5 scale, LLM-judged):")
    print(f"    avg faithfulness:       {_fmt(summary['avg_faithfulness'])}")
    print(f"    avg answer relevance:   {_fmt(summary['avg_relevance'])}")
    print()
    print(f"  ADVERSARIAL:")
    print(f"    refusal accuracy:       {_fmt(summary['refusal_accuracy'])}")


def _fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.2f}"


def print_comparison(summaries):
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    cols = ["mode", "recall", "precision", "faith", "relev", "refusal"]
    print(f"  {'mode':<10} {'recall':>8} {'precision':>10} "
          f"{'faith':>7} {'relev':>7} {'refusal':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*7} {'-'*7} {'-'*8}")
    for s in summaries:
        print(f"  {s['mode']:<10} "
              f"{_fmt(s['avg_recall']):>8} "
              f"{_fmt(s['avg_precision']):>10} "
              f"{_fmt(s['avg_faithfulness']):>7} "
              f"{_fmt(s['avg_relevance']):>7} "
              f"{_fmt(s['refusal_accuracy']):>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid", "rerank"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--compare", action="store_true",
                        help="Run all 3 modes and print a comparison")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.compare:
        summaries = []
        for mode in ["dense", "hybrid", "rerank"]:
            print(f"\n>>> Running mode: {mode}")
            summary, _rows = run_eval(mode, k=args.k, verbose=not args.quiet)
            print_summary(summary)
            summaries.append(summary)
        print_comparison(summaries)
    else:
        summary, _rows = run_eval(args.mode, k=args.k, verbose=not args.quiet)
        print_summary(summary)


if __name__ == "__main__":
    main()
