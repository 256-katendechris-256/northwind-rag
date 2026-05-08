"""
retriever.py

Phase 4 — hybrid search and re-ranking.

This module builds on top of the Chroma index from Phase 3 by adding:
  1. A BM25 keyword index over the same chunks
  2. Reciprocal Rank Fusion to merge dense + sparse results
  3. An optional cross-encoder re-ranker for final ordering

CLI:
  python retriever.py --query "your question" --mode dense
  python retriever.py --query "your question" --mode hybrid
  python retriever.py --query "your question" --mode rerank
  python retriever.py --query "your question" --compare      # all 3 modes side-by-side
"""

import argparse
from rank_bm25 import BM25Okapi

from index import (
    get_chroma_client,
    get_or_create_collection,
    embed,
    COLLECTION_NAME,
)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Re-ranker only loaded when actually used (it's another model download).
_reranker = None
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# -----------------------------------------------------------------------------
# LOAD ALL CHUNKS FROM CHROMA INTO MEMORY
# We need the raw text + ids in memory to build BM25, and to look up
# chunks by id when fusing dense/sparse results.
# -----------------------------------------------------------------------------

_chunks_cache = None

def get_all_chunks():
    """Pull every chunk out of Chroma so we can build BM25 and look up by id."""
    global _chunks_cache
    if _chunks_cache is not None:
        return _chunks_cache

    client = get_chroma_client()
    coll = get_or_create_collection(client)
    if coll.count() == 0:
        raise RuntimeError(
            "Chroma index is empty. Run `python index.py` first."
        )

    # .get() with no filter returns everything.
    raw = coll.get(include=["documents", "metadatas"])
    chunks = []
    for i in range(len(raw["ids"])):
        chunks.append({
            "id": raw["ids"][i],
            "text": raw["documents"][i],
            "metadata": raw["metadatas"][i],
        })
    _chunks_cache = chunks
    return chunks


# -----------------------------------------------------------------------------
# BM25 INDEX
# Built in memory at first use. For 69 chunks this is instant.
# -----------------------------------------------------------------------------

_bm25_cache = None
_bm25_chunks_cache = None

def get_bm25():
    """Build (or reuse) a BM25 index over all chunks."""
    global _bm25_cache, _bm25_chunks_cache
    if _bm25_cache is not None:
        return _bm25_cache, _bm25_chunks_cache

    chunks = get_all_chunks()
    # Tokenise each chunk: lowercase + naive split. BM25 is robust to crude
    # tokenisation; a stemmer would help marginally but isn't needed here.
    tokenised = [c["text"].lower().split() for c in chunks]
    _bm25_cache = BM25Okapi(tokenised)
    _bm25_chunks_cache = chunks
    return _bm25_cache, _bm25_chunks_cache


# -----------------------------------------------------------------------------
# RETRIEVAL MODES
# -----------------------------------------------------------------------------

def dense_search(query: str, k: int = 5):
    """Pure vector search via Chroma."""
    client = get_chroma_client()
    coll = get_or_create_collection(client)
    q_emb = embed([query])[0]
    raw = coll.query(query_embeddings=[q_emb], n_results=k)
    hits = []
    for i in range(len(raw["ids"][0])):
        hits.append({
            "id": raw["ids"][0][i],
            "text": raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": raw["distances"][0][i],
        })
    return hits


def sparse_search(query: str, k: int = 5):
    """Pure BM25 search."""
    bm25, chunks = get_bm25()
    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    # Get top k by score.
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for idx in ranked:
        hits.append({
            "id": chunks[idx]["id"],
            "text": chunks[idx]["text"],
            "metadata": chunks[idx]["metadata"],
            "bm25_score": float(scores[idx]),
        })
    return hits


def hybrid_search(query: str, k: int = 5, candidates: int = 20):
    """
    Hybrid search via Reciprocal Rank Fusion.
    Pulls `candidates` from each retriever, fuses, returns top k.
    """
    dense_hits = dense_search(query, k=candidates)
    sparse_hits = sparse_search(query, k=candidates)

    K = 60   # RRF smoothing constant. 60 is the standard value.
    rrf_scores = {}
    chunk_lookup = {}

    # Add up reciprocal-rank contributions from each retriever.
    for rank, hit in enumerate(dense_hits):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0) + 1 / (K + rank + 1)
        chunk_lookup[hit["id"]] = hit
    for rank, hit in enumerate(sparse_hits):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0) + 1 / (K + rank + 1)
        chunk_lookup.setdefault(hit["id"], hit)

    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    hits = []
    for chunk_id in top_ids:
        h = dict(chunk_lookup[chunk_id])
        h["rrf_score"] = rrf_scores[chunk_id]
        hits.append(h)
    return hits


def rerank_search(query: str, k: int = 5, candidates: int = 20):
    """
    Hybrid search followed by a cross-encoder rerank.
    First pass collects `candidates` via hybrid; second pass scores and
    reorders them using the reranker; returns top k.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker {RERANKER_MODEL_NAME}...")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)

    candidate_hits = hybrid_search(query, k=candidates, candidates=candidates)
    pairs = [(query, h["text"]) for h in candidate_hits]
    scores = _reranker.predict(pairs)

    for h, s in zip(candidate_hits, scores):
        h["rerank_score"] = float(s)

    candidate_hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return candidate_hits[:k]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def show_hits(label, hits):
    print(f"\n--- {label} ---")
    for i, h in enumerate(hits, 1):
        src = h["metadata"].get("source", "?")
        stype = h["metadata"].get("source_type", "?")
        # Pick the most relevant score to display per mode.
        if "rerank_score" in h:
            score_str = f"rerank: {h['rerank_score']:+.3f}"
        elif "rrf_score" in h:
            score_str = f"rrf: {h['rrf_score']:.4f}"
        elif "bm25_score" in h:
            score_str = f"bm25: {h['bm25_score']:.2f}"
        else:
            score_str = f"distance: {h['distance']:.3f}"
        print(f"  {i}. [{stype}] {src}  ({score_str})")
        preview = h["text"][:200].replace("\n", " ")
        print(f"     {preview}{'...' if len(h['text']) > 200 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid", "rerank"],
                        default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--compare", action="store_true",
                        help="Show dense vs hybrid vs rerank side-by-side")
    args = parser.parse_args()

    print(f"Query: {args.query}")

    if args.compare:
        show_hits("DENSE only", dense_search(args.query, k=args.k))
        show_hits("HYBRID (dense + BM25 + RRF)",
                  hybrid_search(args.query, k=args.k))
        show_hits("HYBRID + RERANK",
                  rerank_search(args.query, k=args.k))
        return

    if args.mode == "dense":
        show_hits("dense", dense_search(args.query, k=args.k))
    elif args.mode == "sparse":
        show_hits("sparse (BM25)", sparse_search(args.query, k=args.k))
    elif args.mode == "hybrid":
        show_hits("hybrid", hybrid_search(args.query, k=args.k))
    else:
        show_hits("hybrid + rerank", rerank_search(args.query, k=args.k))


if __name__ == "__main__":
    main()
