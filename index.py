"""
index.py

Phase 3 — embedding and storage.

What this does:
  1. Loads chunks from corpus/ via loader.py
  2. Embeds each chunk locally with sentence-transformers
  3. Stores everything in a persistent Chroma collection
  4. Exposes search(query, k) for retrieval
  5. CLI mode: run it to build the index and try sample queries

Run modes:
  python index.py              # build (if needed) + run sample queries
  python index.py --rebuild    # wipe and rebuild from scratch
  python index.py --query "what is the refund policy?"
"""

import argparse
import sys
from sentence_transformers import SentenceTransformer
import chromadb

from loader import load_corpus

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, runs locally, free
COLLECTION_NAME = "northwind_knowledge"
PERSIST_DIR = "chroma_db"                   # on-disk store for the index


# -----------------------------------------------------------------------------
# EMBEDDING
# -----------------------------------------------------------------------------

# Lazy-load the model. First call downloads it (~80MB), then cached on disk.
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed(texts):
    """Take a list of strings, return a list of embedding vectors (lists of floats)."""
    model = get_embedder()
    # encode() returns numpy arrays; Chroma wants Python lists.
    return model.encode(texts, show_progress_bar=False).tolist()


# -----------------------------------------------------------------------------
# INDEX BUILD
# -----------------------------------------------------------------------------

def get_chroma_client():
    """Persistent Chroma client — stores data in PERSIST_DIR on disk."""
    return chromadb.PersistentClient(path=PERSIST_DIR)


def get_or_create_collection(client):
    """Returns the Northwind collection, creating it if it doesn't exist."""
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_index(rebuild=False):
    """
    Build the vector index from the corpus.
    If rebuild=True, wipes the existing collection first.
    If rebuild=False and the collection already has the right number of
    items, this is a no-op.
    """
    client = get_chroma_client()

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Wiped existing collection.")
        except Exception:
            pass

    coll = get_or_create_collection(client)

    # Load chunks first so we know how many we expect.
    print("\nLoading corpus...")
    chunks = load_corpus("corpus")
    print(f"  loaded {len(chunks)} chunks")

    # Skip rebuild if the existing index is already up to date.
    if coll.count() == len(chunks) and not rebuild:
        print(f"Index already has {coll.count()} chunks — skipping rebuild.")
        print("  (use --rebuild to force re-embed)")
        return coll

    # If counts differ, wipe and rebuild for safety.
    if coll.count() > 0:
        client.delete_collection(COLLECTION_NAME)
        coll = get_or_create_collection(client)

    print("\nEmbedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)
    print(f"  embedded {len(embeddings)} chunks "
          f"({len(embeddings[0])} dimensions each)")

    print("Inserting into Chroma...")
    # Chroma can't accept lists in metadata, so we store only scalar metadata.
    metadatas = []
    for c in chunks:
        m = {
            "source": c["source"],
            "source_type": c["source_type"],
        }
        # Add per-chunk metadata fields (already scalars from loader.py).
        for k, v in c["metadata"].items():
            m[k] = v
        metadatas.append(m)

    coll.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"  collection now has {coll.count()} items")
    return coll


# -----------------------------------------------------------------------------
# SEARCH
# -----------------------------------------------------------------------------

def search(query: str, k: int = 5):
    """
    Embed the query, retrieve the top-k closest chunks from Chroma,
    and return them as a list of dicts.
    """
    client = get_chroma_client()
    coll = get_or_create_collection(client)

    if coll.count() == 0:
        raise RuntimeError(
            "Index is empty. Run `python index.py --rebuild` first."
        )

    q_emb = embed([query])[0]
    raw = coll.query(
        query_embeddings=[q_emb],
        n_results=k,
    )

    # Chroma returns parallel lists. Zip them into per-chunk dicts.
    hits = []
    for i in range(len(raw["ids"][0])):
        hits.append({
            "id": raw["ids"][0][i],
            "text": raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": raw["distances"][0][i],   # smaller = more similar
        })
    return hits


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def show_hits(query, hits):
    print(f"\nQuery: {query}")
    print(f"Top {len(hits)} hits:")
    for i, h in enumerate(hits, 1):
        src = h["metadata"].get("source", "?")
        stype = h["metadata"].get("source_type", "?")
        # Lower distance = closer match in Chroma's default cosine setup.
        print(f"\n  {i}. [{stype}] {src}  (distance: {h['distance']:.3f})")
        preview = h["text"][:240].replace("\n", " ")
        print(f"     {preview}{'...' if len(h['text']) > 240 else ''}")


SAMPLE_QUERIES = [
    # "What's the refund policy?",
    # "Why did the Helios engagement fail?",
    # "How does Maya think about pricing differently from James?",
    # "What's the Below-3 rule?",
    # "Tell me about Tessera Capital.",
    "what are the signs that CEO is running away from work?",
    "What are maya and james' roles in the company?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe and rebuild the index from scratch")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query instead of the sample set")
    parser.add_argument("-k", type=int, default=5,
                        help="Number of chunks to retrieve (default 5)")
    args = parser.parse_args()

    build_index(rebuild=args.rebuild)

    if args.query:
        show_hits(args.query, search(args.query, k=args.k))
    else:
        print("\n=== Sample queries ===")
        for q in SAMPLE_QUERIES:
            show_hits(q, search(q, k=3))


if __name__ == "__main__":
    main()
