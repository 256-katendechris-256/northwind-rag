import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()
groq_client = Groq()  # reads GROQ_API_KEY from env

# Local embedding model. First run downloads ~80MB, then it's cached.
# 384-dimensional embeddings, very fast, good enough for most RAG.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- 1 & 2: LOAD and CHUNK ---
def load_and_chunk(folder, chunk_size=300):
    chunks = []
    for path in Path(folder).glob("*.md"):
        text = path.read_text()
        for i in range(0, len(text), chunk_size):
            chunks.append({
                "text": text[i:i+chunk_size],
                "source": path.name,
                "id": f"{path.name}-{i}"
            })
    return chunks

# --- 3: EMBED (now local) ---
def embed(texts):
    # encode() returns numpy arrays; Chroma wants lists
    return embedder.encode(texts).tolist()

# --- 4: STORE ---
def build_vector_store(chunks):
    chroma = chromadb.Client()
    try: chroma.delete_collection("knowledge_base")
    except: pass
    coll = chroma.create_collection("knowledge_base")
    coll.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return coll

# --- 5: QUERY (now via Groq) ---
def ask(question, coll, k=3):
    q_emb = embed([question])[0]
    results = coll.query(query_embeddings=[q_emb], n_results=k)
    context_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": sources,
        "context_used": context_chunks,
    }

if __name__ == "__main__":
    print("Loading docs...")
    chunks = load_and_chunk("docs")
    print(f"  {len(chunks)} chunks created")

    print("Building vector store...")
    coll = build_vector_store(chunks)

    questions = [
        "How much does the discovery phase cost?",
        "Can I get my money back after we start?",
        "What does the diagnostic framework measure?",
        "Who is the CEO of Microsoft?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = ask(q, coll)
        print(f"A: {result['answer']}")
        print(f"   sources: {result['sources']}")
