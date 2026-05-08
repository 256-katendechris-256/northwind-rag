"""
loader.py

Walks the corpus/ folder, loads every file with the appropriate parser,
and chunks it using a strategy chosen to fit the file type.

Output: a list of chunk dicts ready for embedding.
"""

from pathlib import Path
import csv
import re
from pypdf import PdfReader


# -----------------------------------------------------------------------------
# CHUNKING HELPERS
# -----------------------------------------------------------------------------

def recursive_split(text, chunk_size=400, overlap=80):
    """
    Splits text into chunks of roughly `chunk_size` characters.
    Tries to split on the strongest separator first (paragraph),
    then sentence, then word, then hard character cut.
    Adds `overlap` characters from the end of each chunk to the start
    of the next, so facts on chunk boundaries aren't lost.
    """
    # If text is already small enough, return as a single chunk.
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try splitting by paragraph first.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""
    for para in paragraphs:
        # If adding this paragraph would exceed chunk_size,
        # flush the buffer as a chunk first.
        if len(buffer) + len(para) + 2 > chunk_size and buffer:
            chunks.append(buffer.strip())
            # Start the next buffer with overlap from previous chunk's tail.
            buffer = buffer[-overlap:] + " " + para if overlap else para
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

        # If a single paragraph is itself longer than chunk_size,
        # we have to break it harder. Fall back to sentence split.
        if len(buffer) > chunk_size * 1.5:
            sentences = re.split(r"(?<=[.!?])\s+", buffer)
            sub = ""
            for s in sentences:
                if len(sub) + len(s) > chunk_size and sub:
                    chunks.append(sub.strip())
                    sub = sub[-overlap:] + " " + s if overlap else s
                else:
                    sub = (sub + " " + s).strip() if sub else s
            buffer = sub

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


# -----------------------------------------------------------------------------
# PER-FORMAT LOADERS
# Each returns a list of chunk dicts.
# -----------------------------------------------------------------------------

def load_markdown(path: Path, source_type: str):
    """
    Markdown: split on ## headers, then recursive-split within each section.
    Each chunk keeps its section header attached so retrieval has context.
    """
    text = path.read_text(encoding="utf-8")
    # Split by ## headers but keep the header attached to its section.
    parts = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    chunks = []
    chunk_idx = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try to extract the section header for metadata.
        header_match = re.match(r"^##\s+(.+)", part)
        section_header = header_match.group(1).strip() if header_match else "(intro)"
        # Recursive-split inside the section.
        for sub in recursive_split(part, chunk_size=500, overlap=80):
            chunks.append({
                "id": f"{path.name}::{chunk_idx}",
                "text": sub,
                "source": str(path.relative_to(path.parents[1])),
                "source_type": source_type,
                "metadata": {
                    "section_header": section_header,
                    "chunk_index": chunk_idx,
                }
            })
            chunk_idx += 1
    return chunks


def load_text(path: Path, source_type: str):
    """
    Plain text (Notion exports, transcripts).
    No reliable headers — pure recursive split.
    """
    text = path.read_text(encoding="utf-8")
    chunks = []
    for i, sub in enumerate(recursive_split(text, chunk_size=500, overlap=100)):
        chunks.append({
            "id": f"{path.name}::{i}",
            "text": sub,
            "source": str(path.relative_to(path.parents[1])),
            "source_type": source_type,
            "metadata": {"chunk_index": i}
        })
    return chunks


def load_csv(path: Path, source_type: str):
    """
    CSV: one chunk per row. Each row is rendered as a readable sentence
    so embeddings can match on the *meaning* of the row, not commas.
    """
    chunks = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Render the row as a natural-language sentence.
            text_parts = [f"{k.replace('_', ' ')}: {v}" for k, v in row.items()]
            text = ". ".join(text_parts) + "."
            chunks.append({
                "id": f"{path.name}::{i}",
                "text": text,
                "source": str(path.relative_to(path.parents[1])),
                "source_type": source_type,
                "metadata": {
                    "chunk_index": i,
                    "engagement_id": row.get("engagement_id", ""),
                    "client": row.get("client", ""),
                    "year": row.get("year", ""),
                }
            })
    return chunks


def load_pdf(path: Path, source_type: str):
    """
    PDF: extract text page-by-page, concatenate, then recursive-split.
    PDF text extraction is imperfect; we accept that and move on.
    """
    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    full_text = "\n\n".join(pages_text)
    chunks = []
    for i, sub in enumerate(recursive_split(full_text, chunk_size=500, overlap=80)):
        chunks.append({
            "id": f"{path.name}::{i}",
            "text": sub,
            "source": str(path.relative_to(path.parents[1])),
            "source_type": source_type,
            "metadata": {
                "chunk_index": i,
                "page_count": len(reader.pages),
            }
        })
    return chunks


# -----------------------------------------------------------------------------
# DISPATCHER
# -----------------------------------------------------------------------------

# Map folder name to source_type label (used for metadata + filtering later).
FOLDER_SOURCE_TYPES = {
    "handbook": "handbook",
    "notion": "notion",
    "transcripts": "transcript",
    "engagements": "engagement_record",
    "decks": "deck",
}

# Map file extension to loader function.
EXTENSION_LOADERS = {
    ".md": load_markdown,
    ".txt": load_text,
    ".csv": load_csv,
    ".pdf": load_pdf,
}


def load_corpus(corpus_root: str = "corpus"):
    """
    Walk the corpus folder, dispatch each file to the right loader,
    return the full list of chunk dicts.
    """
    root = Path(corpus_root)
    all_chunks = []
    for folder_name, source_type in FOLDER_SOURCE_TYPES.items():
        folder = root / folder_name
        if not folder.exists():
            continue
        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue
            loader = EXTENSION_LOADERS.get(file_path.suffix.lower())
            if loader is None:
                print(f"  skipping {file_path.name} (no loader for {file_path.suffix})")
                continue
            chunks = loader(file_path, source_type)
            print(f"  loaded {file_path.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
    return all_chunks


# -----------------------------------------------------------------------------
# RUN AS A SCRIPT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading corpus...")
    chunks = load_corpus("corpus")
    print(f"\nTotal: {len(chunks)} chunks across {len(set(c['source'] for c in chunks))} files")

    # Show one chunk per source type so we can sanity-check the loader.
    print("\n--- Sample chunk per source type ---")
    seen_types = set()
    for c in chunks:
        if c["source_type"] in seen_types:
            continue
        seen_types.add(c["source_type"])
        print(f"\n[{c['source_type']}] from {c['source']} (id: {c['id']})")
        print(f"metadata: {c['metadata']}")
        print(f"text preview: {c['text'][:200]}{'...' if len(c['text']) > 200 else ''}")
