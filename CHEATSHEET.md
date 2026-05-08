# Cheat sheet

These answers are built around the Northwind RAG project you shipped. Each
answer references something you actually did — not theory. That's the
difference between candidates who read about RAG and candidates who built it.

When in doubt: keep answers under 90 seconds, finish with one concrete thing
you observed in your project, and offer to go deeper if Seb wants.

---

## Q1 — Walk me through a RAG pipeline end to end

> RAG has five stages. **Loading**: read source documents — for my project
> that meant five formats: Markdown, plain text, PDF, CSV, and Notion exports.
> **Chunking**: split each document into pieces of around 400-500 characters
> with ~80 character overlap, using recursive splitting that prefers paragraph
> breaks over sentence breaks over character cuts. Different formats need
> different strategies — Markdown by header, CSV one chunk per row.
> **Embedding**: turn each chunk into a 384-dimensional vector with a
> sentence-transformers model running locally. **Retrieval**: at query time
> embed the question, find the most similar chunks via vector similarity in
> Chroma, optionally augment with BM25 and rerank. **Generation**: build a
> prompt with the retrieved chunks numbered for citation, send to Llama 3.3
> 70B on Groq with temperature 0, return the answer plus the chunks used as
> citations.

---

## Q2 — Sparse vs dense vs hybrid retrieval

> Dense retrieval uses embeddings — good at meaning, finds "money back" when
> you search "refund." Sparse retrieval is keyword-based, BM25 being the
> standard — good at exact tokens like product codes, names, acronyms,
> identifiers. Hybrid runs both and fuses the result lists with Reciprocal
> Rank Fusion. I default to hybrid because pure dense has a known failure
> mode on exact-match queries — in my project, asking about a specific
> engagement code or partner name worked far better with BM25 in the mix.
> Pure sparse loses to dense whenever the query and document use different
> phrasings of the same idea, which is most of natural language. Hybrid gives
> you both signals; the cost is small, the wins are consistent.

---

## Q3 — How do you chunk documents? Default chunk size and why?

> I default to recursive character splitting that tries paragraph breaks
> first, sentence breaks second, hard cuts last. Around 400-500 characters
> per chunk with 80 character overlap so facts on chunk boundaries survive.
> Then I tune from there based on eval. The trade-off is precision vs context:
> smaller chunks retrieve more precisely but lose surrounding context; larger
> chunks have more context but the embedding averages over more meaning, so
> the relevance signal dilutes. Critically, different file types deserve
> different strategies. In my project, Markdown got split on `##` headers
> first, CSV got one chunk per row formatted as a sentence, transcripts got
> larger overlap to preserve conversational context.

---

## Q4 — Which embedding model and why?

> For prototyping I used `all-MiniLM-L6-v2` — 384 dimensions, runs locally
> via sentence-transformers, no API cost, fast on CPU. For production I'd
> benchmark on the actual data. Strong candidates: OpenAI's
> `text-embedding-3-small` and `-3-large`, Voyage AI's `voyage-3` — which is
> what Anthropic recommends for use with Claude since Anthropic doesn't ship
> their own embedding model — and Cohere's `embed-english-v3.0`. The choice
> depends on eval results plus data residency. If the source data is sensitive
> enough that I can't send it to a third-party API, I'd run an open-weight
> model on my own infrastructure. Whatever I pick, the indexing model and
> the query model must be the same — different embedding models produce
> different "maps" and the coordinates aren't comparable.

---

## Q5 — Why pgvector instead of Pinecone, Weaviate, Qdrant?

> Pgvector is the right default for any team already on Postgres. You add
> vector search without adding new infrastructure to operate, monitor, back
> up, or pay for. The killer feature is filtered vector search via SQL —
> you can combine metadata predicates with similarity in one query.
> "Find chunks similar to this query, but only from 2023 engagements where
> the client is in FinServ" is trivially expressible. Pinecone is a hosted
> SaaS — easy, powerful, vendor lock-in. Qdrant is fast and Rust-based, good
> at scale. Weaviate has strong hybrid search built in but is heavier to
> operate. The principle: don't add infrastructure you don't need. Start
> with pgvector. You should outgrow it before you adopt something else,
> not start heavy. For my project I used Chroma because I wanted to focus
> on RAG concepts rather than Postgres setup; in production at Set Piece I'd
> default to pgvector.

---

## Q6 — How do you evaluate a RAG system? What metrics?

> I split eval into two layers, retrieval and generation, with a labeled eval
> set as the ground truth. **Retrieval metrics** are deterministic and cheap:
> context recall — did we retrieve at least one of the chunks that contains
> the answer — and context precision — what fraction of retrieved chunks were
> relevant. **Generation metrics** require judgment: faithfulness — does the
> answer only contain claims supported by the retrieved context, catches
> hallucination — and answer relevance — does the answer actually address
> the question. I use LLM-as-judge with a structured rubric and JSON output,
> at temperature 0 for determinism. RAGAS is the standard library that wraps
> all four. For my project I built an eval set of 11 questions across easy,
> medium, hard, and adversarial categories, and ran the full eval against
> dense, hybrid, and rerank to compare. Adding hybrid lifted recall
> meaningfully; adding rerank lifted precision and faithfulness further.
> Without measurement, you can't tell whether a change is an improvement or
> a regression.

---

## Q7 — Retrieval failure vs generation failure. How do you debug each?

> Retrieval failure means the right chunks weren't pulled — even a perfect
> LLM can't answer from chunks it doesn't have. Generation failure means
> the right chunks were pulled but the LLM still got the answer wrong —
> hallucinated, ignored context, or missed the question. My debugging
> workflow always starts with retrieval. I look at exactly which chunks
> came back; if the gold chunk isn't in there, generation is irrelevant —
> the fix is upstream. If the gold chunks are there and the answer is still
> wrong, that's a generation failure — fix the system prompt, drop
> temperature, try a different model. In my project I saw a clean example
> of this: a comparison query about two partners pulled chunks from only
> one of them. The LLM correctly hedged, which was the right behavior given
> what it had — that's a retrieval failure, not a generation failure, and
> no prompt tweak would have fixed it.

---

## Q8 — Three failure modes of RAG

> **Out-of-domain queries** — user asks something the corpus has no answer
> to. A weak system invents an answer; a strict system prompt with explicit
> refusal instructions makes the model say "I don't know based on the
> available documents." I tested this in my project with adversarial items
> like "who is the CEO of Microsoft" and the system refused correctly.
>
> **Retrieval miss on hard queries** — the answer is in the corpus but
> retrieval pulls the wrong chunks. Comparison queries are the canonical
> case — I saw this firsthand asking about two partners' roles, where role
> info was in transcript headers that got buried by character-based chunking.
> The fix is structural: extract metadata at ingest time, or query rewriting
> to split a comparison into sub-queries.
>
> **Hallucination on top of irrelevant context** — when retrieved chunks
> aren't relevant and the system prompt is too permissive, the LLM falls
> back on training data and answers anyway. The answer might even be correct,
> but it's not grounded — it's a fluke. I demonstrated this by force-feeding
> irrelevant chunks and comparing strict vs permissive prompts. The system
> prompt is the cheapest place to enforce groundedness.
>
> Two more worth mentioning: **stale data** (the index doesn't reflect doc
> updates — that's Q10), and **context window overflow** (too many chunks
> stuffed into the prompt — modern long-context models tolerate this but
> attention degrades in the middle of long inputs, the "lost in the middle"
> phenomenon).

---

## Q9 — Why isn't semantic search alone enough? What's reranking?

> Semantic search alone has two known weaknesses. First, it's bad at exact
> matches — embeddings average over a chunk's meaning and dilute the signal
> for short, distinctive tokens like product codes, identifiers, and proper
> nouns. That's why hybrid + BM25 helps. Second, even when the right chunks
> are in the top-N, their *ordering* is noisy because embeddings encode
> query and chunk independently and then compare. A reranker — typically a
> cross-encoder — reads query and chunk together and outputs a single
> relevance score per pair. It's slower, but you only run it on the top 20
> candidates from first-pass retrieval, so the cost is bounded. Reranking
> also helps with the "lost in the middle" problem: by giving you a tighter
> top-K of high-precision chunks, you can pass fewer chunks to the LLM,
> which the LLM attends to more reliably. I used `cross-encoder/ms-marco-MiniLM-L-6-v2`
> in my project; for production I'd consider Cohere Rerank 3.5 or BGE
> rerankers depending on data residency.

---

## Q10 — How do you handle freshness?

> Three strategies, in increasing sophistication. **Scheduled re-ingestion**:
> rebuild the index nightly via cron. Simple, brute-force, fine for slowly-
> changing data. **Per-document hashing with incremental updates**: track a
> content hash for every doc; on each ingest cycle, re-embed only the docs
> whose hashes changed and update those entries. This is what I'd reach for
> next in my project — my current `--rebuild` flag is full re-index only.
> **Event-driven**: source systems push change events via webhooks (Notion,
> Google Drive both expose these). The pipeline reacts in near real-time.
> The right choice depends on the SLA and the volume. For a consultancy with
> a few hundred docs, nightly cron with hashing is plenty. For a high-volume
> system, event-driven. Either way, version the index — when you switch
> embedding models, you must rebuild from scratch because old vectors and
> new vectors aren't comparable.

---

## Q11 — A consulting firm has 8 years of methodology in Notion, Drive PDFs, and partners' brains. Walk me through how you'd RAG that.

This is the question Seb is most likely to actually ask. Treat it as a
mini design discussion. ~3 minutes max, structured as: ingestion, chunking,
retrieval, generation, ops.

> **Ingestion**: every source needs a connector with the right parser. Notion
> via API or markdown export, Drive PDFs via a layout-aware parser like
> unstructured.io or Adobe's extraction (basic pypdf for clean PDFs, OCR for
> scanned ones), and the partners' brains via structured interview transcripts
> — that's the highest-effort, highest-value source. I'd build a transcript
> protocol the way Northwind did in my project, with consent, structured
> prompts, and lightweight metadata at the top: name, role, date, topic.
>
> **Chunking**: format-specific. Notion pages by header. PDFs by page or
> section after extraction. CSV-like data one chunk per row. Transcripts
> with larger overlap because partners deliver insights inside conversational
> context. Critically, extract structural metadata at ingest — author, role,
> source type, date — and attach it to every chunk so it's available at
> retrieval time. In my project I missed this for transcript headers and saw
> the cost: a query for partners' roles couldn't be answered well.
>
> **Retrieval**: hybrid by default. Dense for semantic match, BM25 for exact
> identifiers (engagement codes, client names, partner names). RRF to fuse.
> Cross-encoder rerank on the top 20 to get to a clean top 5. Filtered search
> on metadata — "only methodology from the last 2 years," "only engagements
> in FinServ" — using pgvector or Qdrant for production.
>
> **Generation**: tight system prompt with explicit citation, temperature 0,
> refusal instructions for out-of-corpus queries. Return the answer plus the
> chunks used so users can audit. For partners' interviews I'd add a metadata
> note in citations — "Maya Okonkwo, Managing Partner, 2024-02-03" — so the
> answer carries provenance.
>
> **Ops**: per-doc hashing for incremental re-indexing. An eval set built
> from real partner questions, run on every change. Version the index when
> you change embedding models. And — the part most candidates skip — a
> feedback loop. Let users flag wrong answers; review weekly; use the flagged
> items as new eval items.
>
> **What would go wrong**: PDF extraction is imperfect on layout-heavy decks
> (charts, multi-column slides) — sample-check the extracted text early.
> Transcripts have low signal-to-noise — chunks need overlap and headers must
> be metadata, not text. Comparison queries fail with naive retrieval — needs
> query rewriting or agentic retrieval. Stale data — partners' beliefs evolve
> faster than docs are updated; transcripts must be dated and the system
> should know to prefer recent over old. And the social one: partners often
> resist documentation. The methodology engine only works if the firm commits
> to making partners' tacit knowledge legible — which is a process problem,
> not a technology problem.

---

## Bonus — things worth saying if asked open questions

**"What's the hardest part?"**
Eval. Everything else has a clear right answer; eval requires building a
labeled set, accepting LLM-as-judge has biases, and committing to running
it on every change.

**"Where would you start at Set Piece?"**
Knowledge base inventory and a small eval set first. Don't build a system
without a way to measure it. The methodology engine you describe in your
case studies is exactly the shape of system I built — the failure modes I
saw apply directly.

**"What about agents?"**
Agentic retrieval is the next step beyond static RAG — let an LLM plan
multi-step retrieval, reformulate queries, check whether enough info has
been gathered. Useful for hard questions that single-shot retrieval can't
handle, like cross-source comparisons. The cost is more LLM calls and
harder eval. I'd start with strong static RAG plus query rewriting before
reaching for full agents.

**"Have you used LangChain / LlamaIndex?"**
Mention them as awareness. Your project deliberately avoided them because
they hide the moving parts — fine for prototypes, harder to debug in
production. Frameworks like LangChain are worth using when you need their
specific abstractions; otherwise the moving parts you understand are more
valuable than abstractions you don't.

---

## Set Piece-specific framings

The work they describe in their case studies maps directly to what you built:

- "Methodology engine" — encoding 8 years of frameworks → that's a RAG over
  internal knowledge, with the same partners-brains problem we addressed in
  the Northwind transcripts.
- "Intelligence platform" — aggregating client engagement data → CSV/structured
  chunking + filtered retrieval, just like our engagements CSV.
- "Automated reporting" — pulling live data into client-ready decks → RAG +
  tool use + templates, an obvious next step from where we got to.

You can credibly say: "The shape of the system I built — multi-format
ingestion, hybrid retrieval over a consultancy's mixed knowledge sources,
metadata-rich citations — is essentially a methodology-engine prototype.
The failure modes I observed on it are exactly the ones I'd expect Set
Piece to be solving for paying clients."

That sentence puts you on Seb's wavelength immediately.

---

## Final reminders for the conversation

1. **Don't recite.** Answer the question asked, briefly, with a concrete
   example from your project. Stop. Let Seb dig.

2. **Bridge to your project early.** "I built X. On X I saw Y. The fix
   is Z." That structure beats abstract knowledge every time.

3. **It's fine to say "I don't know."** Senior engineers say it constantly.
   "I haven't used Cohere Rerank — what's been your experience?" is a great
   answer. "I don't have hands-on with multi-tenant vector DBs but here's
   how I'd think about it" is also great. Faking confidence is the failure
   mode.

4. **Your eval table is your strongest artifact.** Have it ready to describe
   if asked. The mode-to-mode comparison is the kind of disciplined
   measurement that gets you hired.

5. **End strong.** When asked if you have questions for them, ask Seb about
   his hardest current problem at Set Piece. Then listen. People hire for
   curiosity as much as competence.

You've got this.
