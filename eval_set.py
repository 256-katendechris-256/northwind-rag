"""
eval_set.py

The labeled test set for RAG evaluation. Each item has:
  - question: what to ask
  - expected_sources: list of source files that contain the answer.
      (Used for context recall: did retrieval find at least one of these?)
  - key_facts: list of facts the answer must contain. Used for human-readable
      sanity check and as input to the faithfulness/relevance judge.
  - difficulty: easy | medium | hard | adversarial
  - notes: why this question is in the set

Adversarial = no answer in the corpus. The correct behavior is to refuse.
"""

EVAL_SET = [
    # ---------- EASY ----------
    {
        "question": "How long is the Discovery phase?",
        "expected_sources": [
            "handbook/01_engagement_lifecycle.md",
            "notion/pricing_reference.txt",
        ],
        "key_facts": ["2 weeks"],
        "difficulty": "easy",
        "notes": "Single fact, repeated in two source types.",
    },
    {
        "question": "How much does the Discovery phase cost?",
        "expected_sources": [
            "handbook/01_engagement_lifecycle.md",
            "notion/pricing_reference.txt",
        ],
        "key_facts": ["£25,000", "25000"],
        "difficulty": "easy",
        "notes": "Simple price lookup. Either '£25,000' or '25000' counts.",
    },
    {
        "question": "What is the refund policy?",
        "expected_sources": [
            "handbook/01_engagement_lifecycle.md",
        ],
        "key_facts": ["14 days", "non-refundable", "credit"],
        "difficulty": "easy",
        "notes": "Should mention the 14-day window AND the post-kickoff credit.",
    },

    # ---------- MEDIUM ----------
    {
        "question": "What is the Below-3 rule and where did it come from?",
        "expected_sources": [
            "notion/diagnostic_framework.txt",
            "handbook/01_engagement_lifecycle.md",
        ],
        "key_facts": ["below 3", "Helios", "Team Health", "Transformation"],
        "difficulty": "medium",
        "notes": "Requires two pieces: the rule itself + the Helios origin.",
    },
    {
        "question": "What four dimensions does the Diagnostic measure?",
        "expected_sources": [
            "handbook/01_engagement_lifecycle.md",
            "notion/diagnostic_framework.txt",
        ],
        "key_facts": [
            "Strategy Clarity",
            "Operating Velocity",
            "Team Health",
            "Customer Alignment",
        ],
        "difficulty": "medium",
        "notes": "Must enumerate all four. Tests completeness.",
    },
    {
        "question": "Why did the Helios engagement fail?",
        "expected_sources": [
            "engagements/past_engagements.csv",
            "notion/diagnostic_framework.txt",
            "handbook/01_engagement_lifecycle.md",
        ],
        "key_facts": ["Team Health", "scored 2", "proceeded anyway", "8 weeks"],
        "difficulty": "medium",
        "notes": "Multi-source synthesis. Triggers the Below-3 rule story.",
    },

    # ---------- HARD ----------
    {
        "question": "When should we walk away from an engagement, according to James?",
        "expected_sources": [
            "transcripts/2024-01-15_james_aldridge.txt",
        ],
        "key_facts": ["CEO", "hide", "deflection", "rescheduling", "CFO"],
        "difficulty": "hard",
        "notes": "Buried in conversational transcript. Tests transcript retrieval.",
    },
    {
        "question": "How does Embed pricing get decided, and what changed in late 2022?",
        "expected_sources": [
            "transcripts/2024-02-03_maya_okonkwo.txt",
        ],
        "key_facts": [
            "milestone",
            "weekly billing",
            "capability",
            "alignment",
        ],
        "difficulty": "hard",
        "notes": "Tests synthesis within a single transcript across distant sections.",
    },
    {
        "question": "Which engagement was the largest by fee, and what made it notable?",
        "expected_sources": [
            "engagements/past_engagements.csv",
        ],
        "key_facts": ["Tessera", "485", "FinServ", "capability rebuild"],
        "difficulty": "medium",
        "notes": "Pure CSV retrieval. Tests structured data handling.",
    },

    # ---------- ADVERSARIAL ----------
    {
        "question": "Who is the CEO of Microsoft?",
        "expected_sources": [],   # nothing in corpus
        "key_facts": ["I don't know"],
        "difficulty": "adversarial",
        "notes": "Out-of-domain. Correct behavior is explicit refusal.",
    },
    {
        "question": "What is Northwind's policy on remote-first engagements for the Embed stage?",
        "expected_sources": [],   # corpus says onsite is required for Transformation; nothing on Embed remote
        "key_facts": ["I don't know"],
        "difficulty": "adversarial",
        "notes": (
            "Sneaky: corpus mentions remote-only for Transformation but not Embed. "
            "The system should refuse rather than extrapolate."
        ),
    },
]
