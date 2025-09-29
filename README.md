# MemeOrigin
Explain any meme (or slang) with receipts — meaning, origin, first-seen timeline, and sourced examples.

> EN-first MVP • two-person, slow-cook project.

## Why
People see a meme or slang term and wonder: *what does this mean and where did it start?* Existing sites either focus on memes (editorial, slower) or slang (crowdsourced, noisy). MemeOrigin is a **citations-first** explainer with freshness.

## MVP (v0)
- Input: **term** or **image**
- Output: 3–5 sentence explainer + **Origin (first seen + link)** + **2–3 citations** 

## Architecture (planned)
flowchart LR
  A[User term/image] --> B[Retriever: BM25 + embeddings]\
  B --> C[RAG Context: sources (KYM, Reddit, UD, etc.)]\
  A --> D[Image path: OCR + CLIP NN]\
  D --> B\
  C --> E[LLM: grounded summary + citations]\
  E --> F[UI: explanation, origin, timeline, examples]
