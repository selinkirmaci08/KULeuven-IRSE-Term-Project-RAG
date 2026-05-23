# IRSE Project 2026 — RAG System

## Setup

Install all dependencies before running anything:

```bash
pip install -r requirements.txt
```

---

## Part 1 — Cooking Recipes RAG

### IR Metrics
Run `part1/main.py` to evaluate the retrieval system across different distance metrics, k values, and vocabulary fields:
- Drawback of TF-IDF
- Reasoning betweeen recipes
- Good retrieval and LM feeding

```bash
python part1/main.py
```

---

## Part 2 — ACL Anthology RAG

Run `part2/main.py` to execute the full Part 2 pipeline, which includes:
- Sentence transformer baseline
- TF-IDF baseline
- Query rewriting
- HyDE 
- Chunking 
- Cross-encoder reranking
- Prompt engineering

```bash
python part2/main.py
```

> **Note**: Requires a GPU. 
> **Note**: There are some cache results for ease of computation.
> **Note**: I have tested this code in colab.