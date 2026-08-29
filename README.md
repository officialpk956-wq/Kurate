# Kurate — Trust-Ranked Search Demo

## What this is
This project is a localized technical demo supporting a product case study for Kurate. It demonstrates a **trust-ranked search engine** — proving the gap between keyword and semantic retrieval, illustrating the filter-bubble risk of over-weighting trust proximity in algorithms, and providing per-result ranking transparency. **All data on this page is purely synthetic**, generated deterministically via `corpus.py`. No real Kurate systems, users, or data are involved anywhere in this project.

## Architecture
* **`corpus.py`**: Handles synthetic data generation (users, bios, posts, and the trust follow-graph) with built-in validation.
* **`retrieval.py`**: The core ranking engine featuring BM25 keyword search, TF-IDF/LSA semantic search, Reciprocal Rank Fusion (RRF), and a BFS-based trust re-ranker.
* **`app.py`**: The frontend Streamlit UI that ties the simulation together for visual analysis.

**Intentional Constraint:** We explicitly use TF-IDF + Latent Semantic Analysis (LSA) rather than a deep learning/neural embedding model. This is an intentional architectural trade-off designed to operate fully offline and fit comfortably within Streamlit Community Cloud's free-tier memory limits (~1GB RAM) without requiring heavy PyTorch dependencies or API calls.

## Run locally

### Prerequisites
- Python 3.11+
- Git

### Installation
Clone the repository and install the runtime dependencies:
```bash
git clone <repository_url>
cd Kurate
pip install -r requirements.txt
```

### Running the App
Run the Streamlit application:
```bash
streamlit run app.py
```

### Running the Tests
If you want to run the automated test suite, ensure you install the dev dependencies first:
```bash
python -m pytest tests/ -v
```

## Deliberately out of scope for this demo

- **A labeled relevance benchmark (Recall@10, NDCG@10, zero-result precision)** — Not built because a statistically meaningful benchmark needs a query set large and realistic enough that results generalize, and this corpus (150 synthetic posts) is sized to demonstrate specific mechanisms, not to support a real quality metric. A real version would need a labeled query set spanning exact match, synonym, misspelling, person-name, and zero-result cases, evaluated against actual member search logs.
- **Freshness/recency as a ranking signal** — Not built because generating realistic time-decay scores requires modeling a continuously flowing stream of content, which exceeds the scope of this static 150-post synthetic snapshot.
- **Duplicate/near-duplicate content suppression** — Not built because synthetic content is manually controlled to avoid accidental dupes, bypassing the need for SimHash or exact-match deduplication pipelines required in production.
- **Permission/privacy filtering (private, deleted, blocked content)** — Not built because the demo assumes all 150 posts are globally public, avoiding the heavy read-path complexity of filtering out content based on requester ACLs.
- **CI/dependency pinning** — Not built because this is an isolated, single-developer demonstration artifact, avoiding the overhead of Dockerfiles or strict pip-compile locks that a multi-contributor repo requires.
- **Full accessibility audit (screen reader traversal, keyboard navigation, 200% zoom, tap-target sizing, color-independent trust indicators)** — Not built because Streamlit's component model doesn't give reliable control over several of these (focus order, ARIA labeling on custom HTML cards) without a different frontend stack entirely. However, the trust badges DO already avoid being color-only signals (they carry text labels, not just color), which is one accessibility principle this demo does follow.
- **Real semantic retrieval (a neural embedding model) vs. the TF-IDF+LSA approximation actually used** — Restating the memory constraints outlined above, LSA's semantic recall is bounded by this corpus's own vocabulary co-occurrence and will NOT generalize to misspellings, multilingual queries, or genuinely novel concepts the corpus never modeled. This is a mechanism demonstration, not evidence that production semantic search would perform this well.
- **Production system components (mobile client, auth, permissions/blocks, source-of-truth database, background indexing, index versioning, monitoring/alerting, feature flags, staged rollout, rate limiting, abuse detection, data retention, rollback)** — Not built because the assignment explicitly scopes production code as unrequired. The separate written proposal (Mission 02) is where that architecture is specified and reasoned about; this Streamlit app is a proof of concept for the RANKING LOGIC and UX pattern, not a preview of production architecture.

Building the ranking formula surfaced that trust cannot be treated as an independent source of relevance; naively blending a trust score with a relevance score let trust rescue irrelevant results instead of merely reordering relevant ones, which is a concrete, implementation-derived version of the "personalization risk" already named in the product write-up — not a theoretical concern, but a bug this build actually had and fixed.
