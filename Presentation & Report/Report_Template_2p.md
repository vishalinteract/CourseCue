# Project Report (2 Pages)

**Title:** EduRecSys V2 — Education Platform Content Recommendation  
**Student:** Vishal Shukla | IIT Ropar Minor in AI | Track: Recommender Systems

## 1. Problem Statement & Objectives
Recommend courses/modules to each learner to maximize engagement (clicks, completions) while respecting latency (<100–200ms) and privacy constraints. Objectives: improve CTR/NDCG@10 vs. popularity baseline; maintain guardrails (latency, policy compliance).

## 2. Dataset & Signals
Primary: OULAD (Open University Learning Analytics Dataset). Signals: daily VLE clicks → implicit views (value=sum_click), enrolment/withdrawal dates for filtering; optionally assessments as downstream outcomes.
Unified schema: *(user_id, item_id, timestamp, event_type, value).*

## 3. Methodology
- **Cleaning:** drop bad IDs & duplicates; cap per‑user per‑day outliers; thresholds: ≥5 interactions per user/item.
- **Temporal split:** train on past, validate/test on future (avoid leakage).
- **Negatives:** sample 4 unseen items per positive.
- **Baselines:** Popularity; TF‑IDF content‑based (title/description).
- **Model:** BPR‑MF (pairwise ranking with user/item embeddings).
- **Serving (demo):** two‑stage (candidate gen + rank) with FastAPI stub.

## 4. Implementation
- `src/data_prep.py`: builds interaction log & splits from OULAD.
- `src/train_eval.py`: trains/evaluates baselines & BPR‑MF; computes NDCG@k, HitRate@k, MRR.
- `src/serve/app.py`: minimal API for recommendations (demo).

## 5. Results & Observations
- Report **NDCG@10, HitRate@10, MRR** on validation/test for Pop vs. BPR‑MF. 
- Typical expectation: BPR‑MF > TF‑IDF > Popularity on ranking metrics.
- Note any cold‑start and popularity bias; discuss mitigations (content features, exploration).

## 6. Conclusion & Future Work
A compact, production‑shaped pipeline that improves ranking quality with pairwise learning. Next steps: SASRec for sequences, knowledge‑graph features, real‑time features (recency), diversity‑aware re‑ranking, A/B testing plan.

## 7. Tools
Python, Pandas, scikit‑learn, PyTorch, FastAPI, Annoy; configs via YAML; progress tracked with tqdm logs.
