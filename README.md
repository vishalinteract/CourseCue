# EduRecSys V2 — Education Platform Content Recommendation (IIT Ropar Project)

A best‑practice, end‑to‑end recommendation system for an e‑learning platform that suggests **courses/modules** to each learner using a **two‑stage retrieval + ranking** pipeline. 
Designed to mirror industry patterns while mapping directly to the assignment rubric.

## Why this project?
- Clear **problem framing** and **metrics** (CTR proxy, NDCG@10, HitRate@10, MRR).
- Robust **data pipeline** with temporal split, negative sampling, cleaning, and bot filtering.
- Strong **baselines** (Popularity, Content‑based) + **BPR‑MF** (pairwise ranking) for a compact yet powerful model set.
- **Serving** via a minimal FastAPI demo with candidate generation and ranking stubs.
- **Reproducible** folder structure with config‑driven runs.

## Use Case (Education)
Recommend **courses/modules** that match a learner’s skills and recent activity. Optimizes for **engagement** (clicks, completions), with latency budget \< 100–200 ms for online serving in a cloud environment (demo server is local).

## Datasets
We use **OULAD (Open University Learning Analytics Dataset)** as the primary public dataset and provide adapters for **ASSISTments** and **EdNet**. See `data/README.md` for download links and setup.

## Folder Structure
```
edu-recsys-v2/
├─ data/
│  ├─ raw/                 # place downloaded datasets here
│  └─ processed/           # generated interaction logs / matrices
├─ src/
│  ├─ data_prep.py         # build interaction log, temporal split, negatives
│  ├─ metrics.py           # NDCG, HitRate, MRR, MAP helpers
│  ├─ train_eval.py        # trains/evaluates baselines + BPR‑MF
│  ├─ models/
│  │  ├─ popularity.py
│  │  ├─ tfidf_content.py
│  │  └─ bpr_mf.py
│  └─ serve/
│     ├─ app.py            # FastAPI inference demo
│     ├─ candidate_gen.py  # ANN/NN retrieval (cosine)
│     └─ ranker.py         # re‑ranking stub (blend features)
├─ configs/
│  └─ config.yaml          # hyper‑params & paths
├─ scripts/
│  └─ run_all.sh           # end‑to‑end demo script
├─ tests/
│  └─ test_metrics.py
└─ Presentation & Report/
   ├─ Report_Template_2p.md
   ├─ Slides_Outline.md
   └─ Video_Script.txt
```

## Quickstart
1. **Install**: `pip install -r requirements.txt`
2. **Download data** into `data/raw` (see `data/README.md`).
3. **Process**: `python src/data_prep.py --dataset oulad --data_dir data/raw --out_dir data/processed`
4. **Train**: `python src/train_eval.py --dataset oulad --model bpr --config configs/config.yaml`
5. **Serve** (demo): `uvicorn src.serve.app:app --reload` then hit `http://127.0.0.1:8000/docs`

## Mapping to the Assignment Pages
- P3: problem framing, signals, metrics → README + `configs/config.yaml`
- P5–8: data schema, cleaning, temporal split, negatives → `src/data_prep.py`
- P9–12: baselines & BPR‑MF → `src/models/*`, `src/train_eval.py`
- P13: training loop → `src/models/bpr_mf.py` & `src/train_eval.py`
- P14: serving → `src/serve/*`
- P15–16: metrics & online testing → `src/metrics.py`, README
- P17: MLOps registry/feature store notes included in comments
- P18: monitoring & ethics → notes at end of README and report template

## Notes
- This repo is **teaching‑oriented**; models are intentionally lightweight and fast to run on a laptop.
- Swap in deeper models (SASRec/Transformer4Rec) later if you wish.
