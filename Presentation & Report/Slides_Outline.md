# Slide Deck Outline

1. **Title** — EduRecSys V2 | Vishal Shukla
2. **Problem & Objectives** — what/why, KPIs, latency
3. **Use Case (E‑Learning)** — signals & item metadata
4. **Dataset (OULAD)** — schema & example rows
5. **Data Pipeline** — cleaning, temporal split, negatives
6. **Baselines** — Popularity, Content‑based (TF‑IDF)
7. **Model** — BPR‑MF (diagram: users/items → embeddings → pairwise loss)
8. **Training & Metrics** — NDCG@10, HitRate@10, MRR; early stopping
9. **Serving** — two‑stage (retrieve+rank), FastAPI demo
10. **Results** — table/plot (Pop vs TF‑IDF vs BPR‑MF)
11. **Monitoring & Ethics** — drift, bias, privacy
12. **Conclusion & Future Work** — roadmap

*Tip:* Include a brief live demo GIF/video of the API returning recommendations.
