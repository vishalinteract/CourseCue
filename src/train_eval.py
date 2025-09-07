#!/usr/bin/env python3
import argparse, os, json
import pandas as pd
import numpy as np
from pathlib import Path

from models.popularity import PopularityRecommender
from models.tfidf_content import TFIDFContentRecommender
from metrics import ndcg_at_k, hitrate_at_k, mrr_at_k


def _read(base_path_without_ext: str) -> pd.DataFrame:
    """Read a dataset saved as Parquet (preferred) or CSV (fallback)."""
    pq = base_path_without_ext + ".parquet"
    cs = base_path_without_ext + ".csv"
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(cs):
        return pd.read_csv(cs)
    raise FileNotFoundError(f"Neither {pq} nor {cs} found.")


def evaluate(users, ground_truth, recs, k=10):
    ndcgs, hits, mrrs = [], [], []
    for u in users:
        gt = ground_truth.get(u, set())
        pred = recs.get(u, [])
        ndcgs.append(ndcg_at_k(gt, pred, k))
        hits.append(hitrate_at_k(gt, pred, k))
        mrrs.append(mrr_at_k(gt, pred, k))
    return dict(NDCG=float(np.mean(ndcgs)),
                HitRate=float(np.mean(hits)),
                MRR=float(np.mean(mrrs)))


def build_ground_truth(df: pd.DataFrame):
    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="oulad")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--model", default="bpr", choices=["pop", "tfidf", "bpr"])
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # Load splits (CSV/Parquet)
    train = _read(os.path.join(args.processed_dir, "train"))
    val   = _read(os.path.join(args.processed_dir, "val"))
    test  = _read(os.path.join(args.processed_dir, "test"))
    # Triples only needed for BPR
    triples = None
    if args.model == "bpr":
        triples = _read(os.path.join(args.processed_dir, "train_triples"))

    # Popularity is always available (also acts as cold-start fallback)
    pop = PopularityRecommender()
    pop.fit(train)

    users = sorted(test["user_id"].unique())
    gt = build_ground_truth(test)

    if args.model == "pop":
        recs = {u: pop.recommend(u, top_k=args.k) for u in users}
        print(evaluate(users, gt, recs, k=args.k))
        return

    if args.model == "tfidf":
        # Minimal item metadata (demo): you can plug real titles/descriptions if you have them
        items = pd.DataFrame({"item_id": sorted(train["item_id"].unique())})
        items["title"] = items["item_id"].astype(str).radd("Activity ")
        items["description"] = "learning module"
        tfidf = TFIDFContentRecommender(max_features=5000, ngram_range=(1, 2))
        tfidf.fit(items, text_cols=["title", "description"])

        history = train.groupby("user_id")["item_id"].apply(list).to_dict()

        recs = {}
        for u in users:
            seen = history.get(u, [])
            r = tfidf.recommend(seen, top_k=args.k, exclude_items=seen)
            if not r:  # cold-start or no metadata coverage → fallback to popularity
                r = pop.recommend(u, top_k=args.k, exclude_items=seen)
            recs[u] = r

        print(evaluate(users, gt, recs, k=args.k))
        return

    if args.model == "bpr":
        # Lazy import torch model only if requested
        from models.bpr_mf import BPRMF, train_bpr
        # Remap IDs to contiguous indices
        all_users = sorted(pd.concat([train["user_id"], val["user_id"], test["user_id"]]).unique())
        all_items = sorted(pd.concat([train["item_id"], val["item_id"], test["item_id"]]).unique())
        u2idx = {u: i for i, u in enumerate(all_users)}
        i2idx = {it: j for j, it in enumerate(all_items)}
        inv_item = {v: k for k, v in i2idx.items()}

        # Remap triples
        triples_idx = pd.DataFrame({
            "user_id": triples["user_id"].map(u2idx),
            "pos_item": triples["pos_item"].map(i2idx),
            "neg_items": triples["neg_items"].apply(lambda xs: [i2idx.get(x, -1) for x in xs]),
        })

        num_users, num_items = len(u2idx), len(i2idx)
        model = BPRMF(num_users, num_items, embedding_dim=args.embedding_dim)
        model = train_bpr(model,
                          triples_idx,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr=args.lr)

        # Build recs (exclude seen)
        history = train.groupby("user_id")["item_id"].apply(set).to_dict()
        recs = {}
        import torch
        with torch.no_grad():
            for u in users:
                uid = u2idx[u]
                scores = model.score_all(torch.tensor([uid], dtype=torch.long)).numpy().ravel()
                seen_idx = {i2idx[it] for it in history.get(u, set()) if it in i2idx}
                ranked = [inv_item[idx] for idx in np.argsort(-scores) if idx not in seen_idx][:args.k]
                recs[u] = ranked

        print(evaluate(users, gt, recs, k=args.k))
        return


if __name__ == "__main__":
    main()
