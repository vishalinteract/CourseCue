#!/usr/bin/env python3
"""
Builds a unified interaction log:
user_id, item_id, timestamp, event_type, value

- Cleans missing IDs / timestamps / duplicates
- Filters bots/outliers (excessive clicks in short spans)
- Applies minimum interaction thresholds (auto-relaxes if empty)
- Temporal train/val/test split (guards against empty/NaN)
- Negative sampling for implicit feedback
"""
import argparse, os, json
import pandas as pd
import numpy as np


def seed_everything(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)


def load_oulad(data_dir: str) -> pd.DataFrame:
    """
    Minimal OULAD adapter.
    Requires: studentVle.csv, vle.csv
    Converts daily click counts into implicit 'view' events with value=sum_click.
    """
    sv = pd.read_csv(os.path.join(data_dir, "studentVle.csv"))
    vle = pd.read_csv(os.path.join(data_dir, "vle.csv"))

    df = sv.merge(vle[["id_site", "activity_type"]],
                  on="id_site", how="left")

    # Timestamp = integer "date" (days since course start). Fill missing with 0.
    df["timestamp"] = df["date"].fillna(0).astype(int)

    df = df.rename(columns={
        "id_student": "user_id",
        "id_site": "item_id",
        "sum_click": "value"
    })
    df["event_type"] = "view"
    df = df[["user_id", "item_id", "timestamp", "event_type", "value"]]
    return df


def clean_and_filter(df: pd.DataFrame, min_user_inter=5, min_item_inter=5) -> pd.DataFrame:
    # Basic cleaning
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp", "event_type"])

    # Simple bot filter: cap per-user-per-day volume at 99th percentile
    per_day = df.groupby(["user_id", "timestamp"]).size()
    cap = per_day.quantile(0.99) if len(per_day) else 0
    if cap > 0:
        df = df.join(per_day.rename("cnt"), on=["user_id", "timestamp"])
        df = df[df["cnt"] <= cap].drop(columns=["cnt"])

    # Apply thresholds
    def _apply_thresholds(frame, u_min, i_min):
        user_cnt = frame.groupby("user_id").size()
        item_cnt = frame.groupby("item_id").size()
        keep_u = user_cnt[user_cnt >= u_min].index
        keep_i = item_cnt[item_cnt >= i_min].index
        return frame[frame["user_id"].isin(keep_u) & frame["item_id"].isin(keep_i)]

    out = _apply_thresholds(df, min_user_inter, min_item_inter)
    if out.empty:
        # Auto-relax thresholds progressively
        for u_min in [3, 2, 1]:
            for i_min in [3, 2, 1]:
                out = _apply_thresholds(df, u_min, i_min)
                if not out.empty:
                    print(f"[warn] Thresholds relaxed to user>={u_min}, item>={i_min} due to sparse data.")
                    return out
        # If still empty, return original cleaned df (no thresholds)
        print("[warn] All thresholds yielded empty data; using cleaned data without thresholds.")
        return df
    return out


def temporal_split(df: pd.DataFrame, val_days=21, test_days=35):
    """Temporal split by integer timestamp. Guards for emptiness."""
    if df.empty:
        raise ValueError("No data available after cleaning/filtering; cannot split.")
    tmin = df["timestamp"].min()
    tmax = df["timestamp"].max()
    if pd.isna(tmin) or pd.isna(tmax):
        raise ValueError("Timestamps contain NaN; check input files.")

    tmin, tmax = int(tmin), int(tmax)
    # If the window is too small, shrink split windows
    span = tmax - tmin + 1
    vd = min(val_days, max(1, span // 4))
    td = min(test_days, max(1, span // 4))

    split1 = tmax - td - vd
    split2 = tmax - td

    train = df[df["timestamp"] < split1]
    val   = df[(df["timestamp"] >= split1) & (df["timestamp"] < split2)]
    test  = df[df["timestamp"] >= split2]

    # If any split is empty, do a simple 60/20/20 chronological split
    if train.empty or val.empty or test.empty:
        q1 = int(np.quantile(df["timestamp"], 0.6))
        q2 = int(np.quantile(df["timestamp"], 0.8))
        train = df[df["timestamp"] < q1]
        val   = df[(df["timestamp"] >= q1) & (df["timestamp"] < q2)]
        test  = df[df["timestamp"] >= q2]

    return train, val, test, dict(tmin=tmin, split1=split1, split2=split2, tmax=tmax)


def negative_sample(user_pos_items: set, all_items: np.ndarray, num_neg: int):
    """Uniform negative sampling that avoids seen items."""
    negs = []
    for _ in range(num_neg):
        it = np.random.choice(all_items)
        tries = 0
        while it in user_pos_items and tries < 10:
            it = np.random.choice(all_items); tries += 1
        negs.append(int(it))
    return negs


def build_train_pairs(df: pd.DataFrame, num_negs_per_pos=4):
    if df.empty:
        return pd.DataFrame(columns=["user_id", "pos_item", "neg_items"])
    all_items = df["item_id"].unique()
    by_u = df.groupby("user_id")["item_id"].apply(set).to_dict()
    users, pos, neg_lists = [], [], []
    for u, items in by_u.items():
        for it in items:
            users.append(int(u))
            pos.append(int(it))
            neg_lists.append(negative_sample(items, all_items, num_negs_per_pos))
    triples = pd.DataFrame({"user_id": users, "pos_item": pos, "neg_items": neg_lists})
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="oulad", choices=["oulad", "assistments", "ednet"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_user_inter", type=int, default=5)
    ap.add_argument("--min_item_inter", type=int, default=5)
    ap.add_argument("--val_days", type=int, default=21)
    ap.add_argument("--test_days", type=int, default=35)
    ap.add_argument("--num_negs", type=int, default=4)
    args = ap.parse_args()

    seed_everything(42)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "oulad":
        df = load_oulad(args.data_dir)
    else:
        raise NotImplementedError("Adapters for ASSISTments/EdNet can be added similarly.")

    df = clean_and_filter(df, args.min_user_inter, args.min_item_inter)
    train, val, test, splits = temporal_split(df, args.val_days, args.test_days)

    def _write(frame: pd.DataFrame, base_name: str):
        base = os.path.join(args.out_dir, base_name)
        try:
            frame.to_parquet(base + ".parquet")
            return base + ".parquet"
        except Exception:
            frame.to_csv(base + ".csv", index=False)
            return base + ".csv"

    _write(train, "train")
    _write(val, "val")
    _write(test, "test")
    _write(df, "interactions")
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    triples = build_train_pairs(train, args.num_negs)
    _write(triples, "train_triples")

    print("Done. Wrote processed data to", args.out_dir)


if __name__ == "__main__":
    main()
