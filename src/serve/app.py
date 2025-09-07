# src/serve/app.py
# EduRecSys V2 – FastAPI backend with STRICT Item-kNN + Filtered Personalization + Advanced Trending
import os, time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles


APP_TITLE = "CourseCue API"
DATA_DIR = "data/processed"
TRAIN_PQ = os.path.join(DATA_DIR, "train.parquet")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
CATALOG_CSV = os.path.join(DATA_DIR, "item_catalog.csv")

# ---------- In-memory state ----------
DF: Optional[pd.DataFrame] = None               # interactions
CATALOG_DF: Optional[pd.DataFrame] = None       # items meta
ITEM_TITLE: Dict[int, str] = {}
ITEM_TOPIC: Dict[int, str] = {}
ITEM_DIFF:  Dict[int, str] = {}
ITEM_POP: Dict[int, int] = {}
POPULAR: List[int] = []
HISTORY: Dict[int, List[int]] = {}
ITEMKNN: Dict[int, List[Tuple[int, float]]] = {}

# ---------- API schemas ----------
class RecRequest(BaseModel):
    user_id: int
    seen_items: Optional[List[int]] = None

class RecItem(BaseModel):
    id: int
    title: Optional[str] = None
    reason: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None

class RecResponse(BaseModel):
    user_id: int
    items: List[RecItem]
    mode_used: str

# ---------- App ----------
app = FastAPI(title=APP_TITLE, version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
# Serve /static for docs assets
app.mount("/static", StaticFiles(directory="assets"), name="static")
app = FastAPI(
    title=APP_TITLE,
    version="1.1.0",
    swagger_ui_parameters={"faviconUrl": "/static/coursecue_logo.png"},
)

# ---------- IO ----------
def _read_interactions() -> pd.DataFrame:
    if os.path.exists(TRAIN_PQ):
        df = pd.read_parquet(TRAIN_PQ)
    elif os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV)
    else:
        raise FileNotFoundError("Missing data/processed/train.parquet or train.csv")
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(int)
    else:
        df = df.sort_values(["user_id", "item_id"]).reset_index(drop=True)
        df["timestamp"] = np.arange(len(df))
    return df

def _read_catalog() -> pd.DataFrame:
    if not os.path.exists(CATALOG_CSV):
        return pd.DataFrame(columns=["item_id","title","topic","difficulty","description"])
    cat = pd.read_csv(CATALOG_CSV)
    if "item_id" in cat.columns:
        cat["item_id"] = cat["item_id"].astype(int)
    for c in ["title","topic","difficulty","description"]:
        if c in cat.columns:
            cat[c] = cat[c].astype(str).fillna("")
    return cat

# ---------- Title synthesis ----------
def _looks_like_placeholder(title: str, item_id: int) -> bool:
    t = (title or "").strip().lower()
    return (t == "" or t == f"course {item_id}".lower() or t == f"item {item_id}".lower())

def _synth_title(topic: str, diff: str) -> str:
    topic = (topic or "Learning").strip()
    diff  = (diff  or "").strip().lower()
    if diff in ("beginner", "intro", "introductory"):
        return f"Intro to {topic}"
    if diff in ("intermediate",):
        return f"{topic} Essentials"
    if diff in ("advanced",):
        return f"Advanced {topic}"
    return f"{topic} Fundamentals"

def _build_meta(cat: pd.DataFrame):
    titles, topics, diffs = {}, {}, {}
    has_title = "title" in cat.columns
    for r in cat.itertuples(index=False):
        iid = int(getattr(r, "item_id"))
        topic = str(getattr(r, "topic", "") or "")
        diff  = str(getattr(r, "difficulty", "") or "")
        raw_title = str(getattr(r, "title", "") if has_title else "") or ""
        titles[iid] = (_synth_title(topic, diff) if _looks_like_placeholder(raw_title, iid) else raw_title)
        topics[iid] = topic
        diffs[iid]  = diff
    return titles, topics, diffs

# ---------- Builders ----------
def build_popularity(df: pd.DataFrame):
    global ITEM_POP, POPULAR
    counts = df["item_id"].value_counts()
    ITEM_POP = counts.to_dict()
    POPULAR = counts.index.astype(int).tolist()

def build_history(df: pd.DataFrame, max_per_user: int = 200):
    global HISTORY
    df2 = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    HISTORY = {}
    for uid, g in df2.groupby("user_id", sort=False):
        seq = list(dict.fromkeys(g["item_id"].astype(int).tolist()))
        HISTORY[int(uid)] = seq[-max_per_user:]

def build_itemknn(topk_neighbors: int = 80, max_window: int = 50):
    global ITEMKNN
    t0 = time.time()
    pop = pd.Series(ITEM_POP).astype(float).to_dict()
    co: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for _, seq in HISTORY.items():
        n = len(seq)
        if n <= 1:
            continue
        for i in range(n):
            a = seq[i]
            for j in range(i + 1, min(n, i + 1 + max_window)):
                b = seq[j]
                if a == b: continue
                co[a][b] += 1.0
                co[b][a] += 1.0
    knn: Dict[int, List[Tuple[int, float]]] = {}
    for a, nbrs in co.items():
        pa = max(pop.get(a, 1.0), 1.0)
        scored = []
        for b, c in nbrs.items():
            pb = max(pop.get(b, 1.0), 1.0)
            s = c / np.sqrt(pa * pb)
            scored.append((int(b), float(s)))
        scored.sort(key=lambda x: x[1], reverse=True)
        knn[int(a)] = scored[:topk_neighbors]
    ITEMKNN = knn
    print(f"[serve] item-kNN built: items_with_neighbors={len(ITEMKNN)} in {time.time()-t0:.2f}s")

def load_all(build_knn: bool = True):
    global DF, CATALOG_DF, ITEM_TITLE, ITEM_TOPIC, ITEM_DIFF
    DF = _read_interactions()
    CATALOG_DF = _read_catalog()
    ITEM_TITLE, ITEM_TOPIC, ITEM_DIFF = _build_meta(CATALOG_DF)
    build_popularity(DF)
    build_history(DF)
    if build_knn: build_itemknn()

# ---------- Helpers ----------
def title_for(i: int) -> str:
    return ITEM_TITLE.get(int(i), f"Learning Path {int(i)}")

def topic_for(i: int) -> Optional[str]:
    return ITEM_TOPIC.get(int(i))

def diff_for(i: int) -> Optional[str]:
    return ITEM_DIFF.get(int(i))

def user_recent_items(user_id: int, window_days: Optional[int], max_items: int = 50) -> List[int]:
    """
    Get the user's most recent unique items, optionally restricted to a time window.
    """
    if DF is None or DF.empty:
        return []
    g = DF[DF["user_id"] == int(user_id)].sort_values("timestamp")
    if window_days and window_days > 0:
        tmax = int(DF["timestamp"].max())
        cutoff = tmax - int(window_days) * 86400
        g = g[g["timestamp"] >= cutoff]
    if g.empty:
        return []
    seq = list(dict.fromkeys(g["item_id"].astype(int).tolist()))
    return seq[-max_items:]

def trending_items(
    k: int,
    window_days: Optional[int] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    exclude_seen_ids: Optional[List[int]] = None,
) -> List[int]:
    if DF is None or DF.empty:
        return []
    df = DF
    if window_days is not None and window_days > 0:
        tmax = int(df["timestamp"].max())
        cutoff = tmax - int(window_days) * 86400
        df = df[df["timestamp"] >= cutoff]
    if (topic or difficulty) and CATALOG_DF is not None and not CATALOG_DF.empty:
        mask = pd.Series(True, index=CATALOG_DF.index)
        if topic:      mask &= (CATALOG_DF.get("topic", "").astype(str) == str(topic))
        if difficulty: mask &= (CATALOG_DF.get("difficulty", "").astype(str) == str(difficulty))
        seg_items = set(CATALOG_DF.loc[mask, "item_id"].astype(int).tolist())
        if seg_items:
            df = df[df["item_id"].isin(seg_items)]
    if df.empty:
        base = [i for i in POPULAR if i not in set(exclude_seen_ids or [])]
        return base[:k]
    counts = df["item_id"].value_counts()
    if exclude_seen_ids:
        counts = counts.drop(labels=[int(s) for s in exclude_seen_ids if int(s) in counts.index], errors="ignore")
    ranked = counts.index.astype(int).tolist()
    return ranked[:k]

# ---------- Endpoints ----------
@app.on_event("startup")
def startup():
    print("[serve] Startup… loading")
    load_all(build_knn=True)
    print("[serve] Startup complete")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "users": len(HISTORY),
        "items_indexed": len(POPULAR),
        "items_with_neighbors": len(ITEMKNN),
        "data_source": TRAIN_PQ if os.path.exists(TRAIN_PQ) else (TRAIN_CSV if os.path.exists(TRAIN_CSV) else None),
    }

@app.get("/debug/catalog_facets")
def catalog_facets():
    topics = sorted(CATALOG_DF["topic"].dropna().astype(str).unique().tolist()) if (CATALOG_DF is not None and "topic" in CATALOG_DF.columns) else []
    diffs  = sorted(CATALOG_DF["difficulty"].dropna().astype(str).unique().tolist()) if (CATALOG_DF is not None and "difficulty" in CATALOG_DF.columns) else []
    return {"topics": topics, "difficulties": diffs}

@app.get("/debug/stats")
def debug_stats():
    return {
        "users": len(HISTORY),
        "items": len(ITEM_POP),
        "items_with_neighbors": len(ITEMKNN),
        "sample_user_ids": list(HISTORY.keys())[:12],
        "popular_item_ids": POPULAR[:12],
    }

@app.get("/debug/user_history")
def debug_user_history(user_id: int, window_days: Optional[int] = None):
    hist = user_recent_items(user_id, window_days) if window_days else HISTORY.get(int(user_id), [])
    return {"user_id": int(user_id), "len": len(hist), "history_tail": hist[-10:]}

@app.get("/debug/neighbors")
def debug_neighbors(item_id: int, n: int = 10):
    nbs = ITEMKNN.get(int(item_id), [])[:n]
    return {"item_id": int(item_id), "has_neighbors": bool(nbs), "neighbors": nbs}

@app.post("/reload_index")
def reload_index(fast: bool = Query(True, description="fast=True: reload popularity+catalog only; False: rebuild kNN too")):
    load_all(build_knn=not fast)
    return {
        "ok": True,
        "fast": fast,
        "users": len(HISTORY),
        "popular_items": len(POPULAR),
        "items_with_neighbors": len(ITEMKNN),
    }

@app.post("/recommendations", response_model=RecResponse)
def recommendations(
    req: RecRequest,
    k: int = Query(10, ge=1, le=50),
    mode: str = Query("itemknn"),   # itemknn | pop
    strict: bool = Query(True),     # STRICT for itemknn (no silent fallback)
    # Filters (apply to both modes)
    pop_window_days: Optional[int] = Query(None, ge=1, le=365),
    topic: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    exclude_seen: bool = Query(False),
):
    # Determine seen/seeds
    seen = list(dict.fromkeys([int(x) for x in (req.seen_items or [])]))
    if not seen:
        # respect time window for personalization when provided
        recent = user_recent_items(req.user_id, pop_window_days) if pop_window_days else HISTORY.get(int(req.user_id), [])
        seen = recent[-5:]

    if mode == "pop":
        ids = trending_items(
            k=k, window_days=pop_window_days, topic=topic,
            difficulty=difficulty, exclude_seen_ids=(seen if exclude_seen else []),
        )
        reason_bits = []
        if pop_window_days: reason_bits.append(f"last {pop_window_days}d")
        if topic:           reason_bits.append(f"topic={topic}")
        if difficulty:      reason_bits.append(f"difficulty={difficulty}")
        if exclude_seen:    reason_bits.append("excluding seen")
        reason = "Trending " + ("(" + ", ".join(reason_bits) + ")" if reason_bits else "(all-time)")
        items = [RecItem(id=i, title=title_for(i), reason=reason, topic=topic_for(i), difficulty=diff_for(i)) for i in ids]
        return RecResponse(user_id=req.user_id, items=items, mode_used="pop")

    # -------- Personalized (Item-kNN) with filters --------
    # Score neighbors
    scores: Dict[int, float] = defaultdict(float)
    best_seed: Dict[int, Tuple[int, float]] = {}
    for s in seen:
        for nb, sc in ITEMKNN.get(int(s), []):
            if exclude_seen and nb in seen: 
                continue
            scores[nb] += sc
            if nb not in best_seed or sc > best_seed[nb][1]:
                best_seed[nb] = (int(s), sc)

    if not scores:
        if strict:
            return RecResponse(user_id=req.user_id, items=[], mode_used="itemknn(no-signal)")
        # fallback to trending with filters
        ids_fb = trending_items(
            k=k, window_days=pop_window_days, topic=topic, difficulty=difficulty,
            exclude_seen_ids=(seen if exclude_seen else []),
        )
        items_fb = [RecItem(id=i, title=title_for(i), reason="Popular (fallback)", topic=topic_for(i), difficulty=diff_for(i)) for i in ids_fb]
        return RecResponse(user_id=req.user_id, items=items_fb, mode_used="pop(fallback)")

    # Sort by score and then apply filters
    ranked = sorted(scores.items(), key=lambda x: (x[1], ITEM_POP.get(x[0], 0)), reverse=True)
    out_ids: List[int] = []
    for iid, _ in ranked:
        if exclude_seen and iid in seen: 
            continue
        if topic and (topic_for(iid) or "") != topic:
            continue
        if difficulty and (diff_for(iid) or "") != difficulty:
            continue
        out_ids.append(int(iid))
        if len(out_ids) >= k: break

    if not out_ids:
        if strict:
            return RecResponse(user_id=req.user_id, items=[], mode_used="itemknn(filtered-no-match)")
        # fallback to filtered trending
        ids_fb = trending_items(
            k=k, window_days=pop_window_days, topic=topic, difficulty=difficulty,
            exclude_seen_ids=(seen if exclude_seen else []),
        )
        items_fb = [RecItem(id=i, title=title_for(i), reason="Popular (fallback)", topic=topic_for(i), difficulty=diff_for(i)) for i in ids_fb]
        return RecResponse(user_id=req.user_id, items=items_fb, mode_used="pop(fallback)")

    out = []
    for i in out_ids:
        seed = best_seed.get(i, (None, None))[0]
        reason = f"Because you studied {title_for(seed)}" if seed is not None else "Similar learners also studied"
        out.append(RecItem(id=i, title=title_for(i), reason=reason, topic=topic_for(i), difficulty=diff_for(i)))
    return RecResponse(user_id=req.user_id, items=out, mode_used="itemknn(filtered)")
