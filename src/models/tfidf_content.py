from typing import List, Dict, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFContentRecommender:
    def __init__(self, max_features=5000, ngram_range=(1,2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.item_vectors = None
        self.item_ids = None

    def fit(self, items_df: pd.DataFrame, text_cols: List[str]):
        items_df = items_df.copy()
        items_df["__text__"] = items_df[text_cols].fillna("").agg(" ".join, axis=1)
        self.item_ids = items_df["item_id"].tolist()
        self.item_vectors = self.vectorizer.fit_transform(items_df["__text__"].tolist())

    def user_profile(self, user_history_items: List[int]):
        if not user_history_items:
            return None
        idxs = [self.item_ids.index(it) for it in user_history_items if it in self.item_ids]
        if not idxs:
            return None
        return self.item_vectors[idxs].mean(axis=0)

    def recommend(self, user_history_items: List[int], top_k=10, exclude_items=None):
        profile = self.user_profile(user_history_items)
        if profile is None:
            # cold start fallback: return arbitrary popular items (handled upstream)
            return []
        sims = cosine_similarity(profile, self.item_vectors).ravel()
        # exclude seen items
        exclude = set(exclude_items or [])
        scored = [(self.item_ids[i], sims[i]) for i in range(len(self.item_ids)) if self.item_ids[i] not in exclude]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:top_k]]
