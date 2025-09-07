import pandas as pd

class PopularityRecommender:
    def __init__(self):
        self.ranking = None

    def fit(self, interactions: pd.DataFrame):
        # count item frequency as implicit popularity
        counts = interactions.groupby("item_id").size().sort_values(ascending=False)
        self.ranking = list(counts.index.values)

    def recommend(self, user_id, top_k=10, exclude_items=None):
        exclude = set(exclude_items or [])
        out = [it for it in self.ranking if it not in exclude]
        return out[:top_k]
