import numpy as np

def _dcg(relevances):
    return np.sum([(rel / np.log2(idx+2)) for idx, rel in enumerate(relevances)])

def ndcg_at_k(y_true, y_pred, k=10):
    """
    y_true: set or list of relevant item_ids
    y_pred: ranked list of item_ids
    """
    y_true = set(y_true)
    y_pred_k = y_pred[:k]
    rels = [1 if it in y_true else 0 for it in y_pred_k]
    dcg = _dcg(rels)
    ideal = _dcg(sorted(rels, reverse=True))
    return float(dcg / ideal) if ideal > 0 else 0.0

def hitrate_at_k(y_true, y_pred, k=10):
    y_true = set(y_true)
    return 1.0 if any(it in y_true for it in y_pred[:k]) else 0.0

def mrr_at_k(y_true, y_pred, k=10):
    y_true = set(y_true)
    for i, it in enumerate(y_pred[:k], 1):
        if it in y_true:
            return 1.0 / i
    return 0.0
