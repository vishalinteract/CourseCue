from src.metrics import ndcg_at_k, hitrate_at_k, mrr_at_k

def test_metrics_basic():
    y_true = {3, 5}
    y_pred = [1, 3, 7, 5]
    assert 0 < ndcg_at_k(y_true, y_pred, 3) <= 1
    assert hitrate_at_k(y_true, y_pred, 3) == 1.0
    assert mrr_at_k(y_true, y_pred, 4) > 0
