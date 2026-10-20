def get_result(set, pred, actual):
    results = set.copy()

    results['predicted_returns'] = pred
    results['actual_return'] = actual

    best_daily = (
        results
        .sort_values('predicted_returns', ascending=False)
        .groupby(level=0)
        .head(1)
        .sort_index()
    )

    return best_daily