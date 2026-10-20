
def train_test_split(X, y, df):
    dates = df.index.unique().sort_values()

    train_end = int(len(dates) * 0.6)
    val_end = int(len(dates) * 0.8)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    X_train = X.loc[X.index.isin(train_dates)]
    y_train = y.loc[y.index.isin(train_dates)]

    X_val = X.loc[X.index.isin(val_dates)]
    y_val = y.loc[y.index.isin(val_dates)]

    X_test = X.loc[X.index.isin(test_dates)]
    y_test = y.loc[y.index.isin(test_dates)]

    return X_train, y_train, X_val, y_val, X_test, y_test
