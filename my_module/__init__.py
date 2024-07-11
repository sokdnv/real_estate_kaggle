from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Преобразуем столбец MSSubClass в строковый тип
        X['MSSubClass'] = X['MSSubClass'].astype(str)
        X['OverallCond'] = X['OverallCond'].astype(str)
        X['YrSold'] = X['YrSold'].astype(str)
        X['MoSold'] = X['MoSold'].astype(str)
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        return X


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_col, target_col):
        self.groupby_col = groupby_col
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.target_col] = X.groupby(self.groupby_col)[self.target_col].transform(
            lambda x: x.fillna(x.median()))
        return X
