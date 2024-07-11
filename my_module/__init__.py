from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Преобразуем столбец MSSubClass в строковый тип
        X['MSSubClass'] = X['MSSubClass'].astype(str)
        return X