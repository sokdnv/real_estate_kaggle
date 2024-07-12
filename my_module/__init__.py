from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['MSSubClass'] = X['MSSubClass'].astype(str)
        X['OverallCond'] = X['OverallCond'].astype(str)
        X['YrSold'] = X['YrSold'].astype(str)
        X['MoSold'] = X['MoSold'].astype(str)

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


class SomeCustomShit(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Total_sqr_footage'] = (X['BsmtFinSF1'] + X['BsmtFinSF2'] +
                                             X['1stFlrSF'] + X['2ndFlrSF'])
        X.drop(['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2'], axis=1, inplace=True)

        X['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) +
                                           X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))

        X.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)

        X['GarageYrBlt'] = 2024 - X['GarageYrBlt']

        X['YearRemodAdd'] = 2024 - X['YearRemodAdd'].astype(
            int)

        X['YrSold'] = 2024 - X['YrSold'].astype(
            int)

        X.drop(['YearBuilt'], axis=1, inplace=True)

        return X
