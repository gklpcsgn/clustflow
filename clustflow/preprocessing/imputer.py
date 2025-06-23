
import pandas as pd

class Imputer:
    def __init__(self, strategy: dict):
        """
        strategy: dict where keys are column names, values are:
        - 'mean' / 'median'
        - ('group_median', group_col)
        """
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, df):
        for col, method in self.strategy.items():
            if method == 'mean':
                self.fill_values_[col] = df[col].mean()
            elif method == 'median':
                self.fill_values_[col] = df[col].median()
            elif isinstance(method, tuple) and method[0] == 'group_median':
                group_col = method[1]
                self.fill_values_[col] = df.groupby(group_col)[col].median()
            else:
                raise ValueError(f"Unsupported imputation strategy: {method}")
        return self

    def transform(self, df):
        df_copy = df.copy()
        for col, method in self.strategy.items():
            if method in ['mean', 'median']:
                df_copy[col] = df_copy[col].fillna(self.fill_values_[col])
            elif isinstance(method, tuple) and method[0] == 'group_median':
                group_col = method[1]
                df_copy[col] = df_copy[col].fillna(
                    df_copy[group_col].map(self.fill_values_[col])
                )
        return df_copy

    def fit_transform(self, df):
        return self.fit(df).transform(df)
