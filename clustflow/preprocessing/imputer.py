"""
Imputer module for clustflow.

Provides per-column missing value imputation using strategies such as:
- 'mean': fill with overall column mean
- 'median': fill with overall column median
- ('group_median', group_col): fill with group-wise median based on another column

Example
-------
>>> from clustflow.preprocessing.imputer import Imputer
>>> imputer = Imputer(strategy={
...     'age': 'mean',
...     'income': 'median',
...     'education_level': ('group_median', 'region')
... })
>>> df_filled = imputer.fit_transform(df)
"""

import pandas as pd

class Imputer:
    """
    Flexible imputer for handling missing values column-wise.

    Parameters
    ----------
    strategy : dict
        Dictionary specifying imputation strategy per column.
        Values can be 'mean', 'median', or ('group_median', group_col).

    Attributes
    ----------
    fill_values_ : dict
        Stores computed fill values per column after fitting.
    """

    def __init__(self, strategy: dict):
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, df):
        """
        Learns imputation values from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to fit on.

        Returns
        -------
        self : Imputer
            Fitted instance.
        """
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
        """
        Applies learned imputations to a new DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform.

        Returns
        -------
        df_copy : pd.DataFrame
            Transformed DataFrame with missing values filled.
        """
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
        """
        Fits and transforms in a single step.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            Imputed data.
        """
        return self.fit(df).transform(df)
