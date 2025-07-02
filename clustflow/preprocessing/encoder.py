"""
Encoder module for clustflow.

Provides encoding of categorical variables using:
- 'onehot' : scikit-learn OneHotEncoder (default)
- 'ordinal': scikit-learn OrdinalEncoder

If `categorical_cols` is not specified, all object or category dtype columns are auto-detected.

Example
-------
>>> from clustflow.preprocessing.encoder import Encoder
>>> encoder = Encoder(strategy='onehot')
>>> X_encoded = encoder.fit_transform(df, categorical_cols=['gender', 'race'])
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class Encoder:
    """
    Encodes categorical columns in a DataFrame.

    Parameters
    ----------
    strategy : {'onehot', 'ordinal'}, default='onehot'
        Encoding strategy to apply.
    handle_unknown : str, optional
        Only used for one-hot encoding. Passed to sklearn's OneHotEncoder.

    Attributes
    ----------
    encoder : fitted encoder object (OneHotEncoder or OrdinalEncoder)
    columns : list
        List of columns used during fitting.
    """

    def __init__(self, strategy='onehot', handle_unknown='ignore'):
        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.columns = None

    def fit(self, df, categorical_cols=None):
        """
        Learns encoding for specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        categorical_cols : list of str, optional
            Columns to encode. If None, auto-detects object/category dtype columns.

        Returns
        -------
        self : Encoder
            Fitted instance.
        """
        if categorical_cols is None:
            self.columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            print(f"[clustflow] Auto-detected categorical columns: {self.columns}")
        else:
            self.columns = categorical_cols

        if self.strategy == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown)
        elif self.strategy == 'ordinal':
            self.encoder = OrdinalEncoder()
        else:
            raise ValueError("Unsupported encoding strategy")

        self.encoder.fit(df[self.columns])
        return self

    def transform(self, df):
        """
        Transforms categorical columns to encoded numeric format.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        df_copy : pd.DataFrame
            Transformed DataFrame with encoded columns.
        """
        df_copy = df.copy()
        encoded = self.encoder.transform(df_copy[self.columns])

        if self.strategy == 'onehot':
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.columns),
                index=df_copy.index
            )
        else:  # ordinal
            encoded_df = pd.DataFrame(encoded, columns=self.columns, index=df_copy.index)

        df_copy = df_copy.drop(columns=self.columns)
        df_copy = pd.concat([df_copy, encoded_df], axis=1)
        return df_copy

    def fit_transform(self, df, categorical_cols=None):
        """
        Fits and transforms in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        categorical_cols : list of str, optional
            Columns to encode. If None, auto-detects.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        return self.fit(df, categorical_cols).transform(df)
