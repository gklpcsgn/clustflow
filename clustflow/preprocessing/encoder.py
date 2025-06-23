import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class Encoder:
    def __init__(self, strategy='onehot', handle_unknown='ignore'):
        """
        strategy: 'onehot' or 'ordinal'
        """
        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.columns = None

    def fit(self, df, categorical_cols=None):
        if categorical_cols is None:
            self.columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            print(f"[clustflow] Auto-detected categorical columns: {self.columns}")
        else:
            self.columns = categorical_cols

        if self.strategy == 'onehot':
            self.encoder = OneHotEncoder(sparse=False, handle_unknown=self.handle_unknown)
        elif self.strategy == 'ordinal':
            self.encoder = OrdinalEncoder()
        else:
            raise ValueError("Unsupported encoding strategy")

        self.encoder.fit(df[self.columns])
        return self

    def transform(self, df):
        df_copy = df.copy()
        encoded = self.encoder.transform(df_copy[self.columns])
        if self.strategy == 'onehot':
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.columns), index=df_copy.index)
        else:
            encoded_df = pd.DataFrame(encoded, columns=self.columns, index=df_copy.index)

        df_copy = df_copy.drop(columns=self.columns)
        df_copy = pd.concat([df_copy, encoded_df], axis=1)
        return df_copy

    def fit_transform(self, df, categorical_cols=None):
        return self.fit(df, categorical_cols).transform(df)
