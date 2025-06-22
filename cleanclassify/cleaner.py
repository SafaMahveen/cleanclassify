import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(df, target_column, max_categories=50, max_rows=2500):
    """
    Cleans the DataFrame by handling missing values, encoding categoricals,
    and scaling numeric features. Also downsamples if data is too large.

    Parameters:
    - df: pandas DataFrame
    - target_column: str, name of the target column
    - max_categories: int, maximum number of unique values to allow for one-hot encoding
    - max_rows: int, maximum number of rows to keep for cleaning

    Returns:
    - X: cleaned feature DataFrame
    - y: target Series
    """

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    df = df.copy()
    df = df[df[target_column].notnull()]

    # Downsample if needed
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        print(f" Downsampled to {max_rows} rows for performance.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Fill missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].mean())

    # One-hot encode categorical variables with limited categories
    dropped_cols = []
    for col in X.select_dtypes(include='object').columns:
        if X[col].nunique() > max_categories:
            X = X.drop(columns=[col])
            dropped_cols.append(col)
        else:
            dummies = pd.get_dummies(X[col], drop_first=True, prefix=col)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    if dropped_cols:
        print(f" Dropped high-cardinality columns: {dropped_cols}")

    # Scale numeric features
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = StandardScaler().fit_transform(X[num_cols])

    return X, y

