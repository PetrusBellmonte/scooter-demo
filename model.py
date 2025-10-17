from typing import Callable
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
import argparse

TARGETS = ["type_a", "type_b"]


def convert_and_add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def adjust_temp(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["temp"] > 100, "temp"] = df.loc[df["temp"] > 100, "temp"] - 273
    return df


def adjust_co(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["co"] < 0, "co"] = np.nan
    return df


def preprocess_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by cleaning and transforming specific columns.
    Steps performed:
    - Converts the 'co' column to numeric, replacing commas with dots and coercing errors to NaN.
    - Sets negative values in TARGETS columns to NA.
    - Filters out rows with missing values in TARGETS and FEATURES columns (if training).
    - Fills remaining NA values in TARGETS columns with 0.
    - Adjusts 'temp' values greater than 100 by subtracting 273 (assumed conversion from Kelvin to Celsius).
    Args:
        df (pd.DataFrame): Input DataFrame containing raw data.
        training (bool): Flag indicating if the preprocessing is for training data.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with cleaned and transformed columns.
    """
    FEATURES = [
        "hour",
        "holiday",
        "weather",
        "temp",
        "temp_felt",
        "humidity",
        "windspeed",
        "co",
    ]

    df["co"] = pd.to_numeric(df["co"].str.replace(",", "."), errors="coerce")
    df[TARGETS] = df[TARGETS].where(df[TARGETS] >= 0, pd.NA)
    if training:
        mask = df[TARGETS].notna().all(axis=1) & df[FEATURES].notna().all(axis=1)
        df = df[mask]
    df.loc[:, TARGETS].fillna(0, inplace=True)
    return df


class FeatureImputer(TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        self.imputer = SimpleImputer(strategy="mean")

    def fit(self, X, y=None) -> "FeatureImputer":
        self.imputer.fit(X[self.features])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.features] = self.imputer.transform(X[self.features])
        return X


class MultiLinearRegression(BaseEstimator):
    def __init__(self, targets=TARGETS):
        self.targets = targets
        self.models = {target: LinearRegression() for target in targets}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "MultiLinearRegression":
        for target in self.targets:
            self.models[target].fit(X, y[target])
        self._fitted = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return self._fitted

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Usage is always positive soo use max 0
        preds = {
            target: np.maximum(0, model.predict(X))
            for target, model in self.models.items()
        }
        return pd.DataFrame(preds)


class ScooterModel:
    def __init__(self, targets=TARGETS):
        self.targets = targets

        # Select and convert columns to regression friendly features
        feature_transformer = ColumnTransformer(
            [
                (
                    "hour_ohe",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ["hour"],
                ),
                (
                    "dow_ohe",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ["day_of_week"],
                ),
                (
                    "month_ohe",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ["month"],
                ),
                (
                    "weather_ohe",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ["weather"],
                ),
                (
                    "holiday_ohe",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ["holiday"],
                ),
                (
                    "other_features",
                    "passthrough",
                    ["temp", "temp_felt", "humidity", "windspeed", "co"],
                ),
            ]
        )

        self.pipeline = Pipeline(
            [
                (
                    "add_date_features",
                    FunctionTransformer(convert_and_add_date_features),
                ),
                ("correct_temperatures", FunctionTransformer(adjust_temp)),
                # ('correct_co', FunctionTransformer(adjust_co)),
                (
                    "impute_features",
                    FeatureImputer(
                        features=["temp", "temp_felt", "humidity", "windspeed", "co"]
                    ),
                ),
                ("transform_features", feature_transformer),
                ("standardize", StandardScaler()),
                ("regression", MultiLinearRegression()),
            ]
        )

    def fit(self, df: pd.DataFrame) -> "ScooterModel":
        df = df.copy()
        df = preprocess_data(df, training=True)
        y = df[TARGETS]
        self.pipeline.fit(df, y)
        return self

    def _predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.predict(df)

    def predict(self, df: pd.DataFrame, keep_data: bool = False) -> pd.DataFrame:
        data = df.copy()
        data = preprocess_data(data, training=False)
        preds = self._predict(data)
        if keep_data:
            preds.index = df.index
            joined_data = df.rename(
                columns={c: f"true_{c}" if c in self.targets else c for c in df.columns}
            )
            preds = joined_data.join(preds, how="right")
        return preds

    def score(
        self, df: pd.DataFrame, scores: list[Callable] | dict[str:Callable]
    ) -> dict[str, dict[str, float]]:
        df = df.copy()
        df = preprocess_data(df, training=True)
        y_true = df[self.targets]
        y_pred = self._predict(df)
        results = {}
        for target in self.targets:
            results[target] = (
                {
                    score_name: score(y_true[target], y_pred[target])
                    for score_name, score in scores.items()
                }
                if isinstance(scores, dict)
                else {
                    score.__name__: score(y_true[target], y_pred[target])
                    for score in scores
                }
            )
        return results


def load_csv(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file, delimiter=",")
    if df.columns[0] != "date":
        df.set_index(df.columns[0], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Scooter Model CLI")
    parser.add_argument(
        "--train", type=str, default="scooter.csv", help="Path to training dataset CSV"
    )
    parser.add_argument(
        "--eval", type=str, default=None, help="Optional evaluation dataset CSV"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    args = parser.parse_args()

    # Load data
    df = load_csv(args.train)

    df_train, df_test = train_test_split(df, test_size=args.test_split, random_state=42)

    # Create model
    model = ScooterModel(targets=TARGETS)

    # Run pipeline (fit on train)
    print("Fitting pipeline...")
    model.fit(df_train)

    # Predict and score
    def print_scores(data: pd.DataFrame, type: str = "Eval"):
        stats = model.score(data, scores={"R2": r2_score})
        print(f"{type} R2 scores:", stats)

    print_scores(df_train, type="Train")
    print_scores(df_test, type="Test")

    if args.eval:
        df_eval = load_csv(args.eval)
        print_scores(df_eval, type="Eval")


if __name__ == "__main__":
    main()
