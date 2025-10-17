import streamlit as st
import pandas as pd
import numpy as np
from model import ScooterModel, TARGETS, load_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("Scooter Model Training, Optimization & Evaluation")

st.header("1. Model Characteristics")
"""
The model/pipeline consists of 3 phases:

1. Data loading and rudimentary preprocessing:
    - Loading the CSV data.
    - Setting the appropriate index.
    - Handling "wrongly" formatted values and data type conversions.
2. Data Preprocessing and Feature Engineering:
    - Correction of erroneous data (e.g., negative `co` values, Kelvin to Celsius).
    - Encoding of categorical variables.
    - Imputation of missing values.
    - Creation of new features from existing ones (e.g., adding day of the week and month).
    - Scaling and normalization of features.
3. Model Training:

    I use a multi-output regression model (Linear Regression) to predict the scooter usage types.
    I also clip the values to be non-negative since scooter counts cannot be negative.
    More interesting models could be used however data cleanup took longer than expected so I don't have the time to experiment with them.
"""

# Load and split data
st.header("2. Training and Optimization")
with st.spinner("Loading and splitting data..."):
    df = load_csv("scooter.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Train model
with st.spinner("Training model..."):
    model = ScooterModel(targets=TARGETS)
    model.fit(df_train)


# Evaluate on test set
def get_stats(y_true, y_pred):
    stats = {}
    stats = model.score(
        df_test,
        scores={
            "R2": r2_score,
            "MAE": mean_absolute_error,
            "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        },
    )
    return stats


def display_stats(stats, title):
    st.subheader(title)
    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.style.format("{:.3f}")
    st.table(df_stats)


with st.spinner("Evaluating on test set..."):
    preds_test = model.predict(df_test)
    stats_test = get_stats(df_test[TARGETS], preds_test)
    display_stats(stats_test, "Test Set Evaluation")

# Option to upload a new file for evaluation
st.header("3. Evaluate on Your Own Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file with the same columns as the training data:"
)
if uploaded_file is not None:
    user_df = load_csv(uploaded_file)
    if all(t in user_df.columns for t in TARGETS):
        with st.spinner("Evaluating uploaded data..."):
            preds_user = model.predict(user_df, keep_data=True)
            stats_user = get_stats(user_df[TARGETS], preds_user)
            display_stats(stats_user, "Uploaded Data Evaluation")
            st.subheader("Predictions (first 20 rows)")
            st.dataframe(preds_user.head(20))
            st.download_button(
                "Download Predictions as CSV",
                preds_user.to_csv().encode("utf-8"),
                file_name="predictions.csv",
            )
    else:
        st.error(f"Uploaded data must contain the target columns: {TARGETS}")
