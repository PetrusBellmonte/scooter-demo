import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Scooter Data Analysis")

# Load the CSV file
df = pd.read_csv("scooter.csv")
df = df.set_index(df.columns[0])

FEATURES = [
    "date",
    "hour",
    "holiday",
    "weather",
    "temp",
    "temp_felt",
    "humidity",
    "windspeed",
    "co",
]
TARGETS = ["type_a", "type_b", "type_ab"]


# View Format
st.subheader("Scooter Data")
st.dataframe(df.head(10))

# Connection between type_a, type_b and type_ab
with st.container(border=True):
    f"""
    `type_ab` seems to be the sum of `type_a` and `type_b` (Correct for {(df["type_a"].fillna(0) + df["type_b"].fillna(0) == df["type_ab"].fillna(0)).mean() * 100:.2f}% of rows).
    The only lines where this does not hold are shown below:
    """

    st.write(
        df[df["type_a"].fillna(0) + df["type_b"].fillna(0) != df["type_ab"].fillna(0)][
            ["date", "hour", "type_a", "type_b", "type_ab"]
        ]
    )

    """
    This is probably incorrect data!

    Any prediction task can just predict `type_a` and `type_b` and then sum them to get `type_ab`.
    """
"`co` is wrongly formatted with a comma instead of a dot as decimal separator."

df["co"] = pd.to_numeric(df["co"].str.replace(",", "."), errors="coerce")

# Display summary statistics
st.subheader("Scooter Data Summary")
st.write(df.describe())

"""
 Observations:
 - All feature columns have missing values, so imputation will be necessary.
 - `type a` seems to have some outliers.
 - `co` is impossible negative values.
 - The maximum temperatures `temp` and `temp_felt` are almost 300. That's unrealistic... (More on that in the end)
"""

# Set all negative 'co' values to (np.nan)
df.loc[df["co"] < 0, "co"] = np.nan

# Data availability
with st.container(border=True):
    "**Missing Features?**"
    # Count NaNs in each feature column
    nan_counts = df[FEATURES].isna().sum(axis=1)

    # Plot histogram of NaN counts
    fig = px.bar(
        x=nan_counts.index,
        y=nan_counts.values,
        labels={"x": "ID", "y": "Number of NaNs"},
        title="Missing Values per row",
    )
    st.plotly_chart(fig)
    """
    While there are some missing values throughout the dataset, between id 10111 and 10300 basically no weather measurements were taken.
    This has to be excluded from training.
    """
with st.container(border=True):
    "**Missing Targets?**"
    # Create a DataFrame indicating missing values for each target
    if st.checkbox("Show stacked bar plot of missing targets per row?"):
        missing_targets = df[TARGETS].isna().astype(int)
        missing_targets["ID"] = missing_targets.index

        # Melt for stacked bar plot
        melted = missing_targets.melt(
            id_vars="ID", value_vars=TARGETS, var_name="Target", value_name="Missing"
        )
        # Only plot rows with at least one missing target
        # melted = melted[melted['Missing'] > 0]
        fig = px.bar(
            melted,
            x="ID",
            y="Missing",
            color="Target",
            labels={"ID": "ID", "Missing": "Missing (1=missing)", "Target": "Target"},
            title="Missing Targets per Row (Stacked by Target)",
        )
        st.plotly_chart(fig)
    """
    Between ids 200 and 5xx the target value are completely missing. This data cannot be used for training.
    """
    # Absolute and relative percentage of missing values for each target
    total_rows = len(df)
    for target in TARGETS:
        missing_count = df[target].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        st.write(f"- {target}: {missing_count} missing ({missing_pct:.2f}%)")
    """

    `type_a` is missing 10% of its values. 
    That's substantial, but the target columns also don't now a `0` (even though the value should occur).
    The `0` values seem to be `nan`/empty instead.
    Before filling in missing values with `0`, I want to check wether `type_a` is probably missing values.
    As I cannot distinguish between `0` and *missing*, I have to take whole chunks of `nan` as an indicator of missing values.
    """
    window_size = st.slider(
        "Check for how many consecutive missing values should be considered a gap?",
        min_value=3,
        max_value=50,
        value=24,
        step=1,
    )
    # Find occurrences where consecutive 'type_a' are missing but 'type_b' is not
    mask_type_a_missing = df["type_a"].isna()
    mask_type_b_present = df["type_b"].notna()
    combined_mask = mask_type_a_missing & mask_type_b_present

    # Rolling window to find sequences of consecutive True values
    rolling_sum = combined_mask.rolling(window=window_size).sum()

    # Count the number of times there are 25 consecutive missing 'type_a' (and not missing 'type_b')
    occurrences = (rolling_sum == window_size).sum()
    st.write(
        f"There are {occurrences} occurrences where {window_size} consecutive 'type_a' values are missing, but 'type_b' is present."
    )

    # Show one example of 5 consecutive missing 'type_a' but present 'type_b'
    if occurrences > 0:
        idx = rolling_sum[rolling_sum == window_size].index[0]
        example_indices = range(max(0, idx - window_size + 1 - 10), idx + 1 + 10)
        st.write(
            f"Example of {window_size} consecutive missing 'type_a' values (but 'type_b' present):"
        )
        st.write(
            df.loc[example_indices, ["date", "hour", "type_a", "type_b", "type_ab"]]
        )

    """
    The longest gap in `type_a` data is 9 hours and the gaps appear to mostly occur at night.
    This is plausible given the overall lower `type_a` usage and lower usage at night.

    In conclusion, I will fill missing target values with `0`.
    """


# Feature/Target Distribution Explorer
st.subheader("Feature Distribution Explorer")
all_columns = FEATURES + TARGETS
all_columns.remove("date")  # Remove 'date' from selection as it's not plottable
selected_col = st.selectbox(
    "Select a feature or target to plot its distribution", all_columns
)

with st.container(border=True):
    st.subheader(f"Distribution of {selected_col}")
    col_data = df[selected_col].dropna()
    if pd.api.types.is_numeric_dtype(col_data):
        fig = px.histogram(col_data, nbins=30, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)
    else:
        fig = px.bar(
            col_data.value_counts().reset_index(),
            x="index",
            y=selected_col,
            title=f"Distribution of {selected_col}",
        )
        st.plotly_chart(fig)

"""
`weather` and `holiday` are categorical features and will be one-hot encoded.

"""
st.subheader("Temperature Correction Explorer")
"""
Some temperature values are above 100°C, which is unrealistic. 
Assuming the temperature midnight 15.3.2025 stayed roughly the same 280.19 and 6.1 should be about the same temperature.
The 273 difference is about right for a Kelvin to Celsius conversion.

To check this assumption, I have a look at the distribution of the corrected temperatures.
The faulty month were in the deep winter, so I compare the newly calculated temperatures to December and March.
"""
correction = st.slider(
    "Subtract this value from temperatures over 100",
    min_value=260,
    max_value=300,
    value=273,
    step=1,
)
# Only use data from December and March for uncorrected data
df["month"] = pd.to_datetime(df["date"]).dt.month
# December = 12, March = 3
mask_dec_mar = df["month"].isin([12, 3])

# Create corrected temperature column only for values > 100
df["temp_corrected"] = df["temp"].where(df["temp"] <= 100, df["temp"] - correction)

temp_uncorrected = df.loc[(df["temp"] <= 100) & mask_dec_mar, "temp"]
temp_corrected = df.loc[df["temp"] > 100, "temp_corrected"]

# Combine into a DataFrame for plotting
fig = px.histogram(
    x=[*temp_uncorrected, *temp_corrected],
    color=(["Uncorrected (≤100, Dec/Mar)"] * len(temp_uncorrected))
    + (["Corrected (>100)"] * len(temp_corrected)),
    nbins=30,
    opacity=0.7,
    barmode="overlay",
    title="Temperature Distribution: Corrected vs Uncorrected (Dec/Mar only for uncorrected)",
)
st.plotly_chart(fig)
"The distributions sufficiently overlap to assume Kelvin as the erroneous unit."
