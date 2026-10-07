import pandas as pd
from pathlib import Path

# ============================================================
# Zimbabwe Weather Intelligence
# Feature Engineering
# ============================================================

INPUT_PATH = Path(
    "data/processed/zimbabwe_weather_2015_2025.csv"
)

OUTPUT_PATH = Path(
    "data/processed/zimbabwe_weather_features.csv"
)


def create_features(df):
    """Create time-based and lag features."""

    # --------------------------------------------------------
    # Time-based features
    # --------------------------------------------------------

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear

    # --------------------------------------------------------
    # Cyclical encoding
    # Helps models understand that December and January
    # are close to each other in the yearly cycle.
    # --------------------------------------------------------

    import numpy as np

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # --------------------------------------------------------
    # Lag features
    # Previous-hour and previous-day temperature
    # --------------------------------------------------------

    df["temperature_lag_1h"] = (
        df.groupby("city")["temperature_2m"]
        .shift(1)
    )

    df["temperature_lag_24h"] = (
        df.groupby("city")["temperature_2m"]
        .shift(24)
    )

    # --------------------------------------------------------
    # Rolling temperature features
    # --------------------------------------------------------

    df["temperature_rolling_24h"] = (
        df.groupby("city")["temperature_2m"]
        .transform(
            lambda x: x.rolling(24).mean()
        )
    )

    # --------------------------------------------------------
    # Previous precipitation
    # --------------------------------------------------------

    df["precipitation_lag_1h"] = (
        df.groupby("city")["precipitation"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Remove rows created with incomplete lag/rolling values
    # --------------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    return df


def main():

    print("=" * 60)
    print("Zimbabwe Weather Intelligence")
    print("Feature Engineering")
    print("=" * 60)

    print("\nLoading processed dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["time"] = pd.to_datetime(df["time"])

    print(
        f"Original shape: {df.shape}"
    )

    print("\nCreating features...")

    df = create_features(df)

    print(
        f"Feature dataset shape: {df.shape}"
    )

    print("\nNew columns:")

    original_columns = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
        "city"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in original_columns
    ]

    for column in feature_columns:
        print(f"✓ {column}")

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nSaving feature dataset...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 60)
    print("Feature engineering completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()