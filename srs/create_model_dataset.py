import pandas as pd
from pathlib import Path

# ============================================================
# Zimbabwe Weather Intelligence
# Model Dataset Preparation
# ============================================================

INPUT_PATH = Path(
    "data/processed/zimbabwe_weather_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/zimbabwe_weather_model.csv"
)


def create_target(df):
    """Create the next-hour temperature target."""

    df["target_temperature_1h"] = (
        df.groupby("city")["temperature_2m"]
        .shift(-1)
    )

    return df


def main():

    print("=" * 60)
    print("Zimbabwe Weather Intelligence")
    print("Model Dataset Preparation")
    print("=" * 60)

    print("\nLoading feature dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["time"] = pd.to_datetime(df["time"])

    print(
        f"Original shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Create forecasting target
    # --------------------------------------------------------

    print("\nCreating next-hour temperature target...")

    df = create_target(df)

    # The final observation for each city has no
    # next-hour target, so remove those rows.
    df = df.dropna(
        subset=["target_temperature_1h"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Verify target
    # --------------------------------------------------------

    print(
        "\nTarget column:"
    )

    print(
        "target_temperature_1h"
    )

    print(
        "\nTarget statistics:"
    )

    print(
        df["target_temperature_1h"].describe()
    )

    print(
        "\nFinal dataset shape:"
    )

    print(
        df.shape
    )

    print(
        "\nMissing values:"
    )

    print(
        df.isnull().sum().sum()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 60)
    print(
        "Model dataset preparation completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()