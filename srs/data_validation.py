import pandas as pd
from pathlib import Path

# ============================================================
# Zimbabwe Weather Intelligence
# Raw Weather Data Validation
# ============================================================

CITIES = [
    "harare",
    "bulawayo",
    "mutare",
    "gweru"
]

DATA_FOLDER = Path("data/raw")


def validate_city_data(city):
    """Validate the raw weather dataset for one city."""

    file_path = DATA_FOLDER / f"{city}_weather_test.csv"

    print("\n" + "=" * 60)
    print(f"Validating: {city.title()}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    # Convert time column
    df["time"] = pd.to_datetime(df["time"])

    print(f"\nDataset shape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------
    print("\nMissing values:")
    print(df.isnull().sum())

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # --------------------------------------------------------
    # Duplicate timestamps
    # --------------------------------------------------------
    print("\nDuplicate timestamps:")
    print(df["time"].duplicated().sum())

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------
    print("\nDate range:")
    print("Start:", df["time"].min())
    print("End:  ", df["time"].max())

    # --------------------------------------------------------
    # Number of unique timestamps
    # --------------------------------------------------------
    print("\nUnique timestamps:")
    print(df["time"].nunique())

    # --------------------------------------------------------
    # Weather variable ranges
    # --------------------------------------------------------
    weather_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover"
    ]

    print("\nWeather variable ranges:")

    for column in weather_columns:
        print(
            f"{column}: "
            f"min={df[column].min()}, "
            f"max={df[column].max()}"
        )


def main():

    print("=" * 60)
    print("Zimbabwe Weather Intelligence")
    print("Raw Weather Data Validation")
    print("=" * 60)

    for city in CITIES:

        try:
            validate_city_data(city)

        except FileNotFoundError:
            print(f"\nFile not found for {city.title()}")

        except Exception as error:
            print(
                f"\nUnexpected error validating "
                f"{city.title()}: {error}"
            )

    print("\n" + "=" * 60)
    print("Raw data validation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()