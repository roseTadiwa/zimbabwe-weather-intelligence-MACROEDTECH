import pandas as pd
from pathlib import Path

# ============================================================
# Zimbabwe Weather Intelligence
# Weather Data Preprocessing
# ============================================================

CITIES = [
    "harare",
    "bulawayo",
    "mutare",
    "gweru"
]

RAW_FOLDER = Path("data/raw")
PROCESSED_FOLDER = Path("data/processed")


def load_city_data(city):
    """Load the raw weather data for one city."""

    file_path = RAW_FOLDER / f"{city}_weather_test.csv"

    df = pd.read_csv(file_path)

    # Convert time to datetime
    df["time"] = pd.to_datetime(df["time"])

    return df


def clean_city_data(df, city):
    """Clean and standardise weather data for one city."""

    # Ensure observations are ordered chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # Remove exact duplicate rows if any exist
    df = df.drop_duplicates()

    # Remove duplicate timestamps if any exist
    df = df.drop_duplicates(
        subset=["time"],
        keep="first"
    )

    # Ensure city name is consistent
    df["city"] = city.title()

    return df


def main():

    print("=" * 60)
    print("Zimbabwe Weather Intelligence")
    print("Weather Data Preprocessing")
    print("=" * 60)

    city_data = []

    for city in CITIES:

        print(f"\nProcessing {city.title()}...")

        try:
            # Load data
            df = load_city_data(city)

            print(f"Original shape: {df.shape}")

            # Clean data
            df = clean_city_data(df, city)

            print(f"Processed shape: {df.shape}")

            print(
                f"Missing values: "
                f"{df.isnull().sum().sum()}"
            )

            print(
                f"Duplicate timestamps: "
                f"{df['time'].duplicated().sum()}"
            )

            city_data.append(df)

        except FileNotFoundError:
            print(
                f"File not found: "
                f"{city}_weather_test.csv"
            )

        except Exception as error:
            print(
                f"Error processing "
                f"{city.title()}: {error}"
            )

    # --------------------------------------------------------
    # Combine all cities
    # --------------------------------------------------------

    if not city_data:
        print("\nNo datasets were processed.")
        return

    master_df = pd.concat(
        city_data,
        ignore_index=True
    )

    # Sort by city and time
    master_df = master_df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    PROCESSED_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_FOLDER /
        "zimbabwe_weather_2015_2025.csv"
    )

    master_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL PROCESSED DATASET")
    print("=" * 60)

    print(f"\nShape: {master_df.shape}")

    print("\nRecords by city:")
    print(master_df["city"].value_counts())

    print("\nDate range:")
    print("Start:", master_df["time"].min())
    print("End:  ", master_df["time"].max())

    print("\nMissing values:")
    print(master_df.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(master_df.duplicated().sum())

    print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 60)
    print("Data preprocessing completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()