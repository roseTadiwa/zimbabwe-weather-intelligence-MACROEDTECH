import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Zimbabwe Weather Intelligence
# Exploratory Data Analysis
# ============================================================

DATA_PATH = Path(
    "data/processed/zimbabwe_weather_2015_2025.csv"
)

OUTPUT_FOLDER = Path("reports/figures")

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def load_data():
    """Load the processed weather dataset."""

    df = pd.read_csv(DATA_PATH)

    df["time"] = pd.to_datetime(df["time"])

    return df


def temperature_distribution(df):
    """Compare temperature distributions across cities."""

    plt.figure(figsize=(10, 6))

    for city in df["city"].unique():

        city_data = df[
            df["city"] == city
        ]["temperature_2m"]

        plt.hist(
            city_data,
            bins=40,
            alpha=0.5,
            label=city
        )

    plt.title(
        "Temperature Distribution by City"
    )

    plt.xlabel(
        "Temperature (°C)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER /
        "temperature_distribution_by_city.png"
    )

    plt.close()


def monthly_temperature(df):
    """Calculate and plot average monthly temperature."""

    df["month"] = df["time"].dt.month

    monthly = (
        df.groupby(
            ["city", "month"]
        )["temperature_2m"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for city in monthly["city"].unique():

        city_data = monthly[
            monthly["city"] == city
        ]

        plt.plot(
            city_data["month"],
            city_data["temperature_2m"],
            marker="o",
            label=city
        )

    plt.title(
        "Average Monthly Temperature by City"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Average Temperature (°C)"
    )

    plt.xticks(
        range(1, 13)
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER /
        "average_monthly_temperature.png"
    )

    plt.close()


def monthly_precipitation(df):
    """Calculate and plot average monthly precipitation."""

    monthly = (
        df.groupby(
            [
                "city",
                df["time"].dt.month
            ]
        )["precipitation"]
        .sum()
        .reset_index()
    )

    monthly = monthly.rename(
        columns={"time": "month"}
    )

    plt.figure(figsize=(10, 6))

    for city in monthly["city"].unique():

        city_data = monthly[
            monthly["city"] == city
        ]

        plt.plot(
            city_data["month"],
            city_data["precipitation"],
            marker="o",
            label=city
        )

    plt.title(
        "Total Monthly Precipitation by City"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Precipitation (mm)"
    )

    plt.xticks(
        range(1, 13)
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER /
        "monthly_precipitation_by_city.png"
    )

    plt.close()


def temperature_trend(df):
    """Plot long-term annual temperature trends."""

    df["year"] = df["time"].dt.year

    annual = (
        df.groupby(
            ["city", "year"]
        )["temperature_2m"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    for city in annual["city"].unique():

        city_data = annual[
            annual["city"] == city
        ]

        plt.plot(
            city_data["year"],
            city_data["temperature_2m"],
            marker="o",
            label=city
        )

    plt.title(
        "Annual Average Temperature Trend (2015–2025)"
    )

    plt.xlabel("Year")

    plt.ylabel(
        "Average Temperature (°C)"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER /
        "annual_temperature_trend.png"
    )

    plt.close()


def correlation_analysis(df):
    """Create a correlation matrix for weather variables."""

    columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover"
    ]

    correlation = df[
        columns
    ].corr()

    plt.figure(figsize=(10, 8))

    plt.imshow(
        correlation,
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(columns)),
        columns
    )

    plt.title(
        "Correlation Between Weather Variables"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER /
        "weather_variable_correlation.png"
    )

    plt.close()


def main():

    print("=" * 60)
    print("Zimbabwe Weather Intelligence")
    print("Exploratory Data Analysis")
    print("=" * 60)

    print("\nLoading processed dataset...")

    df = load_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Cities: {df['city'].unique()}"
    )

    print("\nGenerating EDA visualisations...")

    temperature_distribution(df)
    print("✓ Temperature distribution")

    monthly_temperature(df)
    print("✓ Monthly temperature")

    monthly_precipitation(df)
    print("✓ Monthly precipitation")

    temperature_trend(df)
    print("✓ Annual temperature trend")

    correlation_analysis(df)
    print("✓ Weather correlation")

    print("\n" + "=" * 60)
    print("EDA completed successfully.")
    print(
        f"Figures saved to: {OUTPUT_FOLDER}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()