
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# Zimbabwe Weather Intelligence
# Exploratory Data Analysis
# ============================================================

DATA_PATH = Path("data/processed/zimbabwe_weather_2015_2025.csv")
OUTPUT_FOLDER = Path("reports/figures")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    """Load the processed weather dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}\n"
            "Please run this script from the project root folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "time",
        "city",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    invalid_times = df["time"].isna().sum()

    if invalid_times > 0:
        print(
            f"Warning: removing {invalid_times} rows "
            "with invalid timestamps."
        )
        df = df.dropna(subset=["time"]).copy()

    return df


# ============================================================
# 2. SAVE FIGURES
# ============================================================

def save_figure(filename):
    """Save a figure at high resolution and close it."""

    output_path = OUTPUT_FOLDER / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# 3. TEMPERATURE DISTRIBUTION BY CITY
# ============================================================

def temperature_distribution(df):
    """Compare temperature distributions across cities."""

    plt.figure(figsize=(10, 6))

    for city in sorted(df["city"].dropna().unique()):

        city_data = df.loc[
            df["city"] == city,
            "temperature_2m"
        ].dropna()

        plt.hist(
            city_data,
            bins=40,
            alpha=0.5,
            label=city
        )

    plt.title("Temperature Distribution by City")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.legend(title="City")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    save_figure("temperature_distribution_by_city.png")


# ============================================================
# 4. AVERAGE MONTHLY TEMPERATURE
# ============================================================

def monthly_temperature(df):
    """Plot average monthly temperature for each city."""

    working = df.copy()
    working["month"] = working["time"].dt.month

    monthly = (
        working.groupby(
            ["city", "month"],
            as_index=False
        )["temperature_2m"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    for city in sorted(monthly["city"].dropna().unique()):

        city_data = monthly[
            monthly["city"] == city
        ].sort_values("month")

        plt.plot(
            city_data["month"],
            city_data["temperature_2m"],
            marker="o",
            label=city
        )

    plt.title(
        "Average Monthly Temperature by City (2015–2025)"
    )
    plt.xlabel("Month")
    plt.ylabel("Average Temperature (°C)")
    plt.xticks(range(1, 13))
    plt.legend(title="City")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_figure("average_monthly_temperature.png")


# ============================================================
# 5. MONTHLY PRECIPITATION
# ============================================================

def monthly_precipitation(df):
    """Plot average monthly precipitation totals by city."""

    working = df.copy()

    working["year"] = working["time"].dt.year
    working["month"] = working["time"].dt.month

    # Calculate total precipitation for each city,
    # year, and calendar month.
    monthly_totals = (
        working.groupby(
            ["city", "year", "month"],
            as_index=False
        )["precipitation"]
        .sum(min_count=1)
    )

    # Average each calendar month's totals across years.
    monthly_average = (
        monthly_totals.groupby(
            ["city", "month"],
            as_index=False
        )["precipitation"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    for city in sorted(monthly_average["city"].dropna().unique()):

        city_data = monthly_average[
            monthly_average["city"] == city
        ].sort_values("month")

        plt.plot(
            city_data["month"],
            city_data["precipitation"],
            marker="o",
            label=city
        )

    plt.title(
        "Average Monthly Precipitation by City (2015–2025)"
    )
    plt.xlabel("Month")
    plt.ylabel("Average Monthly Precipitation (mm)")
    plt.xticks(range(1, 13))
    plt.legend(title="City")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_figure("monthly_precipitation_by_city.png")


# ============================================================
# 6. ANNUAL TEMPERATURE TREND
# ============================================================

def temperature_trend(df):
    """Plot annual average temperature for each city."""

    working = df.copy()
    working["year"] = working["time"].dt.year

    annual = (
        working.groupby(
            ["city", "year"],
            as_index=False
        )["temperature_2m"]
        .mean()
    )

    plt.figure(figsize=(12, 6))

    for city in sorted(annual["city"].dropna().unique()):

        city_data = annual[
            annual["city"] == city
        ].sort_values("year")

        plt.plot(
            city_data["year"],
            city_data["temperature_2m"],
            marker="o",
            label=city
        )

    plt.title(
        "Annual Average Temperature (2015–2025)"
    )
    plt.xlabel("Year")
    plt.ylabel("Annual Average Temperature (°C)")
    plt.xticks(sorted(annual["year"].dropna().unique()))
    plt.legend(title="City")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_figure("annual_temperature_trend.png")


# ============================================================
# 7. WEATHER VARIABLE CORRELATION HEATMAP
# ============================================================

def correlation_analysis(df):
    """
    Calculate Pearson correlations and create a heatmap
    with numerical correlation coefficients in every cell.
    """

    columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover"
    ]

    # Ensure the selected variables are numeric.
    weather_data = df[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Calculate Pearson correlation coefficients.
    correlation = weather_data.corr(
        method="pearson"
    )

    # Print the actual values for interpretation.
    print("\nPearson Correlation Matrix:")
    print(correlation.round(3).to_string())

    # Save the correlation coefficients to a CSV file.
    csv_path = (
        OUTPUT_FOLDER /
        "weather_variable_correlation.csv"
    )

    correlation.to_csv(
        csv_path,
        float_format="%.4f"
    )

    print(f"\nCorrelation values saved to: {csv_path}")

    # Create the heatmap.
    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    image = ax.imshow(
        correlation.to_numpy(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="equal"
    )

    # Add numerical values inside each cell.
    for row in range(len(columns)):

        for col in range(len(columns)):

            value = correlation.iloc[row, col]

            if pd.isna(value):
                label = "N/A"
                text_colour = "black"
            else:
                label = f"{value:.2f}"

                # Improve readability against darker colours.
                text_colour = (
                    "white"
                    if abs(value) >= 0.55
                    else "black"
                )

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color=text_colour,
                fontsize=10,
                fontweight="bold"
            )

    # Display full variable names.
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))

    ax.set_xticklabels(
        columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(columns)

    ax.set_title(
        "Correlation Between Weather Variables",
        fontsize=14,
        pad=18
    )

    # Use a consistent colour scale from -1 to +1.
    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    colorbar.set_label(
        "Pearson correlation coefficient"
    )

    colorbar.set_ticks(
        [-1, -0.5, 0, 0.5, 1]
    )

    # Add cell boundaries.
    ax.set_xticks(
        np.arange(-0.5, len(columns), 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, len(columns), 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    fig.tight_layout()

    save_figure(
        "weather_variable_correlation.png"
    )


# ============================================================
# 8. RUN ALL EDA FUNCTIONS
# ============================================================

def main():

    print("=" * 60)
    print("Zimbabwe Weather Intelligence")
    print("Exploratory Data Analysis")
    print("=" * 60)

    print("\nLoading processed dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    print(
        f"Cities: {sorted(df['city'].dropna().unique())}"
    )

    print(
        f"Date range: {df['time'].min()} to {df['time'].max()}"
    )

    print("\nGenerating EDA visualisations...\n")

    temperature_distribution(df)
    print("Completed: temperature distribution")

    monthly_temperature(df)
    print("Completed: monthly temperature")

    monthly_precipitation(df)
    print("Completed: monthly precipitation")

    temperature_trend(df)
    print("Completed: annual temperature trend")

    correlation_analysis(df)
    print("Completed: weather correlation")

    print("\n" + "=" * 60)
    print("EDA completed successfully.")
    print(f"Figures saved to: {OUTPUT_FOLDER.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()