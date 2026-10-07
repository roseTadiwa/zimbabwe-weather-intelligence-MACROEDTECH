import requests
import pandas as pd
from pathlib import Path

# ============================================================
# Zimbabwe Weather Intelligence
# Reusable Weather Data Acquisition
# ============================================================

# Zimbabwe cities and coordinates
CITIES = {
    "Harare": {
        "latitude": -17.8252,
        "longitude": 31.0335
    },
    "Bulawayo": {
        "latitude": -20.1325,
        "longitude": 28.6265
    },
    "Mutare": {
        "latitude": -18.9707,
        "longitude": 32.6709
    },
    "Gweru": {
        "latitude": -19.4552,
        "longitude": 29.8149
    }
}

# Test period
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# Open-Meteo Historical Weather API
URL = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover"
]


def fetch_weather_data(city, latitude, longitude):
    """Fetch historical weather data for one city."""

    print(f"\nConnecting to Open-Meteo for {city}...")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "Africa/Harare",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    response = requests.get(
        URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def convert_to_dataframe(data, city):
    """Convert API response into a Pandas DataFrame."""

    hourly_data = data["hourly"]

    df = pd.DataFrame(hourly_data)

    df["time"] = pd.to_datetime(df["time"])

    df["city"] = city

    return df


def save_data(df, city):
    """Save raw weather data for a city."""

    filename = f"{city.lower()}_weather_test.csv"

    output_path = Path("data/raw") / filename

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Data saved to: {output_path}")


def main():

    print("=" * 60)
    print("Zimbabwe Weather Intelligence")
    print("Multi-City Weather Data Acquisition Test")
    print("=" * 60)

    for city, coordinates in CITIES.items():

        try:
            data = fetch_weather_data(
                city,
                coordinates["latitude"],
                coordinates["longitude"]
            )

            df = convert_to_dataframe(
                data,
                city
            )

            print(f"\n{city} dataset shape:")
            print(df.shape)

            print(f"\n{city} missing values:")
            print(df.isnull().sum().sum())

            save_data(df, city)

        except requests.exceptions.RequestException as error:
            print(f"\nError retrieving {city}: {error}")

        except Exception as error:
            print(f"\nUnexpected error for {city}: {error}")

    print("\n" + "=" * 60)
    print("Multi-city data acquisition test completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()