
import requests
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Zimbabwe Weather Research Server")

API_BASE_URL = (
    "https://zimbabwe-weather-intelligence-macroedtech.onrender.com"
)


@mcp.tool()
def get_weather_api_health() -> dict:
    """Check whether the deployed Zimbabwe weather prediction API is responding."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        return {
            "error": str(error),
            "message": "The weather prediction API could not be reached."
        }


@mcp.tool()
def get_available_cities() -> dict:
    """Get the cities supported by the Zimbabwe weather prediction API."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/cities",
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        return {
            "error": str(error),
            "message": "Could not retrieve the available cities."
        }


@mcp.tool()
def predict_temperature(
    city: str,
    prediction_datetime: str,
    temperature: float,
    humidity: float,
    precipitation: float,
    pressure: float,
    wind_speed: float,
    wind_direction: float,
    cloud_cover: float
) -> dict:
    """
    Predict the temperature one hour ahead using the deployed XGBoost API.

    Supply actual weather measurements for the specified city and time.
    Datetime format: YYYY-MM-DD HH:MM.
    """

    payload = {
        "city": city,
        "prediction_datetime": prediction_datetime,
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "cloud_cover": cloud_cover
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        return {
            "error": str(error),
            "message": "The temperature prediction request failed."
        }


if __name__ == "__main__":
    print("Starting Zimbabwe Weather MCP Server...")
    mcp.run(transport="stdio")
