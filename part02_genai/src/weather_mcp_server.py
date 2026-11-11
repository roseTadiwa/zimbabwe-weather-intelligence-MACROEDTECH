
import requests
import sys
import time
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Zimbabwe Weather Research Server")

API_BASE_URL = (
    "https://zimbabwe-weather-intelligence-macroedtech.onrender.com"
)

# Automatic retry configuration
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT = 90


def make_api_request(method: str, url: str, **kwargs) -> dict:
    """
    Send an API request with automatic retries.

    Allows the Render service time to wake up if it is sleeping.
    Retries connection failures and request timeouts.
    """

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=REQUEST_TIMEOUT,
                **kwargs
            )

            response.raise_for_status()
            return response.json()

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as error:

            if attempt == MAX_ATTEMPTS:
                raise error

            wait_time = 10 * attempt

            print(
                f"API unavailable or still waking up. "
                f"Retrying in {wait_time} seconds "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS})...",
                file=sys.stderr,
                flush=True
            )

            time.sleep(wait_time)

    raise Exception("API request failed after all retry attempts.")


@mcp.tool()
def get_weather_api_health() -> dict:
    """Check whether the deployed Zimbabwe weather prediction API is responding."""

    try:
        return make_api_request(
            "GET",
            f"{API_BASE_URL}/health"
        )

    except requests.RequestException as error:
        return {
            "error": str(error),
            "message": (
                "The weather prediction API could not be reached "
                "after multiple attempts."
            )
        }


@mcp.tool()
def get_available_cities() -> dict:
    """Get the cities supported by the Zimbabwe weather prediction API."""

    try:
        return make_api_request(
            "GET",
            f"{API_BASE_URL}/cities"
        )

    except requests.RequestException as error:
        return {
            "error": str(error),
            "message": (
                "Could not retrieve the available cities "
                "after multiple attempts."
            )
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
        return make_api_request(
            "POST",
            f"{API_BASE_URL}/predict",
            json=payload
        )

    except requests.RequestException as error:
        return {
            "error": str(error),
            "message": (
                "The temperature prediction request failed "
                "after multiple attempts. Please try again shortly."
            )
        }


if __name__ == "__main__":
    print(
        "Starting Zimbabwe Weather MCP Server...",
        file=sys.stderr,
        flush=True,
    )

    mcp.run(transport="stdio")
