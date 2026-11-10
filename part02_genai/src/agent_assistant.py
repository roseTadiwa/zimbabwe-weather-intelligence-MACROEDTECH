
from pathlib import Path
from datetime import datetime

import requests
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import MemorySaver


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_DIR / "data" / "faiss_index"

API_URL = (
    "https://zimbabwe-weather-intelligence-macroedtech"
    ".onrender.com"
)

VALID_CITIES = ["Harare", "Bulawayo", "Mutare", "Gweru"]

print("Loading knowledge base...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = FAISS.load_local(
    str(INDEX_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

print("Knowledge base loaded.")


# --------------------------------------------------
# 2. TOOL: RESEARCH
# --------------------------------------------------

@tool
def search_weather_knowledge(question: str) -> str:
    """Retrieve information from the weather research
    knowledge base to answer weather and climate questions."""

    documents = vector_store.similarity_search(question, k=3)

    if not documents:
        return "No relevant information was found in the knowledge base."

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )


# --------------------------------------------------
# 3. TOOL: COLLECT INPUTS AND PREDICT
# --------------------------------------------------

@tool
def predict_temperature() -> str:
    """Interactively collect the required city, date, time and
    seven weather measurements from the user, then call the
    deployed XGBoost API for a one-hour-ahead temperature prediction.
    Use only when the user requests a numerical prediction.
    Never invent missing measurements."""

    def ask_number(label):
        while True:
            value = input(f"{label}: ").strip()
            try:
                return float(value)
            except ValueError:
                print("Please enter a valid number.")

    print("\n--- Temperature Prediction ---")
    print("Available cities:", ", ".join(VALID_CITIES))

    while True:
        city = input("City: ").strip().title()
        if city in VALID_CITIES:
            break
        print("Please select one of the listed cities.")

    while True:
        date_text = input("Date (YYYY-MM-DD): ").strip()
        time_text = input("Time (HH:MM, 24-hour format): ").strip()

        try:
            prediction_datetime = datetime.strptime(
                f"{date_text} {time_text}",
                "%Y-%m-%d %H:%M",
            ).strftime("%Y-%m-%d %H:%M")
            break
        except ValueError:
            print("Invalid date or time. Example: 2025-12-31 and 12:00.")

    print(
        "\nEnter weather measurements for that city and time."
        "\nUse the dashboard or a trusted historical weather source."
        "\nDo not guess the measurements."
    )

    temperature = ask_number("Temperature (°C)")

    while True:
        humidity = ask_number("Relative humidity (%)")
        if 0 <= humidity <= 100:
            break
        print("Humidity must be between 0 and 100.")

    while True:
        precipitation = ask_number("Precipitation (mm)")
        if precipitation >= 0:
            break
        print("Precipitation cannot be negative.")

    while True:
        pressure = ask_number("Pressure (hPa)")
        if pressure > 0:
            break
        print("Pressure must be greater than zero.")

    while True:
        wind_speed = ask_number("Wind speed (km/h)")
        if wind_speed >= 0:
            break
        print("Wind speed cannot be negative.")

    while True:
        wind_direction = ask_number("Wind direction (degrees)")
        if 0 <= wind_direction <= 360:
            break
        print("Wind direction must be between 0 and 360.")

    while True:
        cloud_cover = ask_number("Cloud cover (%)")
        if 0 <= cloud_cover <= 100:
            break
        print("Cloud cover must be between 0 and 100.")

    payload = {
        "city": city,
        "prediction_datetime": prediction_datetime,
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "cloud_cover": cloud_cover,
    }

    print("\nSending measurements to the XGBoost API...")

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=(10, 45),
        )
        response.raise_for_status()
        result = response.json()

    except requests.exceptions.Timeout:
        return "The API timed out. No prediction was obtained."

    except requests.exceptions.RequestException as error:
        return (
            "The API request failed. No prediction was obtained. "
            f"Details: {error}"
        )

    except ValueError:
        return "The API returned an unreadable response."

    required_fields = [
        "city",
        "forecast_datetime",
        "predicted_temperature_c",
        "model",
    ]

    if any(field not in result for field in required_fields):
        return "The API response was incomplete. No prediction reported."

    try:
        predicted = float(result["predicted_temperature_c"])
    except (ValueError, TypeError):
        return "The API returned an invalid temperature."

    return (
        "\nACTUAL API PREDICTION\n"
        f"City: {result['city']}\n"
        f"Forecast time: {result['forecast_datetime']}\n"
        f"Predicted temperature: {predicted:.2f} °C\n"
        f"Model: {result['model']}\n"
        f"Forecast horizon: "
        f"{result.get('forecast_horizon', '1 hour ahead')}\n"
        "This is a model prediction, not a direct observation."
    )


# --------------------------------------------------
# 4. CREATE THE AGENT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are the Zimbabwe Weather Research Assistant.

You have two tools:

1. search_weather_knowledge:
   Use this to retrieve information for weather and climate
   research questions.

2. predict_temperature:
   Use this when the user requests a numerical temperature
   prediction. The tool itself collects the required inputs
   interactively. Do not invent or supply weather measurements.

The project uses an XGBoost model to predict temperature one
hour ahead for Harare, Bulawayo, Mutare and Gweru.

The model predicts temperature only. Do not claim it predicts
thunderstorms or rainfall probabilities.

Use tool results for factual answers. If a tool fails, explain
that the requested result was not obtained. Never invent a
numerical prediction.
"""

tools = [
    search_weather_knowledge,
    predict_temperature,
]

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)


# --------------------------------------------------
# 5. CHAT LOOP
# --------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("ZIMBABWE WEATHER RESEARCH AGENT")
    print("LangChain + LangGraph + Ollama + FAISS + XGBoost API")
    print("Type 'exit' to end.")
    print("=" * 55)

    config = {
        "configurable": {
            "thread_id": "zimbabwe-weather-session"
        }
    }

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Conversation ended.")
                break

            if not question:
                continue

            print("\nAgent is thinking...")

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                },
                config=config,
            )

            print("\nAssistant:", result["messages"][-1].content)

        except KeyboardInterrupt:
            print("\nConversation ended.")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()
