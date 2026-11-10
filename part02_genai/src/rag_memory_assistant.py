
from pathlib import Path
from datetime import datetime

import requests
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_DIR / "data" / "faiss_index"

API_URL = (
    "https://zimbabwe-weather-intelligence-macroedtech"
    ".onrender.com"
)

VALID_CITIES = ["Harare", "Bulawayo", "Mutare", "Gweru"]

PROJECT_FACTS = """
The Zimbabwe Weather Intelligence project uses an XGBoost model
to predict temperature one hour ahead for Harare, Bulawayo,
Mutare and Gweru.

The model predicts temperature, not thunderstorm probabilities
or rainfall forecasts. Actual numerical predictions must come
from the prediction API. Never invent a prediction.
"""

# =========================================================
# LOAD RAG KNOWLEDGE BASE
# =========================================================

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

conversation_history = []


# =========================================================
# ACTUAL MODEL PREDICTION
# =========================================================

def get_actual_prediction():
    """Collect inputs and request a real XGBoost prediction."""

    print("\nAvailable cities:", ", ".join(VALID_CITIES))

    city = input("City: ").strip().title()

    if city not in VALID_CITIES:
        return (
            f"Invalid city: {city}. "
            f"Choose from {', '.join(VALID_CITIES)}."
        )

    print(
        "\nEnter the date and time of the weather inputs "
        "in Zimbabwe local time."
    )
    print("Format: YYYY-MM-DD HH:MM")
    prediction_datetime = input("Date and time: ").strip()

    try:
        datetime.strptime(prediction_datetime, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Invalid date/time. Use YYYY-MM-DD HH:MM."

    print(
        "\nEnter the weather measurements for that time."
        "\nUse actual observations or trusted data, not guesses."
    )

    try:
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Relative humidity (%): "))
        precipitation = float(input("Precipitation (mm): "))
        pressure = float(input("Pressure (hPa): "))
        wind_speed = float(input("Wind speed (km/h): "))
        wind_direction = float(input("Wind direction (degrees): "))
        cloud_cover = float(input("Cloud cover (%): "))
    except ValueError:
        return "Invalid input. All weather values must be numbers."

    if not 0 <= humidity <= 100:
        return "Relative humidity must be between 0 and 100."

    if precipitation < 0:
        return "Precipitation cannot be negative."

    if pressure <= 0 or wind_speed < 0:
        return "Pressure must be positive and wind speed non-negative."

    if not 0 <= wind_direction <= 360:
        return "Wind direction must be between 0 and 360 degrees."

    if not 0 <= cloud_cover <= 100:
        return "Cloud cover must be between 0 and 100."

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

    print("\nRequesting a prediction from the XGBoost API...")

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=(10, 45),
        )
        response.raise_for_status()
        result = response.json()

    except requests.exceptions.Timeout:
        return (
            "The weather API did not respond in time. "
            "No prediction was obtained. The deployed service "
            "may be sleeping or unavailable; try again later."
        )

    except requests.exceptions.RequestException as error:
        return (
            "The prediction API request failed. "
            f"No prediction was obtained. Details: {error}"
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
        return (
            "The API response did not contain all expected "
            "prediction fields. No result will be reported."
        )

    try:
        predicted_temp = float(result["predicted_temperature_c"])
    except (ValueError, TypeError):
        return "The API returned an invalid temperature value."

    answer = (
        "\nACTUAL API PREDICTION\n"
        f"City: {result['city']}\n"
        f"Forecast time: {result['forecast_datetime']}\n"
        f"Predicted temperature: {predicted_temp:.2f} °C\n"
        f"Model: {result['model']}\n"
        f"Forecast horizon: {result.get('forecast_horizon', '1 hour ahead')}\n"
        "\nThis is a model prediction, not a direct observation."
    )

    return answer


# =========================================================
# RAG QUESTION ANSWERING WITH CONVERSATION MEMORY
# =========================================================

def ask_weather_question(question: str) -> str:
    documents = vector_store.similarity_search(question, k=3)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    history = "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in conversation_history[-10:]
    )

    prompt = f"""
You are a Zimbabwe Weather Research Assistant.

PROJECT FACTS:
{PROJECT_FACTS}

RETRIEVED KNOWLEDGE:
{context}

CONVERSATION HISTORY:
{history if history else "No previous conversation."}

CURRENT QUESTION:
{question}

Instructions:
- Use the retrieved context and project facts for factual answers.
- Use conversation history to understand follow-up questions.
- Never invent temperatures, live weather conditions, forecasts,
  probabilities, or API results.
- If asked for an actual numerical prediction, explain that the
  prediction API must be called with valid weather inputs.
- The existing model predicts temperature only, one hour ahead.
- If the context is insufficient, say so clearly.
- Keep the answer concise.
"""

    response = llm.invoke(prompt)
    answer = response.content

    conversation_history.append(
        {"role": "user", "content": question}
    )
    conversation_history.append(
        {"role": "assistant", "content": answer}
    )

    return answer


# =========================================================
# MAIN CHAT LOOP
# =========================================================

def main():
    print("=" * 55)
    print("Zimbabwe Weather Research Assistant")
    print("Local LLM + RAG + Conversation Memory + Weather API")
    print("Type 'exit' to end the conversation.")
    print("=" * 55)

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Conversation ended.")
                break

            if not question:
                continue

            prediction_keywords = (
                "actual predicted temperature",
                "get a prediction",
                "make a prediction",
                "predict the temperature",
                "temperature prediction",
                "give me a prediction",
            )

            wants_prediction = any(
                keyword in question.lower()
                for keyword in prediction_keywords
            )

            if wants_prediction:
                answer = get_actual_prediction()
                print(f"\nAssistant: {answer}")

                conversation_history.append(
                    {"role": "user", "content": question}
                )
                conversation_history.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                print("\nGenerating answer...\n")
                answer = ask_weather_question(question)
                print(f"Assistant: {answer}")

        except KeyboardInterrupt:
            print("\nConversation ended.")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()
