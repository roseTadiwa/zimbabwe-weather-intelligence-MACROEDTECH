# Zimbabwe Weather Intelligence System

An integrated weather intelligence application combining machine learning-based temperature forecasting with a Generative AI research assistant for Zimbabwean climate information.

## Project Overview

The Zimbabwe Weather Intelligence System consists of two main components:

**Part 1: Weather Prediction System**
- Uses historical weather data from Open-Meteo (2015–2025).
- Applies an XGBoost machine learning model to predict temperature one hour ahead.
- Provides a Flask-based prediction API.
- Hosted on Render.

**Part 2: GenAI Weather Research Assistant**
- Provides a conversational interface for weather and climate-related questions.
- Uses Google's Gemini models for language generation and document embeddings.
- Uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from Zimbabwean climate documents.
- Uses FAISS for vector storage and similarity-based document retrieval.
- Integrates weather prediction functionality through Model Context Protocol (MCP).
- Provides an interactive Streamlit interface.

## System Architecture

The system integrates machine learning, generative AI, document retrieval and API-based services.

### Part 1: Weather Prediction Architecture

1. Historical weather data is collected from Open-Meteo.
2. Data is processed and prepared for model training.
3. An XGBoost model is trained to predict temperature one hour ahead.
4. The trained model is exposed through a Flask API.
5. The API is deployed on Render for external access.

**Prediction API:**  
https://zimbabwe-weather-intelligence-macroedtech.onrender.com

### Part 2: GenAI Architecture

The GenAI assistant follows this workflow:

1. **Document Processing:** Zimbabwean climate and environmental documents are loaded and divided into smaller text chunks.
2. **Embedding Generation:** Google's Gemini embedding model (`gemini-embedding-001`) converts document chunks into numerical representations.
3. **Vector Storage:** FAISS stores the embeddings in a searchable vector index.
4. **Information Retrieval:** Relevant document chunks are retrieved based on the user's question.
5. **Response Generation:** Gemini generates responses using the retrieved information.
6. **Tool Integration:** MCP connects the assistant to the deployed weather prediction API, allowing it to access weather-related functionality.
7. **User Interface:** Streamlit provides the interactive application through which users access the assistant.

### Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | XGBoost |
| Prediction API | Flask |
| API hosting | Render |
| Generative AI | Google Gemini |
| Embeddings | Gemini Embedding (`gemini-embedding-001`) |
| LLM integration | LangChain |
| Agent orchestration | LangGraph |
| Vector database | FAISS |
| Retrieval approach | Retrieval-Augmented Generation (RAG) |
| Tool integration | Model Context Protocol (MCP) |
| Application interface | Streamlit |
| Deployment target for Part 2 | Streamlit Community Cloud |

## Project Structure

```text
zimbabwe-weather-intelligence/
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── reports/
│   └── ...
│
├── part02_genai/
│   │
│   ├── data/
│   │   └── faiss_index_gemini/
│   │
│   ├── documents/
│   │   └── ...
│   │
│   ├── src/
│   │   ├── weather_assistant_ui.py
│   │   ├── integrated_weather_assistant.py
│   │   ├── build_knowledge_base.py
│   │   └── weather_mcp_server.py
│   │
│   └── requirements.txt
│
├── requirements.txt
├── README.md
└── ...
```

*Note: The structure above highlights the main files relevant to Part 2. Other project files and directories are omitted for clarity.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH.git
```

Navigate into the project directory:

```bash
cd zimbabwe-weather-intelligence-MACROEDTECH
```

### 2. Create a Virtual Environment

```bash
python -m venv part02_genai/venv
```

Activate the environment on Windows:

```cmd
part02_genai\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r part02_genai/requirements.txt
```

### 4. Configure the Gemini API Key

The GenAI assistant requires a Google Gemini API key.

Obtain an API key through Google AI Studio and configure it as an environment variable.

On Windows Command Prompt:

```cmd
set GEMINI_API_KEY=your_api_key
```

For persistent local configuration, use a Windows environment variable instead.

Do not hardcode API keys in source files or commit them to GitHub.

For Streamlit Community Cloud deployment, configure the key through the application's Secrets settings.

### 5. Build the Knowledge Base

If the Gemini FAISS index has not already been generated, run:

```cmd
python part02_genai\src\build_knowledge_base.py
```

This processes the available climate documents, generates Gemini embeddings and saves the resulting FAISS index.

The application uses:

```text
part02_genai/data/faiss_index_gemini/
```

The knowledge base only needs to be rebuilt when the source documents are changed or the index needs to be regenerated.

## Running the Application Locally

From the repository root, activate the virtual environment and run:

```cmd
python -m streamlit run part02_genai\src\weather_assistant_ui.py
```

Open the local application in your browser:

```text
http://localhost:8501
```

The application provides access to the GenAI research assistant and its integrated weather prediction functionality.

## Deployment

### Part 1: Render

The Flask-based XGBoost prediction API is deployed on Render.

API endpoint:

https://zimbabwe-weather-intelligence-macroedtech.onrender.com

The API supports health checks, available-city retrieval and temperature prediction.

**Note:** Render's free service may spin down after inactivity. The MCP server implements automatic retries to accommodate API cold starts when prediction requests are made.

### Part 2: Streamlit Community Cloud

The Streamlit application is intended for deployment through Streamlit Community Cloud.

Deployment configuration:

- Repository: `roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH`
- Branch: `master`
- Main application file: `part02_genai/src/weather_assistant_ui.py`
- Dependencies: `part02_genai/requirements.txt`
- Secret: `GEMINI_API_KEY`

The Gemini API key must be configured through Streamlit Secrets and must not be committed to the repository.

## Key Features

### Weather Prediction
- One-hour-ahead temperature prediction.
- Integration with the deployed XGBoost model.
- Access to supported Zimbabwean cities.
- API-based prediction requests.

### GenAI Research Assistant
- Natural-language interaction.
- Retrieval of relevant information from Zimbabwean climate documents.
- Gemini-powered response generation.
- FAISS-based semantic document retrieval.
- Evidence-supported responses using retrieved document references.
- Integration with weather prediction tools through MCP.

## Important Configuration and Usage Notes

- An active internet connection is required for Gemini API access and communication with the deployed prediction API.
- Gemini API usage is subject to the applicable model quotas and rate limits.
- The prediction API may take time to respond when a free Render instance is waking up.
- The FAISS index must be available for document retrieval.
- API credentials must be stored securely outside the source code.
- Free hosting and API services are subject to provider limits and availability.

## Future Improvements

Potential areas for further development include:

- Improved weather prediction accuracy and model evaluation.
- Expansion of the climate knowledge base.
- Enhanced source attribution and evidence presentation.
- Improved API cold-start handling and response reliability.
- Additional weather intelligence and climate analytics capabilities.
- Improved application monitoring and deployment reliability.

## Author

**Rosemary Mtape**

Data Science and Systems Graduate

GitHub: [roseTadiwa](https://github.com/roseTadiwa)

Project: Zimbabwe Weather Intelligence System

Developed as part of the MacroEdTech programme.