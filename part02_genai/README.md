# Zimbabwe Weather Intelligence – Part 2: Generative AI Weather Research Assistant

**Organisation:** MacroEdTech  
**Project component:** Part 2 – Generative AI Integration  
**Primary language:** Python  
**User interface:** Streamlit  
**Repository:** [roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH](https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH)

## Local Development and Cloud Deployment

The application was initially developed and tested locally using Ollama, and that working local implementation has been retained. For cloud deployment, hosting Ollama and its downloaded model files on the application server would require additional disk storage, memory and compute resources, increasing hosting complexity and potentially the cost of deployment. The deployed version therefore uses the Gemini API, with language generation and embeddings handled through Google's hosted service rather than by hosting Ollama models alongside the Streamlit application. This allows a supervisor or reviewer to access the application through a browser without installing Ollama or downloading models. The Gemini-based deployment still depends on internet connectivity, API credentials and provider request quotas or applicable charges.

## 1. Project Overview

Part 2 extends the Zimbabwe Weather Intelligence system developed in Part 1. It adds a Generative AI research assistant that can retrieve information from a Zimbabwe-focused climate knowledge base and provide contextual responses to weather, climate and environmental questions. It also connects the application to the Part 1 temperature-prediction service.

The implementation has two model configurations that should be distinguished:

- **Local development implementation:** The application was initially developed and tested on the developer's computer using Ollama, with Llama 3.2:3b for language generation and nomic-embed-text for embeddings.
- **Deployed implementation:** The application was subsequently adapted to use the Gemini API for language generation and embeddings. This made it possible to deploy the application on Streamlit Community Cloud so that a supervisor or other reviewer can access and test it through a browser without installing and running Ollama locally.

The local Ollama implementation has been retained separately as part of the development work. The deployed Gemini implementation is the version intended for browser-based access. The deployed version therefore depends on a Gemini API key and the availability and quota of the configured Gemini service; it is not a fully local LLM deployment.

## 2. Project Objectives

Part 2 aims to:

1. Develop a domain-specific knowledge base containing Zimbabwean climate, weather and environmental information.
2. Extract text from source documents and prepare it for retrieval.
3. Split documents into manageable chunks and generate vector embeddings.
4. Store and retrieve document embeddings using FAISS.
5. Implement semantic search and Retrieval-Augmented Generation (RAG).
6. Develop an AI-assisted research workflow using LangChain and LangGraph.
7. Integrate selected external functionality through the Model Context Protocol (MCP).
8. Connect the GenAI application to the Part 1 temperature-prediction API.
9. Maintain short-term conversation context during an active application session.
10. Provide an accessible user interface through Streamlit.

## 3. Main Features

### 3.1 Weather and Climate Research Assistant

The Research Assistant accepts natural-language questions about Zimbabwe's climate, weather, historical droughts, agricultural impacts, climate vulnerability, emissions and selected climate projections. For document-based questions, it searches the project knowledge base and uses retrieved evidence to support the response.

### 3.2 Retrieval-Augmented Generation

The RAG workflow combines document retrieval with language-model response generation:

1. Load the available research documents.
2. Extract text and associated source metadata.
3. Split the text into overlapping chunks.
4. Generate embeddings for the chunks.
5. store the embeddings in a FAISS vector index.
6. Embed a user's research question and retrieve semantically similar chunks.
7. Pass the retrieved evidence into the response workflow.
8. Generate and display a contextual answer.

The deployed knowledge base uses the Gemini embedding model `models/gemini-embedding-001`. The retained local implementation used Ollama's `nomic-embed-text`. These embedding configurations are not interchangeable: an index should be queried using the compatible embedding model with which it was built.

### 3.3 AI Agent and Workflow Orchestration

LangChain provides model, retrieval and tool integrations. LangGraph is used to organise the agent workflow, including agent execution and response validation. The research tool provides access to the FAISS knowledge base. The deployed agent's available tools should be understood from its configured tool list; the existence of an MCP server tool does not by itself mean that every tool is directly available to the conversational agent.

### 3.4 Model Context Protocol

The Part 2 MCP server exposes tools that communicate with the Part 1 prediction API:

- `get_weather_api_health` – checks the prediction API's health endpoint.
- `get_available_cities` – retrieves the cities supported by the prediction service.
- `predict_temperature` – submits prediction inputs to the Part 1 API and returns its response.

The prediction tool is implemented in the MCP server. In the integrated assistant, temperature prediction is handled through a Python-controlled MCP workflow rather than being included in the general LangChain agent tool list. The Streamlit Temperature Prediction interface also calls the Part 1 API directly.

### 3.5 Temperature Prediction

The application connects to the Part 1 FastAPI service, which uses the XGBoost model to generate a one-hour-ahead temperature prediction for a supported city based on supplied input values. The supported cities are Harare, Bulawayo, Mutare and Gweru.

This is a model prediction, not a direct observation or a guarantee of future conditions. The application does not independently retrieve live weather observations.

### 3.6 Conversation Memory

Recent messages are maintained in Streamlit session state. The configured history limit is ten messages (approximately five user-assistant exchanges). This is short-term, session-based memory; it is not permanent conversation storage across separate sessions.

## 4. Technology Stack

| Technology | Role in the project |
|---|---|
| Python | Main programming language |
| Streamlit | User interface and deployment |
| Gemini API | Language generation and embeddings in the deployed implementation |
| Ollama | Local model execution in the retained local implementation |
| Llama 3.2:3b | Local language model used during development |
| nomic-embed-text | Local embedding model used during development |
| `models/gemini-embedding-001` | Embedding model used by the deployed implementation |
| LangChain | Document processing, retrieval and model/tool integration |
| LangGraph | Agent workflow orchestration |
| FAISS | Vector index and similarity retrieval |
| MCP / FastMCP | Tool interface to the Part 1 API |
| FastAPI | Part 1 prediction API |
| XGBoost | One-hour-ahead temperature prediction |
| pypdf | PDF text extraction |
| Git and GitHub | Version control and source-code management |

The deployed application uses a hosted Gemini API. Streamlit Community Cloud hosts the user interface, while the Part 1 prediction API is hosted separately on Render. Consequently, the deployed configuration is not entirely local or independent of hosted services.

## 5. Knowledge Base

The knowledge base contains Zimbabwe-specific climate, weather, environmental and policy information. The documented source set includes:

- `weather_research_basics.txt`
- `zimbabwe_climate_risk_country_profile.pdf`
- `zimbabwe_revised_ndc_2021.pdf`
- `zimbabwe_revised_ndc_evidence.md`

The build process extracts text, preserves source metadata, applies chunking and creates a FAISS index. The recorded build produced 217 chunks and passed five focused retrieval smoke tests. These tests check whether selected evidence can be retrieved; they do not establish that every generated answer is factually correct.

The deployed FAISS index is stored at:

```text
part02_genai/data/faiss_index_gemini/
```

It contains the FAISS index files, including `index.faiss` and `index.pkl`. The retained Ollama version used a separate index configuration. Do not reuse an index across different embedding models without rebuilding it.

## 6. Repository Structure

The main Part 2 files include:

```text
zimbabwe-weather-intelligence/
├── part02_genai/
│   ├── documents/
│   ├── data/
│   │   └── faiss_index_gemini/
│   │       ├── index.faiss
│   │       └── index.pkl
│   ├── src/
│   │   ├── build_knowledge_base.py
│   │   ├── weather_mcp_server.py
│   │   ├── integrated_weather_assistant.py
│   │   └── weather_assistant_ui.py
│   ├── requirements.txt
│   └── TECHNICAL_MANUAL.md
├── app/
│   └── app.py
└── README.md
```

This is a high-level view of the relevant files; local virtual-environment folders and other project files are omitted. The Streamlit deployment uses `part02_genai/src/weather_assistant_ui.py` as its main file.

## 7. Configuration and Setup

### 7.1 Requirements

- Python 3.12
- Git
- Access to the project repository
- A Gemini API key for the deployed/current Gemini-based application
- Internet access for Gemini API requests and the hosted Part 1 prediction API

The original local Ollama implementation additionally requires Ollama and the relevant local models.

### 7.2 Clone the Repository

```bash
git clone https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH.git
cd zimbabwe-weather-intelligence-MACROEDTECH
```

### 7.3 Create and Activate a Virtual Environment

On Windows:

```cmd
python -m venv part02_genai\venv
part02_genai\venv\Scripts\activate
```

### 7.4 Install Dependencies

```cmd
pip install -r part02_genai\requirements.txt
```

For Streamlit Community Cloud, a copy of the Part 2 requirements file is also maintained at `part02_genai/src/requirements.txt` so that the deployment can detect the dependencies from the app's configured directory.

### 7.5 Configure the Gemini API Key

The deployed application reads the API key from Streamlit secrets using the name `GEMINI_API_KEY`. The application transfers that secret to the environment variables used by the Gemini integration.

For local execution, configure the key through Streamlit secrets or the environment, without committing the key to Git. Never place API keys directly in source code or publish them in the repository.

A local Streamlit secrets file can be created at `.streamlit/secrets.toml` with:

```toml
GEMINI_API_KEY = "your-key"
```

Keep this file out of version control.

### 7.6 Build or Rebuild the Knowledge Base

If the source documents change, rebuild the knowledge base using the embedding configuration intended for the application:

```cmd
python part02_genai\src\build_knowledge_base.py
```

The deployed Gemini configuration requires access to the Gemini embedding service. Review the script output and confirm that the index is saved to the configured Gemini index directory before launching the application.

## 8. Running the Application Locally

From the repository root, activate the virtual environment and run:

```cmd
python -m streamlit run part02_genai\src\weather_assistant_ui.py
```

Using `python -m streamlit` is a practical Windows alternative where direct execution of `streamlit.exe` is blocked by local security controls.

The browser opens the local Streamlit interface. The Gemini-based version requires a valid API key and network access. The research assistant also requires the compatible FAISS index to be present.

The earlier Ollama implementation is retained separately. To run that version, use its corresponding code and local configuration, install Ollama, and ensure the required local models and compatible FAISS index are available. The Ollama setup should not be confused with the currently deployed Gemini configuration.

## 9. Deployment

Part 2 is deployed as a separate Streamlit Community Cloud application. Its configured main file is:

```text
part02_genai/src/weather_assistant_ui.py
```

The Gemini API key is configured in the application's Streamlit Cloud Secrets; it is not stored in the source code.

The Part 1 prediction API is hosted separately:

```text
https://zimbabwe-weather-intelligence-macroedtech.onrender.com
```

The Render free-tier service may enter a sleep state after inactivity, so the first request can take longer while the service starts. The MCP server includes retry handling for connection and timeout errors.

The public Part 2 application URL should be added here when confirmed:
`[Add the deployed Part 2 Streamlit URL]`

## 10. Part 1 Prediction API

The Part 2 application integrates with the Part 1 FastAPI service. Relevant endpoints include:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | API health check |
| `/cities` | GET | Lists supported cities |
| `/predict` | POST | Generates a temperature prediction |
| `/docs` | GET | Interactive API documentation |

API base URL:

```text
https://zimbabwe-weather-intelligence-macroedtech.onrender.com
```

The prediction service supports Harare, Bulawayo, Mutare and Gweru. It provides a one-hour-ahead temperature prediction based on the supplied input variables.

## 11. Testing and Evaluation

Recorded testing activities include:

- Document loading and PDF text extraction.
- NDC evidence checks and climate projection table processing.
- Chunk generation and embedding creation.
- FAISS index construction.
- Five focused retrieval smoke tests, all of which passed.
- Research-question tests covering historical drought impacts, agricultural impacts and AFOLU emissions.
- Prediction API health, supported-city and prediction requests.
- MCP health, city and prediction tool calls.
- Local Streamlit interface testing.
- Deployed Streamlit application testing, including the research assistant and temperature-prediction interface.
- Conversation-history handling through Streamlit session state.

The general research workflow may still produce incomplete or inaccurate interpretations. Retrieval smoke tests demonstrate that selected evidence can be found; they are not a comprehensive evaluation of answer quality.

Gemini API quota limits can also affect testing. A quota-exhaustion response is a service-limit issue and should not, by itself, be interpreted as proof that the MCP server or application code is defective.

## 12. Limitations

1. **Knowledge-base coverage:** The assistant is limited by the range and quality of the documents currently included.
2. **Generated-answer accuracy:** Responses can contain incomplete interpretations or unsupported statements. Important findings should be checked against the source documents.
3. **Retrieval quality:** Relevant evidence may not always appear in the retrieved chunks.
4. **Gemini dependency:** The deployed research assistant depends on the Gemini API, a valid secret and available request quota.
5. **Hosted-service availability:** The Part 1 API is hosted on Render and may experience cold starts or temporary delays.
6. **Prediction scope:** The existing model provides one-hour-ahead temperature predictions for four supported cities; it is not a complete multi-day forecast.
7. **Conversation memory:** Recent messages are retained within the active Streamlit session, not as permanent cross-session records.
8. **Deployment differences:** The deployed Gemini version and retained local Ollama version use different model and embedding configurations.

## 13. Future Improvements

Potential improvements include expanding the source-document collection, improving retrieval and ranking, creating a broader answer-evaluation dataset, strengthening source citations, improving service monitoring, and exploring additional prediction horizons and weather variables.

## 14. Conclusion

Part 2 extends the original temperature-prediction system with document-based climate research and Generative AI. The implementation includes a Zimbabwe-focused knowledge base, FAISS retrieval, RAG, LangChain and LangGraph workflows, MCP integration and a Streamlit interface.

The development process began with a local Ollama implementation. The application was subsequently adapted to use Gemini so that it could be deployed and accessed by a supervisor through Streamlit Community Cloud. The retained Ollama version documents the local development approach, while the Gemini-based version represents the deployed configuration. This distinction is important when reproducing, testing or maintaining the system.
