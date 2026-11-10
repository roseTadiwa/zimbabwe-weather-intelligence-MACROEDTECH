# Zimbabwe Weather Intelligence – Part 2: Generative AI Weather Research Assistant

## 1. Project Overview

Part 2 extends the Zimbabwe Weather Intelligence System developed in Part 1 by integrating Generative AI technologies to create an intelligent, domain-specific Weather Research Assistant.

The application combines a Retrieval-Augmented Generation (RAG) pipeline, a locally hosted Large Language Model (LLM), vector-based semantic search, AI agents, workflow orchestration and Model Context Protocol (MCP) tools. It also integrates with the weather prediction API developed in Part 1.

The system is designed to provide users with research-based information about Zimbabwe's climate and weather, retrieve relevant information from scientific and policy documents, explain documented climate impacts, and interact with the existing temperature prediction service.

The application is developed locally using free and open-source technologies, with Ollama providing local LLM and embedding model execution.

## 2. Project Objectives

The main objectives of Part 2 are to:

- Develop a domain-specific weather knowledge base using relevant Zimbabwean climate and environmental documents.
- Implement document processing, text chunking and embedding generation.
- Store document embeddings in a FAISS vector database.
- Implement semantic search and a Retrieval-Augmented Generation pipeline.
- Develop an AI-powered research assistant using a locally hosted LLM.
- Implement intelligent agent workflows using LangChain and LangGraph.
- Integrate external functionality through the Model Context Protocol.
- Maintain conversation context through short-term conversation memory.
- Connect the GenAI application to the Part 1 weather prediction API.
- Provide an interactive interface through Streamlit.

## 3. Main System Features

### 3.1 Weather Research Assistant

The Research Assistant allows users to ask questions about Zimbabwe's climate, weather patterns, drought impacts, agricultural vulnerability, water availability and environmental issues.

Its main capabilities include:

- Retrieving relevant information from the weather knowledge base.
- Searching scientific and policy documents using semantic similarity.
- Generating responses grounded in retrieved information.
- Providing source references to support research findings.
- Explaining documented historical climate impacts and future projections.
- Maintaining recent conversation context during an interaction.

### 3.2 Retrieval-Augmented Generation (RAG)

The RAG pipeline combines document retrieval with Large Language Model response generation.

The process involves:

1. Loading relevant research documents.
2. Extracting and processing document content.
3. Splitting documents into smaller text chunks.
4. Generating embeddings using the Ollama embedding model.
5. Storing the embeddings in a FAISS vector database.
6. Retrieving relevant chunks through semantic search when a question is submitted.
7. Passing the retrieved evidence to the LLM to generate a context-grounded response.

### 3.3 AI Agents and Workflow Orchestration

The application uses LangChain to create an AI agent capable of interacting with available tools.

LangGraph is used to orchestrate the agent workflow, including agent execution and response validation.

The available functionality includes:

- Searching the weather knowledge base.
- Retrieving supported cities from the prediction service.
- Checking the weather API's health status.
- Accessing the temperature prediction functionality through the MCP integration.

### 3.4 Model Context Protocol (MCP)

MCP provides a standardised way for the AI application to communicate with external tools.

A weather MCP server exposes tools that interact with the Part 1 prediction API. These include:

- `get_available_cities` – retrieves the cities supported by the prediction service.
- `get_weather_api_health` – checks the availability and operational status of the API.
- `predict_temperature` – submits weather input values to obtain a temperature prediction.

The integrated assistant uses MCP to access these tools. The Streamlit temperature prediction interface also communicates directly with the deployed prediction API.

### 3.5 Temperature Prediction Integration

The GenAI application integrates with the existing Part 1 temperature prediction service.

The prediction model uses XGBoost to generate a temperature prediction one hour ahead based on supplied weather input values.

Users can explore different input conditions through the Temperature Prediction interface.

The system does not independently retrieve live weather observations. Predictions depend on the quality and relevance of the supplied input values.

### 3.6 Conversation Memory

The application maintains recent conversation context to support follow-up questions.

The current implementation retains the latest ten messages, equivalent to five user-assistant exchanges, during the conversation session.

## 4. Technologies and Tools

| Component | Technology |
|---|---|
| Programming language | Python |
| Local LLM runtime | Ollama |
| Language model | Llama 3.2:3b |
| Embedding model | nomic-embed-text |
| AI application framework | LangChain |
| Workflow orchestration | LangGraph |
| Tool integration | Model Context Protocol (MCP) |
| Vector database | FAISS |
| Data processing | Pandas and NumPy |
| Prediction model | XGBoost |
| Prediction API | FastAPI (Part 1) |
| User interface | Streamlit |
| Document processing | PyPDF |
| Version control | Git and GitHub |
| Development environment | Visual Studio Code |

## 5. Knowledge Base and Research Sources

The domain-specific knowledge base is built from research materials relating to Zimbabwe's climate, environmental conditions and climate-related risks.

The source documents include:

- `weather_research_basics.txt`
- `zimbabwe_climate_risk_country_profile.pdf`
- `zimbabwe_revised_ndc_2021.pdf`
- `zimbabwe_revised_ndc_evidence.md`

These materials provide information on climate vulnerability, drought impacts, agricultural challenges, water availability, greenhouse gas emissions and environmental risks.

The processed documents are converted into text chunks and indexed using FAISS to support semantic retrieval.

## 6. Project Structure

The following is a simplified representation of the Part 2 directory structure:

```text
part02_genai/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── agent_assistant.py
│   ├── build_knowledge_base.py
│   ├── evaluate_weather_assistant.py
│   ├── integrated_weather_assistant.py
│   ├── mcp_agent_assistant.py
│   ├── rag_assistant.py
│   ├── rag_memory_assistant.py
│   ├── review_weather_evaluation.py
│   ├── test_mcp_client.py
│   ├── weather_assistant_ui.py
│   └── weather_mcp_server.py
│
├── data/
│   └── faiss_index/
│
├── documents/
│   ├── weather_research_basics.txt
│   ├── zimbabwe_climate_risk_country_profile.pdf
│   ├── zimbabwe_revised_ndc_2021.pdf
│   └── zimbabwe_revised_ndc_evidence.md
│
├── reports/
│   └── evaluation/
│
├── tests/
│
└── venv/
```

The `venv` directory contains the local Python virtual environment and is not required to be committed to GitHub.

## 7. Installation and Setup

### 7.1 Prerequisites

Before running the application, ensure that the following are installed:

- Python 3.12
- Visual Studio Code or another Python development environment
- Git
- Ollama
- An internet connection for accessing the deployed prediction API

### 7.2 Clone the Repository

Open Windows Command Prompt and run:

```cmd
git clone https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH.git
cd zimbabwe-weather-intelligence-MACROEDTECH
```

### 7.3 Create and Activate the Virtual Environment

From the project root directory:

```cmd
python -m venv part02_genai\venv
part02_genai\venv\Scripts\activate
```

### 7.4 Install Dependencies

Install the packages listed in the Part 2 requirements file:

```cmd
python -m pip install -r part02_genai\requirements.txt
```

Install Streamlit if it is not already included in the requirements:

```cmd
python -m pip install streamlit
```

### 7.5 Install and Configure Ollama Models

Install Ollama and ensure that the Ollama service is running.

Download the language model:

```cmd
ollama pull llama3.2:3b
```

Download the embedding model:

```cmd
ollama pull nomic-embed-text
```

Confirm that both models are available:

```cmd
ollama list
```

The application uses these models locally for language generation and document embedding.

## 8. Running the Application

From the repository root directory, activate the virtual environment if it is not already active:

```cmd
part02_genai\venv\Scripts\activate
```

Start the Streamlit application:

```cmd
streamlit run part02_genai\src\weather_assistant_ui.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Open the displayed URL in a web browser.

The application provides the following main sections:

- **Research Assistant:** Allows users to submit weather and climate research questions.
- **Temperature Prediction:** Provides an interface for submitting weather conditions and obtaining one-hour-ahead temperature predictions.
- **About:** Provides information about the application and its technologies.

The Ollama service must be running for local LLM and embedding functionality. The deployed Part 1 prediction API must be accessible for temperature prediction requests.

## 9. Prediction API

The application integrates with the deployed Part 1 prediction API:

**Base URL:**  
https://zimbabwe-weather-intelligence-macroedtech.onrender.com

The API provides the following endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | Returns basic API information |
| `/health` | Reports API and model status |
| `/cities` | Returns supported prediction cities |
| `/predict` | Accepts temperature prediction requests |
| `/docs` | Provides interactive API documentation |

The supported cities are:

- Harare
- Bulawayo
- Mutare
- Gweru

The prediction service uses the trained XGBoost model to predict temperature one hour ahead.

## 10. Evaluation and Testing

The application was evaluated through functional testing, research-question testing and manual review of generated responses.

The evaluation activities included:

- Verifying application startup and interface accessibility.
- Checking Ollama model availability and embedding generation.
- Testing research questions against the weather knowledge base.
- Checking the retrieval of documented climate and drought evidence.
- Testing the temperature prediction API integration.
- Testing MCP tools for supported cities, API health and temperature prediction.
- Checking LangGraph agent execution and response validation.
- Assessing the assistant's ability to retain recent conversation context.

Evaluation outputs are stored in:

```text
part02_genai/reports/evaluation/
```

The directory contains automated evaluation and manual-review results in CSV and JSON formats.

The evaluation results should be interpreted in the context of the questions tested and the available research sources. They do not establish that every possible weather or climate question can be answered accurately.

## 11. Limitations

The current implementation has the following limitations:

- The Research Assistant's knowledge is limited to the documents and information included in its knowledge base.
- Generated responses depend on the quality and relevance of retrieved evidence.
- The application does not independently retrieve live weather observations.
- The temperature prediction service predicts temperature one hour ahead; it does not directly provide rainfall probabilities, thunderstorm predictions or official weather warnings.
- Prediction quality depends on the supplied weather input values.
- The prediction API may experience delays or cold starts because it is hosted on a free-tier service.
- Some application components require further testing and refinement to improve reliability and response consistency.
- Conversation memory is limited to the recent messages retained during the active session.

## 12. Future Improvements

Potential improvements include:

- Expanding the weather knowledge base with additional scientific and authoritative climate sources.
- Improving retrieval relevance and response accuracy.
- Expanding automated testing of the RAG pipeline and agent workflows.
- Strengthening tool execution and error handling.
- Improving evaluation of generated research responses.
- Introducing more analytical tools for weather and climate research.
- Improving user interaction and the presentation of research findings.
- Exploring additional weather prediction and climate analysis capabilities.

## 13. Conclusion

Part 2 extends the Zimbabwe Weather Intelligence System by introducing Generative AI capabilities for weather research and interaction with the existing prediction service.

Through the integration of Ollama, LangChain, LangGraph, FAISS, RAG and MCP, the application demonstrates how open-source Generative AI technologies can be combined with an existing machine learning prediction system.

The resulting application provides a foundation for research-oriented weather information retrieval, evidence-based question answering and interaction with predictive weather services.

## 14. Repository

**GitHub Repository:**

https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH

**Part 2 Application:**  
The application is run locally using Streamlit, with the Part 1 temperature prediction API accessed through its deployed endpoint.