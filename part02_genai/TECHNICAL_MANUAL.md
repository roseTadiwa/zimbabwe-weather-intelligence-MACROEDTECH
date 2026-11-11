# TECHNICAL MANUAL
## Zimbabwe Weather Intelligence – Part 2: Generative AI Integration

**Project:** Zimbabwe Weather Intelligence  
**Organisation:** MacroEdTech  
**Component:** Part 2 – Generative AI Integration  
**Primary language:** Python  
**User interface:** Streamlit  
**Deployed LLM and embedding service:** Google Gemini API  
**Local development LLM:** Ollama – Llama 3.2:3b  
**Local development embedding model:** Ollama – nomic-embed-text  
**Vector database:** FAISS  

---

# 1. INTRODUCTION

## 1.1 Purpose of the Technical Manual

This manual documents the architecture, implementation, configuration, operation, testing and limitations of Part 2 of the Zimbabwe Weather Intelligence project. It describes the document-based climate research assistant, Retrieval-Augmented Generation (RAG) pipeline, vector knowledge base, agent workflow, Model Context Protocol (MCP) integration and connection to the Part 1 temperature-prediction API.

The manual distinguishes between the original local Ollama implementation and the Gemini-based implementation deployed for browser access. This distinction is necessary because the two configurations use different language and embedding models and therefore require compatible configuration and vector indexes.

## 1.2 Project Overview

Part 1 developed a temperature-prediction model using XGBoost and exposed its prediction functionality through a FastAPI service. Part 2 extends that system by adding a Generative AI research assistant that can retrieve information from Zimbabwe-specific climate and environmental documents.

The application provides two main user-facing capabilities:

- **Temperature Prediction:** Sends supplied weather input values to the Part 1 API and displays the one-hour-ahead temperature prediction.
- **Weather and Climate Research:** Retrieves relevant material from a FAISS knowledge base and uses the configured response workflow to answer document-based questions.

The system also includes an MCP server that exposes tools for checking the prediction API, retrieving supported cities and submitting temperature-prediction requests.

## 1.3 Development and Deployment Configurations

The project went through two model configurations.

### A. Original local implementation – Ollama

The application was initially developed and tested on the developer's local computer using Ollama. This was the original working local implementation, and its code has been retained:

| Component | Local configuration |
|---|---|
| Language model | `llama3.2:3b` |
| Embedding model | `nomic-embed-text` |
| LangChain integration | `ChatOllama` / `OllamaEmbeddings` |
| Vector store | FAISS |
| Interface | Streamlit |

This configuration allowed local model execution without sending language-generation and embedding requests to a hosted Gemini service. The Ollama implementation has been retained separately as part of the development work.

### B. Deployed implementation – Gemini

The application was subsequently adapted to use the Gemini API so that it could be deployed on Streamlit Community Cloud and accessed by a supervisor or reviewer through a web browser. This change was made for practical deployment reasons.

Running Ollama as part of a cloud-hosted application would require the hosting environment to store the model files and provide sufficient server resources to run the model. That introduces additional storage, memory and compute requirements, which can increase hosting infrastructure costs. It would also make the deployment more demanding to configure and maintain.

Using Gemini means the language model and embedding workloads are handled by Google's hosted API rather than requiring us to host the Ollama models alongside the Streamlit application. This makes the deployed application easier to access and avoids the need to provision our own model-hosting infrastructure. It does not mean that Gemini is unlimited or necessarily free: API access is subject to the provider's quotas and any applicable pricing.

A reviewer can therefore open the deployed application in a browser without installing Ollama or downloading the local models. The original Ollama implementation remains available as the local working version.

| Component | Deployed configuration |
|---|---|
| Language model | Gemini model configured in the application (`gemini-3.8-flash`) |
| Embedding model | `models/gemini-embedding-001` |
| LangChain integration | `langchain-google-genai` |
| Vector store | FAISS |
| Interface and hosting | Streamlit / Streamlit Community Cloud |

The deployed version requires a Gemini API key and is subject to the service's availability and quota. It is therefore not a fully local LLM implementation. The Part 1 prediction API is hosted separately on Render.

The two implementations should not be described as one identical runtime configuration. They are two configurations of the same Part 2 development work: the retained Ollama version is the local working implementation, while the Gemini version is the online deployment chosen to avoid the additional storage and compute infrastructure needed to host Ollama models in the cloud.

## 1.4 Objectives

The objectives of Part 2 are to:

1. Develop a Zimbabwe-specific weather, climate and environmental knowledge base.
2. Extract and process text from PDF, Markdown and text documents.
3. Divide source content into smaller overlapping chunks.
4. Generate embeddings and store them in a FAISS vector database.
5. Implement semantic search and Retrieval-Augmented Generation.
6. Integrate a language model through LangChain.
7. organise agent execution and response validation through LangGraph.
8. Implement MCP tools for communication with the Part 1 prediction API.
9. Maintain short-term conversation context during a user session.
10. Provide a Streamlit interface for research and temperature prediction.
11. Test the knowledge-base pipeline, retrieval, API integration and application functionality.

---

# 2. SYSTEM ARCHITECTURE

## 2.1 Architecture Overview

Part 2 has two principal processing paths.

### A. Climate research and document retrieval

1. A user submits a climate or weather research question through Streamlit.
2. The research workflow processes the question.
3. The question is embedded using the embedding model configured for the selected implementation.
4. FAISS retrieves semantically relevant knowledge-base chunks.
5. Retrieved evidence and source metadata are passed to the response workflow.
6. The configured response workflow generates or formats an answer.
7. The answer is returned to the Streamlit interface.

Some selected climate topics use specialised evidence-based response-processing functions rather than relying exclusively on free-form LLM generation.

### B. Temperature prediction

1. A user selects a supported city and supplies the required prediction inputs.
2. The Streamlit prediction interface sends the request to the Part 1 FastAPI endpoint.
3. The Part 1 service processes the input using its XGBoost model.
4. The service returns a one-hour-ahead temperature prediction and associated model information.
5. Streamlit displays the result.

The MCP server provides a separate tool-based route to the same Part 1 API. In the integrated assistant, the `predict_temperature` tool is handled through a Python-controlled MCP workflow. It is not included in the general LangChain agent tool list.

## 2.2 Main System Components

| Component | Responsibility |
|---|---|
| Streamlit | User interface for research and prediction |
| Gemini API | Language generation and embeddings in the deployed configuration |
| Ollama | Local model execution in the retained local configuration |
| Llama 3.2:3b | Local language generation |
| nomic-embed-text | Local text embeddings |
| `models/gemini-embedding-001` | Embeddings for the deployed Gemini knowledge base |
| LangChain | Document processing, retrieval, model and tool integration |
| LangGraph | Agent workflow orchestration |
| FAISS | Vector storage and similarity retrieval |
| MCP / FastMCP | Exposes tools that communicate with the Part 1 API |
| FastAPI | Part 1 prediction API |
| XGBoost | One-hour-ahead temperature prediction |
| Git and GitHub | Version control and source-code management |

## 2.3 Application and Hosting Arrangement

The Part 2 Streamlit application is deployed separately from the Part 1 prediction API.

- **Part 2 Streamlit main file:** `part02_genai/src/weather_assistant_ui.py`
- **Part 1 API base URL:** `https://zimbabwe-weather-intelligence-macroedtech.onrender.com`
- **Part 2 hosting:** Streamlit Community Cloud
- **Part 1 API hosting:** Render

The Gemini API key is configured in Streamlit Community Cloud Secrets under `GEMINI_API_KEY`. It must not be hard-coded into source files or committed to Git.

The Render free-tier service may sleep after inactivity. The first request after a period of inactivity can therefore take longer while the service starts.

---

# 3. TECHNOLOGY STACK

| Technology / library | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| Streamlit | User interface and cloud deployment |
| Google Gemini API | Deployed language model and embedding service |
| Ollama | Local language and embedding model execution during development |
| LangChain | RAG, document processing and tool/model integration |
| LangGraph | Agent workflow orchestration |
| FAISS | Vector storage and similarity search |
| MCP / FastMCP | Tool interface to the Part 1 API |
| FastAPI | Part 1 prediction API |
| XGBoost | Temperature-prediction model |
| pypdf | PDF text extraction |
| Pandas / NumPy | Data and numerical operations where required |
| Git / GitHub | Version control and source-code management |

The Part 2 Python dependencies are listed in `part02_genai/requirements.txt`. A copy is also maintained at `part02_genai/src/requirements.txt` for Streamlit Community Cloud dependency discovery when the app's main file is configured inside `src/`.

---

# 4. PROJECT DIRECTORY STRUCTURE

The main Part 2 files are organised under `part02_genai/`. The relevant high-level structure is:

```text
zimbabwe-weather-intelligence/
├── app/
│   └── app.py
├── part02_genai/
│   ├── documents/
│   │   ├── weather_research_basics.txt
│   │   ├── zimbabwe_climate_risk_country_profile.pdf
│   │   ├── zimbabwe_revised_ndc_2021.pdf
│   │   └── zimbabwe_revised_ndc_evidence.md
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
│   ├── TECHNICAL_MANUAL.md
│   └── README.md
└── README.md
```

This is a high-level representation of the relevant files; local virtual-environment folders and other repository files are omitted. Confirm the current repository tree before treating the list as an exhaustive inventory.

**Directory descriptions**

- `documents/`: Source documents used to construct the climate and weather knowledge base.
- `data/faiss_index_gemini/`: FAISS index generated for the Gemini embedding configuration.
- `src/`: Python scripts for knowledge-base construction, MCP integration, assistant workflows and the Streamlit interface.
- `requirements.txt`: Python dependencies for Part 2.
- `weather_assistant_ui.py`: Main Streamlit interface used by the deployed Part 2 application.

The retained Ollama implementation uses its own model configuration and compatible index. Do not assume that the deployed Gemini index can be used with the Ollama embedding model, or vice versa.

---

# 5. KNOWLEDGE-BASE CONSTRUCTION

## 5.1 Overview

The knowledge base provides access to Zimbabwe-specific climate, weather, environmental and policy information. The source documents are processed, converted into text records, divided into smaller chunks, embedded and stored in FAISS.

## 5.2 Source Documents

The documented source set includes:

| Document | Format | Description |
|---|---|---|
| `weather_research_basics.txt` | TXT | General weather and climate research information |
| `zimbabwe_climate_risk_country_profile.pdf` | PDF | Zimbabwe climate risks, vulnerabilities and sectoral impacts |
| `zimbabwe_revised_ndc_2021.pdf` | PDF | National climate commitments and policy evidence |
| `zimbabwe_revised_ndc_evidence.md` | Markdown | Curated, page-labelled evidence from the NDC document |

The curated NDC evidence document provides additional page-level references for selected statistics and historical climate events.

## 5.3 Document Loading and Text Extraction

Document processing is implemented in:

```text
part02_genai/src/build_knowledge_base.py
```

The processing workflow uses PDF text extraction and LangChain document objects to preserve extracted text and metadata.

The loader supports:

- PDF files (`.pdf`)
- Markdown files (`.md`)
- Text files (`.txt`)

PDF documents are processed page by page. Non-empty pages are retained as individual document records. Page-labelled Markdown sections are separated into records where the expected printed-page headings are present. Other Markdown and text files are loaded as document records.

Metadata supports source identification and may include:

- Source filename.
- File type.
- PDF page number or printed-page reference, where available.
- Page index, where applicable.
- Record type.

## 5.4 Specialised Climate Evidence Processing

The build script includes additional checks for selected evidence in the Nationally Determined Contribution material. The documented verification markers include:

1. Maize production losses associated with the 2016 fall armyworm outbreak.
2. Elephant deaths in Gonarezhou during the 1991/92 drought.
3. Annual crop losses.
4. Cattle deaths associated with tick-borne disease during the 2017/18 period.
5. AFOLU's contribution to greenhouse gas emissions in 2017.
6. A mitigation target.

The script also processes a normalised climate projection table from the climate risk profile. The table records projected median percentage changes in annual precipitation relative to the 1986–2005 baseline. These are climate projections, not observed rainfall measurements.

## 5.5 Text Chunking

After extraction, the text is divided into smaller sections called chunks using LangChain's `RecursiveCharacterTextSplitter`.

The recorded configuration is:

| Parameter | Value |
|---|---:|
| Chunk size | 1,400 characters |
| Chunk overlap | 180 characters |
| Start index | Enabled |

The splitter prioritises paragraph breaks, line breaks, sentence boundaries and spaces before splitting at individual characters where necessary. Overlap helps preserve context between adjacent chunks. The starting character position is retained as metadata.

## 5.6 Embedding Generation

Each text chunk is converted into a numerical vector representation.

Two embedding configurations were used during the project's development:

| Implementation | Embedding model | Integration |
|---|---|---|
| Original local implementation | `nomic-embed-text` | Ollama / LangChain |
| Deployed implementation | `models/gemini-embedding-001` | Gemini API / LangChain Google GenAI |

The same embedding configuration must be used when constructing and querying a given FAISS index. Changing the embedding model requires rebuilding the index so that document vectors and query vectors are compatible.

## 5.7 FAISS Vector Database

FAISS stores the generated vectors and supports similarity-based retrieval. The deployed Gemini index is stored at:

```text
part02_genai/data/faiss_index_gemini/
```

The directory contains the FAISS index and its associated metadata/pickle file, including `index.faiss` and `index.pkl`.

The index can be loaded by the research assistant without repeating document processing, provided the index exists and the embedding configuration is compatible.

The original Ollama implementation used a separate index location/configuration. Keep the indexes distinct.

## 5.8 Rebuilding the Knowledge Base

From the repository root, activate the Part 2 environment and execute:

```cmd
python part02_genai\src\build_knowledge_base.py
```

For the deployed Gemini configuration, ensure that the Gemini API key is available to the process before running the build. Review the script output to confirm document loading, evidence checks, chunk generation, embedding generation, FAISS index creation and retrieval tests.

Rebuild the index after adding or modifying source documents. Do not use an index generated by a different embedding model.

## 5.9 Recorded Knowledge-Base Build Results

The recorded successful build produced:

| Validation item | Recorded result |
|---|---:|
| Source records loaded | 103 |
| PDF text pages extracted | 84 |
| Page-labelled NDC evidence records | 18 verified |
| Text chunks generated | 217 |
| FAISS index | Successfully generated and saved |
| Retrieval smoke tests | 5 out of 5 passed |

The five retrieval checks covered evidence relating to the fall armyworm outbreak, the 1991/92 drought, livestock disease, annual crop losses and AFOLU emissions in 2017.

These results document a successful build and selected retrieval checks. They do not represent a comprehensive evaluation of all possible research questions or generated answers. The model and embedding configuration should be recorded alongside the index used for each build.

---

# 6. RETRIEVAL-AUGMENTED GENERATION

## 6.1 Overview

Retrieval-Augmented Generation combines information retrieval with language-model response generation. The system retrieves relevant passages from the local knowledge base and uses them as contextual evidence when responding to document-based questions.

This is particularly useful for Zimbabwe-specific climate events, national climate statistics, agricultural impacts and policy evidence.

## 6.2 RAG Processing Workflow

1. The user submits a research question.
2. The question is embedded using the configured embedding model.
3. FAISS performs similarity search against the stored document vectors.
4. Relevant document chunks are retrieved.
5. The retrieved text and metadata are passed to the response workflow.
6. The configured response process generates or formats an answer.
7. The response is displayed to the user.

## 6.3 Semantic Search

Semantic search retrieves information based on meaning rather than exact keyword matches. For example, a question about drought effects on agriculture may retrieve material discussing crop losses, livestock disease, rainfall reduction and agricultural vulnerability even where the wording differs.

The knowledge-base build includes focused retrieval smoke tests using similarity search and the top matching documents.

## 6.4 Contextual Response Generation

For general document-research questions, the integrated assistant uses retrieved information as context for a language-model response. The response workflow includes instructions to ground answers in retrieved evidence. The implementation also includes a second-pass evidence-auditing process and citation normalisation.

These mechanisms are intended to improve traceability and reduce unsupported claims; they do not guarantee that every generated answer is correct.

## 6.5 Specialised Research Responses

Selected topics use dedicated response-processing functions rather than relying entirely on unrestricted language-model generation. The documented topics include:

- Historical drought impacts.
- Agricultural challenges.
- AFOLU and greenhouse gas emissions.
- Annual precipitation projections.

For selected drought and agricultural questions, retrieved evidence is organised into structured sections such as long-term trends, documented historical impacts, future projections and agricultural vulnerability. The AFOLU workflow structures evidence about emissions and land-use activities. The precipitation workflow uses the normalised projection table.

## 6.6 Source Traceability

Retrieved chunks retain source metadata, including the original filename and page references where available. This supports identification of the source material behind retrieved evidence. Users should consult the original documents when using findings in formal research or decision-making.

---

# 7. LANGUAGE MODEL INTEGRATION

## 7.1 Original Local Model Configuration

The first implementation used Ollama to run the language and embedding models on the developer's computer.

| Model | Purpose |
|---|---|
| `llama3.2:3b` | Natural-language response generation |
| `nomic-embed-text` | Text embedding generation |

The language model was accessed through LangChain's Ollama integration. The embedding model converted document chunks and user questions into vectors for FAISS.

To prepare a local Ollama environment, the required models can be downloaded using:

```cmd
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

To list locally available models:

```cmd
ollama list
```

This describes the original local implementation. These commands are not prerequisites for the deployed Gemini version.

## 7.2 Deployed Gemini Configuration

The deployed application uses the Gemini API for language generation and embeddings.

| Component | Configured model |
|---|---|
| Chat model | `gemini-3.8-flash` |
| Embedding model | `models/gemini-embedding-001` |

The corresponding LangChain Google GenAI integration is included in the Part 2 dependencies.

The Gemini API key is read from Streamlit secrets under the key `GEMINI_API_KEY`. The application makes the value available to the Google/Gemini environment variables required by the integration.

The key must be kept private. Do not paste it into source code, commit it to GitHub or include it in this manual.

## 7.3 Why the Model Configuration Changed

The original Ollama version was suitable for local development and testing. However, deploying Ollama together with the Streamlit application would require a hosting environment with enough storage for the model files and sufficient memory and compute capacity to run them. Those additional infrastructure requirements could increase hosting costs and make deployment and maintenance more demanding.

The application was therefore adapted to use the Gemini API and deployed on Streamlit Community Cloud. Gemini handles the model workloads through a hosted API, so the project does not need to host the Ollama models itself. This also allows a supervisor or reviewer to access the application through a browser without installing Ollama or downloading models.

This was a practical decision about deployment resources, cost and accessibility; it was not a replacement of the original local work. The Ollama-based implementation and its code have been retained for local use. The deployed Gemini version does, however, depend on internet connectivity, API credentials and the provider's quota and pricing arrangements.

## 7.4 Configuration Compatibility

The Ollama and Gemini versions use different embedding models. A FAISS index built with `nomic-embed-text` must not be assumed to work with `models/gemini-embedding-001`. Rebuild the index with the intended embedding model when changing implementation configurations.

---

# 8. AI AGENTS AND WORKFLOW ORCHESTRATION

## 8.1 Integrated Assistant

The integrated conversational workflow is implemented in:

```text
part02_genai/src/integrated_weather_assistant.py
```

It uses LangChain and LangGraph to coordinate language-model interaction, knowledge retrieval and selected tools.

## 8.2 LangChain

LangChain provides model, retrieval and tool integration. The research tool `search_weather_knowledge` gives the assistant access to the local FAISS knowledge base.

The deployed Gemini configuration uses the Google GenAI integration. The retained local configuration uses the relevant Ollama integration.

## 8.3 LangGraph

LangGraph organises the agent execution workflow. The integrated assistant defines stages for agent execution and response validation. The graph coordinates the workflow and returns the resulting response.

## 8.4 Tool Availability and Prediction Handling

The integrated assistant's general tool list includes the research tool and selected MCP tools, such as API health and supported-city retrieval.

The `predict_temperature` tool is implemented by the MCP server but is excluded from the general LangChain agent tool list. Temperature prediction is handled through a separate Python-controlled MCP workflow. This distinction is important: the existence of a server-side MCP tool does not mean the general conversational agent can freely select it as one of its tools.

---

# 9. MODEL CONTEXT PROTOCOL INTEGRATION

## 9.1 Overview

The Model Context Protocol provides a standardised interface through which an application can interact with tools and external services.

The Part 2 MCP server connects to the Part 1 prediction API. It is implemented in:

```text
part02_genai/src/weather_mcp_server.py
```

The server uses FastMCP and runs with standard input/output (stdio) transport.

## 9.2 MCP Tools

The MCP server exposes three tools:

| Tool | Function |
|---|---|
| `get_weather_api_health` | Checks the prediction API health endpoint |
| `get_available_cities` | Retrieves the cities supported by the prediction service |
| `predict_temperature` | Sends prediction inputs to the Part 1 API and returns its response |

## 9.3 Part 1 API Integration

The MCP server communicates with:

```text
https://zimbabwe-weather-intelligence-macroedtech.onrender.com
```

Relevant API endpoints include:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | API health check |
| `/cities` | GET | Retrieves supported cities |
| `/predict` | POST | Generates a temperature prediction |
| `/docs` | GET | API documentation |

Supported cities:

- Harare
- Bulawayo
- Mutare
- Gweru

## 9.4 MCP Prediction Workflow

When the Python-controlled prediction workflow invokes the MCP prediction tool:

1. The workflow prepares the prediction input.
2. The MCP client connects to the MCP server.
3. A client session is established.
4. The `predict_temperature` tool is called with the required input.
5. The MCP server sends an HTTP request to the FastAPI `/predict` endpoint.
6. The Part 1 service processes the input using the XGBoost model.
7. The result is returned through MCP.
8. The application formats and presents the prediction.

The Streamlit Temperature Prediction interface also calls the FastAPI endpoint directly; that route is separate from the MCP workflow.

## 9.5 Request Handling and Retry Behaviour

The MCP server uses an HTTP request helper with a configured maximum of three attempts and a request timeout of 90 seconds. It retries timeout and connection errors, waiting 10 seconds after the first failed attempt and 20 seconds after the second. Other HTTP errors are raised rather than treated as retryable connection failures.

This retry logic helps handle temporary connection problems and Render cold starts. It does not guarantee that the remote service will become available.

## 9.6 MCP Testing

The recorded tests exercised API health retrieval, supported-city retrieval and the temperature-prediction tool. Successful calls confirmed communication between the MCP server and the deployed prediction API at the time of testing.

A later model-provider quota error during an agent test should be distinguished from an MCP transport failure. Gemini request quota exhaustion can prevent a language-model response even where the MCP server and API are otherwise functioning.

---

# 10. CONVERSATION MEMORY

## 10.1 Overview

Conversation memory allows the assistant to retain recent messages so that follow-up questions can be interpreted in the context of the active interaction.

## 10.2 Implementation

The Streamlit application maintains conversation history using session state. The configured limit is ten messages, equivalent to approximately five user-assistant exchanges.

The recent history is passed to the relevant assistant workflow when processing subsequent questions.

## 10.3 Scope and Limitations

The memory supports follow-up questions and continuity within the active session. It is session-based and is not permanent storage of conversations across separate application sessions.

---

# 11. STREAMLIT USER INTERFACE

## 11.1 Main Application

The Part 2 Streamlit application is implemented in:

```text
part02_genai/src/weather_assistant_ui.py
```

It is the configured main file for the Part 2 Streamlit Community Cloud deployment.

## 11.2 Main Interface Areas

The application provides the following areas:

### A. Research Assistant

Users can submit natural-language questions relating to weather, climate and environmental topics. The research workflow retrieves information from the knowledge base and uses the configured response process.

Documented areas of use include historical drought impacts, agricultural impacts, AFOLU and emissions, and annual precipitation projections.

### B. Temperature Prediction

Users can select a supported city, provide the required weather input variables, submit a request and view the predicted temperature and associated model information.

The Streamlit prediction interface communicates directly with the FastAPI `/predict` endpoint using Python's HTTP request functionality. This is separate from the MCP prediction workflow.

### C. About

The interface includes an About area providing information about the application.

## 11.3 Session Handling and Caching

Streamlit session state maintains recent conversation history. The application also uses caching for selected research resources and agent-related components to reduce unnecessary reloading during normal use.

---

# 12. CONFIGURATION AND APPLICATION EXECUTION

## 12.1 Prerequisites for the Deployed Gemini Configuration

- Python 3.12.
- Git.
- Project source code and required documents.
- Part 2 dependencies.
- A Gemini API key available through Streamlit secrets or the local environment.
- Internet access to reach Gemini and the Part 1 prediction API.

Ollama is not required to run the deployed Gemini configuration.

## 12.2 Clone the Repository

```cmd
git clone https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH.git
cd zimbabwe-weather-intelligence-MACROEDTECH
```

## 12.3 Create and Activate the Virtual Environment

```cmd
python -m venv part02_genai\venv
part02_genai\venv\Scripts\activate
```

## 12.4 Install Dependencies

```cmd
pip install -r part02_genai\requirements.txt
```

The `part02_genai/src/requirements.txt` copy is maintained for Streamlit Community Cloud dependency installation.

## 12.5 Configure the Gemini API Key Locally

For local Streamlit execution, the key can be configured in a local `.streamlit/secrets.toml` file:

```toml
GEMINI_API_KEY = "your-key"
```

Do not commit this file. The actual key must remain private. In Streamlit Community Cloud, configure the same secret name in the application's Secrets settings.

The application includes a secrets bridge that reads `GEMINI_API_KEY` from `st.secrets` and makes it available to the environment variables used by the Gemini integration.

## 12.6 Build the Knowledge Base

```cmd
python part02_genai\src\build_knowledge_base.py
```

Use the Gemini key and Gemini embedding configuration for the deployed index. Confirm the expected Gemini FAISS index is generated before launching the research assistant.

## 12.7 Run the Streamlit Application Locally

```cmd
python -m streamlit run part02_genai\src\weather_assistant_ui.py
```

The application will display a local browser address. The Gemini configuration requires the API key, network access and the compatible FAISS index.

The `python -m streamlit` form is used in the Windows environment because direct execution of `streamlit.exe` may be blocked by local Device Guard/security restrictions.

## 12.8 Original Ollama Local Configuration

The original local implementation requires Ollama and its model files:

```cmd
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

Run the corresponding retained Ollama implementation with its own configuration and compatible FAISS index. Do not assume the current Gemini-based code and index can be switched to Ollama simply by installing the Ollama models; the model integrations, embedding configuration and index must match.

## 12.9 Running the Integrated Assistant Separately

The integrated assistant source file is:

```text
part02_genai/src/integrated_weather_assistant.py
```

It can be run separately for command-line or integration testing where its configured dependencies, credentials, knowledge base and MCP server requirements are available:

```cmd
python part02_genai\src\integrated_weather_assistant.py
```

The primary reviewer-facing interface is the Streamlit application. The standalone assistant script should be treated as a development/testing entry point, not as the deployed web interface.

---

# 13. DEPLOYMENT

## 13.1 Part 2 Streamlit Community Cloud

The Part 2 application was deployed separately to Streamlit Community Cloud.

Configured main file:

```text
part02_genai/src/weather_assistant_ui.py
```

The Gemini API key is stored in Streamlit Cloud Secrets under:

```text
GEMINI_API_KEY
```

The secret is read at runtime and is not intended to be committed to the repository.

The public Part 2 application URL should be recorded in the project README or deployment record once confirmed. Do not substitute the Part 1 application URL.

## 13.2 Dependency File Placement

The project maintains:

```text
part02_genai/requirements.txt
part02_genai/src/requirements.txt
```

The second file is a copy used to ensure Streamlit Community Cloud detects the dependencies when the main application file is inside `src/`.

## 13.3 Part 1 API Hosting

The Part 1 FastAPI prediction service is hosted at:

```text
https://zimbabwe-weather-intelligence-macroedtech.onrender.com
```

It is separate from the Part 2 Streamlit deployment. The Render free-tier service may sleep after inactivity and can therefore delay the first request.

---

# 14. TESTING AND EVALUATION

## 14.1 Testing Objectives

Testing activities focused on:

- Document loading and extraction.
- Knowledge-base construction.
- NDC evidence verification.
- Chunking and embedding generation.
- FAISS index creation.
- Semantic retrieval.
- Specialised and general research responses.
- Prediction API connectivity.
- MCP tool integration.
- Streamlit interface functionality.
- Conversation-history handling.
- Deployment access and operation.

## 14.2 Knowledge-Base Testing

The knowledge-base construction script includes five focused retrieval smoke tests. All five recorded tests passed. They checked whether expected evidence could be retrieved for selected NDC-related questions.

These tests validate retrieval presence for the selected queries, not the factual correctness of every final answer.

## 14.3 Research Response Testing

Specialised research workflows were tested with questions relating to agricultural challenges, the 1991/92 drought, water availability and future projections, and AFOLU's contribution to greenhouse gas emissions.

The specialised response functions returned expected evidence for the tested questions. The general document-research workflow requires continued evaluation because generated responses may occasionally be incomplete or inaccurate.

## 14.4 Prediction API Testing

The deployed FastAPI service was tested through its health, supported-city and prediction endpoints. The Streamlit Temperature Prediction interface was also tested against the deployed API.

## 14.5 MCP Testing

The MCP health-check and city-retrieval tools were tested. The `predict_temperature` tool was also successfully executed in recorded integration testing and returned a one-hour-ahead prediction.

A later Gemini free-tier quota-exhaustion error during an agent interaction is a model-provider limitation. It does not, on its own, demonstrate that MCP transport or the prediction API is broken.

## 14.6 Streamlit Application Testing

The local Streamlit application was launched and tested. The deployed Streamlit application was also made available through Streamlit Community Cloud. The research assistant and direct temperature-prediction interface were tested in the deployed environment.

## 14.7 Conversation Memory Testing

The application maintains recent conversation messages through Streamlit session state. The configured history limit is ten messages, equivalent to approximately five user-assistant exchanges.

## 14.8 Testing Summary

| Component | Recorded outcome |
|---|---|
| Document extraction | Successful |
| NDC evidence verification | Successful |
| Text chunking | Successful |
| Embedding generation | Successful |
| FAISS index creation | Successful |
| Retrieval smoke tests | 5/5 passed |
| Specialised climate research | Tested successfully |
| General document research | Requires further refinement |
| Part 1 prediction API integration | Tested successfully |
| MCP health and city tools | Tested successfully |
| MCP prediction tool | Tested successfully in recorded integration testing |
| Local Streamlit interface | Launched and tested |
| Deployed Streamlit interface | Deployed and tested |
| Conversation history | Implemented |

---

# 15. TROUBLESHOOTING

## 15.1 Gemini API Key Not Found

**Possible cause:** The `GEMINI_API_KEY` secret is missing, incorrectly named or unavailable to the running application.

**Resolution:**

1. Confirm that the secret is configured in Streamlit Community Cloud Secrets or local Streamlit secrets.
2. Confirm that the name is exactly `GEMINI_API_KEY`.
3. Restart or rerun the application after correcting the secret.
4. Do not print or share the secret value when debugging.

## 15.2 Gemini API Quota or Rate Limit Error

**Possible cause:** The configured Gemini model has reached its request quota or rate limit.

**Resolution:**

1. Check the error details and quota reset information.
2. Wait until the applicable quota becomes available or review the provider's available quota options.
3. Avoid repeated test requests while the quota is exhausted.
4. Distinguish a model-provider quota error from a local code, FAISS or MCP transport error.

## 15.3 FAISS Index Not Found

**Possible cause:** The knowledge base has not been built, the index files are missing, or the application is looking in a different directory.

**Resolution:**

```cmd
python part02_genai\src\build_knowledge_base.py
```

Review the output and confirm that the index is saved in the expected directory. For the deployed Gemini configuration, the index path is:

```text
part02_genai/data/faiss_index_gemini/
```

## 15.4 Embedding Dimension or Compatibility Error

**Possible cause:** The FAISS index was built with a different embedding model from the one currently used to embed queries.

**Resolution:** Rebuild the index using the embedding model configured for the application. Do not mix the Ollama `nomic-embed-text` index with the Gemini `models/gemini-embedding-001` query configuration.

## 15.5 Prediction API Connection Error

**Possible cause:** The Render-hosted API is unavailable, starting from sleep or experiencing a connection delay.

**Resolution:** Check:

```text
https://zimbabwe-weather-intelligence-macroedtech.onrender.com/health
```

If the service is starting or temporarily unavailable, retry after a short interval. The first request after inactivity may take longer.

## 15.6 MCP Connection Error

**Possible cause:** The MCP server did not start, dependencies are missing, or the client could not establish a session.

**Resolution:**

1. Confirm that `weather_mcp_server.py` exists.
2. Confirm that the Part 2 dependencies are installed.
3. Confirm that the Part 1 API is reachable.
4. Review the terminal output for connection and execution errors.
5. Retry after resolving any reported startup or connectivity problem.

## 15.7 Ollama Model Not Found – Local Version Only

**Possible cause:** The required local model has not been downloaded or Ollama is not running.

**Resolution:**

```cmd
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

This troubleshooting step applies only to the retained local Ollama implementation. It is not required for the deployed Gemini version.

## 15.8 Streamlit Application Error

**Possible causes:** Missing dependencies, unavailable knowledge base, missing Gemini secret, incompatible FAISS index or configuration error.

**Resolution:**

1. Confirm that the intended virtual environment is activated.
2. Confirm that dependencies are installed from the Part 2 requirements file.
3. Confirm that `GEMINI_API_KEY` is configured for the Gemini implementation.
4. Confirm that the FAISS index exists and matches the configured embedding model.
5. Confirm that the Part 1 API is reachable if testing prediction functionality.
6. Review the application and terminal error messages.

## 15.9 PDF Extraction Warnings

Some PDF files may produce font-parsing warnings. A warning does not necessarily prevent text extraction or index construction. If extraction or encoding problems occur, inspect the extracted text and install optional font parsing support if required by the error message.

---

# 16. SYSTEM LIMITATIONS

## 16.1 Knowledge-Base Coverage

The knowledge base relies on a limited collection of Zimbabwe-specific climate and weather documents. Its ability to answer questions is constrained by the scope, quality and availability of the source material.

## 16.2 Language Model Accuracy

Language models can generate incomplete or inaccurate interpretations. Retrieval and evidence-grounding instructions reduce some risks but do not guarantee correctness. Research outputs should be checked against original sources, especially for formal reporting or decision-making.

## 16.3 Retrieval Limitations

Retrieval quality depends on document extraction, chunking, embeddings and similarity ranking. Relevant information may not always appear in the retrieved results.

## 16.4 Gemini API Dependency

The deployed research assistant depends on the Gemini API, a valid API key, network connectivity and available request quota. API usage may be subject to free-tier limits or applicable charges. The deployment avoids the need to host Ollama model files and provide the associated server storage, memory and compute resources, but it is not independent of hosted model services.

## 16.5 Hosted API Availability

The Part 1 prediction API is hosted on a free-tier service and may experience cold starts or temporary delays.

## 16.6 Prediction Scope

The existing prediction model provides a one-hour-ahead temperature prediction for Harare, Bulawayo, Mutare and Gweru. It does not currently provide a complete multi-day forecast.

## 16.7 Conversation Memory

Conversation history is maintained within the active Streamlit session and is not permanent storage across separate sessions.

## 16.8 Differences Between Local and Deployed Implementations

The local Ollama and deployed Gemini configurations use different language and embedding models. Their indexes and runtime requirements must be kept separate. The deployed version also depends on hosted Gemini and Streamlit services.

---

# 17. FUTURE IMPROVEMENTS

Potential improvements include:

1. **Expand the knowledge base:** Add more Zimbabwe-specific climate, weather, agricultural and environmental research.
2. **Improve retrieval quality:** Evaluate alternative chunking strategies, retrieval parameters and ranking methods.
3. **Strengthen response evaluation:** Develop a broader test dataset to assess factual accuracy, evidence relevance and response consistency.
4. **Improve source citations:** Provide consistent document and page references in generated research responses.
5. **Enhance conversation memory:** Explore persistent conversation storage where appropriate.
6. **Improve the user experience:** Refine the Streamlit interface and introduce additional relevant visualisations.
7. **Expand prediction capabilities:** Explore additional forecast horizons and weather variables.
8. **Improve service reliability:** Consider more reliable API hosting, quota monitoring and service monitoring.
9. **Document reproducibility:** Maintain clear instructions for reproducing both the local Ollama configuration and the deployed Gemini configuration.

---

# 18. CONCLUSION

Part 2 extends the Zimbabwe Weather Intelligence project beyond temperature prediction by introducing document-based climate research and Generative AI capabilities. The implementation includes a Zimbabwe-specific knowledge base, document extraction and chunking, embeddings, FAISS retrieval, RAG, LangChain, LangGraph, MCP integration and a Streamlit interface connected to the Part 1 prediction API.

The project was initially developed and tested using locally hosted Ollama models, and that working local implementation and its code have been retained. The application was subsequently adapted to use Gemini for deployment on Streamlit Community Cloud. Hosting Ollama in the cloud would have required additional storage for the model files and sufficient server memory and compute resources to run them, increasing infrastructure requirements and potentially hosting costs. Gemini allowed the model workloads to be handled through a hosted API, making the application accessible to a supervisor or reviewer through a browser without requiring Ollama to be installed or hosted by the project. The deployed version remains subject to Gemini API credentials, quotas and any applicable usage charges.

The recorded knowledge-base build generated 217 chunks and passed all five focused retrieval smoke tests. The prediction API and MCP integration were also tested successfully. Further work is appropriate in general research-response evaluation, retrieval quality, knowledge-base coverage and hosted-service reliability.
