# TECHNICAL MANUAL
## Zimbabwe Weather Intelligence – Part 2: Generative AI Integration

**Project:** Zimbabwe Weather Intelligence  
**Organisation:** MacroEdTech  
**Component:** Part 2 – Generative AI Integration  
**Primary Language:** Python  
**Local LLM:** Ollama – Llama 3.2:3b  
**Embedding Model:** Ollama – nomic-embed-text  
**Vector Database:** FAISS  
**User Interface:** Streamlit  

---

# 1. INTRODUCTION

## 1.1 Purpose of the Technical Manual

This technical manual documents the implementation, architecture, configuration and operation of Part 2 of the Zimbabwe Weather Intelligence project.

Part 2 extends the weather prediction functionality developed in Part 1 by integrating Generative AI, Retrieval-Augmented Generation (RAG), local language models, document-based climate research, AI agents and the Model Context Protocol (MCP).

The manual provides technical information on the system components, data processing procedures, knowledge-base construction, model integration, application execution, testing and known limitations.

It is intended to support the understanding, operation, maintenance and further development of the application.

## 1.2 Project Overview

The Zimbabwe Weather Intelligence project combines machine learning-based weather prediction with Generative AI capabilities to provide weather forecasts and contextual climate information for Zimbabwe.

Part 1 developed a temperature prediction model using XGBoost and exposed its prediction functionality through a FastAPI application.

Part 2 integrates this existing prediction service with a Generative AI assistant capable of:

- Responding to general weather-related questions.
- Retrieving information from Zimbabwe-specific climate and weather documents.
- Providing contextual explanations of historical climate events and agricultural impacts.
- Accessing weather prediction functionality through MCP tools.
- Maintaining conversation history during an interaction.
- Presenting weather predictions and research responses through a Streamlit interface.

The application uses locally hosted language and embedding models through Ollama, reducing dependence on externally hosted LLM services.

## 1.3 Objectives

The main objectives of Part 2 are to:

1. Develop a Zimbabwe-specific climate and weather knowledge base.
2. Implement document processing, text extraction and chunking.
3. Generate text embeddings and store them in a FAISS vector database.
4. Implement semantic search and Retrieval-Augmented Generation.
5. Integrate local Large Language Models using Ollama.
6. Develop AI agent workflows using LangChain and LangGraph.
7. Implement MCP tools to connect the assistant to the existing weather prediction API.
8. Maintain conversational context during user interactions.
9. Integrate the different capabilities into a Streamlit application.
10. Evaluate the system's retrieval, prediction integration and conversational functionality.

---

# 2. SYSTEM ARCHITECTURE

## 2.1 Architecture Overview

The system consists of several interconnected components that work together to provide weather prediction and climate research capabilities.

The architecture follows two main processing paths:

**A. Weather prediction and tool integration**

The user submits a temperature prediction request. The Streamlit prediction interface sends the required input directly to the Part 1 FastAPI service. The integrated AI assistant can also access the same prediction service through an MCP tool.

**B. Climate research and document retrieval**

The user submits a climate or weather research question. The system retrieves relevant information from the local FAISS knowledge base. The retrieved information is used to generate a contextual response, with specialised handling for selected climate research topics.

## 2.2 Main System Components

| Component | Responsibility |
|---|---|
| Streamlit | Provides the user interface |
| Ollama | Hosts local language and embedding models |
| Llama 3.2:3b | Generates natural-language responses |
| nomic-embed-text | Converts text into vector embeddings |
| LangChain | Supports document processing, retrieval and tool integration |
| LangGraph | Orchestrates AI agent workflows |
| FAISS | Stores and retrieves document embeddings |
| MCP | Connects the assistant to external tools and services |
| FastAPI | Exposes the Part 1 temperature prediction API |
| XGBoost | Generates one-hour-ahead temperature predictions |

## 2.3 System Workflow

### Document Research Workflow

1. The user submits a research question.
2. The application identifies the type of question.
3. Relevant documents are retrieved from the FAISS knowledge base.
4. Retrieved evidence is passed to the appropriate response-processing function.
5. The local language model generates responses for the general document-research workflow.
6. Selected specialised questions use deterministic, evidence-based response formatting.
7. The answer is displayed in Streamlit.

### Temperature Prediction Workflow

1. The user selects a supported city.
2. The user supplies the required weather input variables.
3. The application sends the input to the Part 1 prediction API.
4. The XGBoost model generates a one-hour-ahead temperature prediction.
5. The API returns the prediction and associated model information.
6. The application displays the result.

The integrated assistant can also invoke the prediction service through the MCP `predict_temperature` tool.

---

# 3. TECHNOLOGY STACK

The following technologies and libraries were used in the implementation.

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Ollama | Local model execution |
| Llama 3.2:3b | Language generation |
| nomic-embed-text | Text embedding generation |
| LangChain | RAG, document processing and tool integration |
| LangGraph | Agent workflow orchestration |
| FAISS | Vector storage and similarity search |
| MCP / FastMCP | Tool and service integration |
| Streamlit | Interactive application interface |
| FastAPI | Part 1 prediction API |
| XGBoost | Temperature prediction model |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| pypdf | PDF text extraction |
| Git and GitHub | Version control and source-code management |

The application uses locally hosted open-weight models through Ollama.

---

# 4. PROJECT DIRECTORY STRUCTURE

The Part 2 implementation is organised within the main project directory.

The relevant directory structure is:

```text
zimbabwe-weather-intelligence/
│
├── part01_weather_prediction/
│
├── part02_genai/
│   │
│   ├── documents/
│   │   ├── weather_research_basics.txt
│   │   ├── zimbabwe_climate_risk_country_profile.pdf
│   │   ├── zimbabwe_revised_ndc_2021.pdf
│   │   └── zimbabwe_revised_ndc_evidence.md
│   │
│   ├── data/
│   │   └── faiss_index/
│   │
│   ├── src/
│   │   ├── build_knowledge_base.py
│   │   ├── weather_mcp_server.py
│   │   ├── integrated_weather_assistant.py
│   │   └── weather_assistant_ui.py
│   │
│   ├── venv/
│   │
│   ├── README.md
│   ├── TECHNICAL_MANUAL.md
│   └── requirements.txt
│
└── README.md
```

**Directory descriptions:**

- `documents/`: Contains the source documents used to construct the climate and weather knowledge base.
- `data/faiss_index/`: Stores the generated FAISS vector index.
- `src/`: Contains the Python scripts responsible for document processing, MCP integration, AI agent workflows and the Streamlit interface.
- `venv/`: Contains the project's Python virtual environment.
- `README.md`: Provides the project overview, setup instructions and general usage information.
- `requirements.txt`: Lists the Python dependencies required by the project.

---

# 5. KNOWLEDGE BASE CONSTRUCTION

## 5.1 Overview

A domain-specific knowledge base was developed to provide the assistant with access to Zimbabwe-specific weather, climate and environmental information.

The knowledge base consists of locally stored documents containing climate research, historical climate events, agricultural impacts, vulnerability information and national climate policy evidence.

The source documents are processed, converted into text records, divided into smaller chunks, embedded and stored in a FAISS vector database.

## 5.2 Source Documents

The following documents were used:

| Document | Format | Description |
|---|---|---|
| `weather_research_basics.txt` | TXT | General weather and climate research information |
| `zimbabwe_climate_risk_country_profile.pdf` | PDF | Zimbabwe's climate risks, vulnerabilities and sectoral impacts |
| `zimbabwe_revised_ndc_2021.pdf` | PDF | Zimbabwe's Nationally Determined Contribution and climate commitments |
| `zimbabwe_revised_ndc_evidence.md` | Markdown | Curated, page-labelled climate evidence extracted from the NDC document |

The curated NDC evidence document provides additional page-level references for important statistics and historical climate events.

## 5.3 Document Loading and Text Extraction

Document processing is implemented in:

`part02_genai/src/build_knowledge_base.py`

The script uses:

- `pypdf` to extract text from PDF documents.
- `pathlib` to locate and manage document paths.
- LangChain `Document` objects to store extracted text and associated metadata.

The document loader supports:

- PDF files (`.pdf`)
- Markdown files (`.md`)
- Text files (`.txt`)

PDF documents are processed page by page. Each non-empty page is stored as a separate document record.

Markdown documents containing headings such as `## Printed page(s) 10–11` are separated into page-labelled records. Other Markdown documents are treated as a single record.

Text files are loaded as individual document records.

Each record contains relevant metadata, including:

- Source filename.
- File type.
- Page number or printed-page reference, where available.
- Page index, where applicable.
- Record type.

This metadata supports source identification and document traceability during retrieval.

## 5.4 Specialised Climate Evidence Processing

The script includes additional processing for selected information in the Nationally Determined Contribution document.

### A. Page-labelled NDC evidence verification

The script checks that the curated NDC evidence records exist and verifies six required evidence markers:

1. 263,000 tonnes of maize production losses associated with the 2016 fall armyworm outbreak.
2. 1,500 elephant deaths in Gonarezhou during the 1991/92 drought.
3. Annual crop losses of US$126 million.
4. More than 3,000 cattle deaths associated with tick-borne disease during the 2017/18 period.
5. AFOLU's 54% contribution to greenhouse gas emissions in 2017.
6. A 40% mitigation target.

These checks help ensure that important climate evidence is available before the knowledge base is constructed.

### B. Normalised climate projection table

The script also extracts and verifies the relevant page of the climate risk profile and adds a normalised version of Table 3 to the knowledge base.

The table contains projected median percentage changes in annual precipitation compared with the 1986–2005 baseline.

| Period | RCP2.6 | RCP4.5 | RCP6.0 | RCP8.5 |
|---|---:|---:|---:|---:|
| 2040–2059 | -1.2% | +1.2% | +4.4% | -4.4% |
| 2080–2099 | +2.8% | +4.1% | +3.0% | -10.7% |

These values are climate projections, not observed rainfall measurements or changes expressed in millimetres.

The normalised table improves retrieval of structured climate projection information.

## 5.5 Text Chunking

After extraction, the documents are divided into smaller text sections known as chunks.

The implementation uses LangChain's `RecursiveCharacterTextSplitter`.

The configured parameters are:

| Parameter | Value |
|---|---:|
| Chunk size | 1,400 characters |
| Chunk overlap | 180 characters |
| Start index | Enabled |

The splitter uses the following separator order:

1. Paragraph breaks.
2. Line breaks.
3. Sentence boundaries.
4. Spaces.
5. Individual characters, where necessary.

The overlap helps preserve context between adjacent chunks and reduces the likelihood of important information being separated at chunk boundaries.

The starting character position is also retained as metadata.

## 5.6 Embedding Generation

After chunking, each text section is converted into a numerical vector representation.

The implementation uses Ollama's `nomic-embed-text` embedding model through LangChain's `OllamaEmbeddings` integration.

Embeddings represent the semantic content of text and allow the system to identify documents that are conceptually related to a user's question.

## 5.7 FAISS Vector Database

The generated embeddings are stored in a local FAISS vector index.

FAISS supports similarity-based retrieval by identifying document vectors that are closest to the embedding of a user's query.

The index is saved at:

`part02_genai/data/faiss_index/`

The script uses LangChain's FAISS vector-store integration to construct and save the index.

The index can subsequently be loaded by the research assistant without repeating the full document-processing procedure, provided the saved index is available and the embedding configuration remains compatible.

---

## 5.8 Knowledge Base Construction Procedure

To rebuild the knowledge base:

**Step 1: Open Command Prompt and navigate to the project root.**

```cmd
cd "%USERPROFILE%\Desktop\ROSE-TADIWA\MACROEDTECH\WEEK5_8Task\zimbabwe-weather-intelligence"
```

**Step 2: Activate the Part 2 virtual environment.**

```cmd
part02_genai\venv\Scripts\activate
```

**Step 3: Execute the knowledge-base construction script.**

```cmd
python part02_genai\src\build_knowledge_base.py
```

**Step 4: Review the execution output.**

The script reports document-loading results, verification checks, chunk generation, embedding generation, FAISS index creation and retrieval smoke-test results.

The index is saved locally for subsequent use by the assistant.

## 5.9 Knowledge Base Build Results

The knowledge-base construction script was executed successfully.

The recorded results were:

| Validation item | Result |
|---|---|
| Source records loaded | 103 |
| PDF text pages extracted | 84 |
| Normalised climate projection table | Successfully added |
| Page-labelled NDC evidence records | 18 verified |
| Text chunks generated | 217 |
| Embedding model | nomic-embed-text |
| FAISS index | Successfully generated and saved |
| Retrieval smoke tests | 5 out of 5 passed |

The five retrieval smoke tests successfully located expected evidence relating to:

1. The 2016 fall armyworm outbreak and maize production losses.
2. Elephant deaths during the 1991/92 drought.
3. Livestock disease during the 2015/16 drought.
4. Zimbabwe's annual crop losses.
5. The AFOLU contribution to greenhouse gas emissions in 2017.

The generated index was saved in:

`part02_genai/data/faiss_index/`

These results confirm that document loading, extraction, preprocessing, chunking, embedding generation and vector-index construction executed successfully.

The five retrieval checks also confirmed that the expected evidence could be found among the retrieved documents.

---

# 6. RETRIEVAL-AUGMENTED GENERATION (RAG)

## 6.1 Overview

Retrieval-Augmented Generation is used to provide the language model with relevant information retrieved from the project's local knowledge base.

Rather than relying entirely on information learned during model training, the system retrieves relevant document sections and uses them as contextual evidence when generating answers.

This approach is particularly important for questions concerning Zimbabwe-specific climate events, national climate statistics and agricultural impacts.

## 6.2 RAG Processing Workflow

The RAG process follows these steps:

1. A user submits a climate or weather research question.
2. The question is converted into an embedding using the configured embedding model.
3. FAISS performs a similarity search against the stored document vectors.
4. The most relevant document chunks are retrieved.
5. Retrieved text is passed to the response-generation workflow.
6. The language model uses the retrieved information to formulate an answer.
7. The response is returned to the user.

## 6.3 Semantic Search

Semantic search allows the system to retrieve information based on meaning rather than relying only on exact keyword matches.

For example, a question about the effects of drought on agriculture may retrieve documents discussing crop losses, livestock disease, reduced rainfall and agricultural vulnerability, even when the exact wording differs.

The knowledge-base construction script performs focused retrieval checks using `similarity_search()` and retrieves the top five matching documents for each test query.

## 6.4 Contextual Response Generation

For general document research questions, the integrated assistant uses the retrieved information as context for a language-model response.

The implementation includes a grounded-response prompt that instructs the model to base its answers on the available evidence.

A second-pass evidence-auditing process is also used to review generated research responses, followed by citation normalisation.

This helps improve the reliability and traceability of research answers.

## 6.5 Specialised Research Responses

Selected research topics use dedicated response-processing functions rather than relying entirely on unrestricted language-model generation.

These include:

- Historical drought impacts.
- Agricultural challenges.
- AFOLU and greenhouse gas emissions.
- Annual precipitation projections.

For drought and agricultural questions, the system retrieves relevant evidence and uses a deterministic response formatter to organise the findings under headings such as:

- Long-term Agricultural Trends.
- Documented Historical Impacts.
- Future Projections.
- Agricultural Vulnerability.

The AFOLU workflow also uses retrieved evidence to provide a structured response about emissions and land-use-related activities.

The precipitation projection workflow uses the normalised climate table to present the relevant projection values.

This specialised handling helps reduce unsupported claims and improves consistency for selected research questions.

## 6.6 Source Traceability

Retrieved document chunks retain their source metadata.

Where available, this includes the original filename, PDF page number, printed-page reference and other relevant document identifiers.

These details support the identification of the underlying evidence used in a response.

---

# 7. LARGE LANGUAGE MODEL INTEGRATION

## 7.1 Ollama

Ollama is used to run the language and embedding models locally.

This enables the application to access its AI capabilities without depending on a remotely hosted commercial LLM service.

Two models are used:

| Model | Purpose |
|---|---|
| `llama3.2:3b` | Natural-language response generation |
| `nomic-embed-text` | Text embedding generation |

## 7.2 Llama 3.2:3b

The Llama 3.2:3b model is used for generating natural-language responses in the research and conversational workflows.

It is accessed through LangChain's `ChatOllama` integration.

The model is used to:

- Interpret user questions.
- Generate contextual responses.
- Support document-based research.
- Participate in AI agent workflows.

## 7.3 Embedding Model

The `nomic-embed-text` model converts document chunks and user questions into vector representations.

The same embedding model configuration is used when constructing and querying the FAISS index.

This ensures compatibility between stored document embeddings and query embeddings.

## 7.4 Model Availability

Before running the application, ensure that Ollama is installed and that both required models are available locally.

The models can be obtained using:

```cmd
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

To confirm the installed models:

```cmd
ollama list
```

---

# 8. AI AGENTS AND WORKFLOW ORCHESTRATION

## 8.1 AI Agent Implementation

The project uses LangChain to create an AI agent that can interact with available tools.

The agent can interpret user requests, determine which tools are relevant and use the available functionality to respond.

The integrated assistant is implemented in:

`part02_genai/src/integrated_weather_assistant.py`

## 8.2 LangChain

LangChain provides the integration between the language model, retrieval functions and external tools.

It is used for:

- Connecting to Ollama models.
- Loading the FAISS knowledge base.
- Defining the document-research tool.
- Connecting to MCP tools.
- Creating the AI agent.

A research tool named `search_weather_knowledge` is defined to allow the assistant to access the local weather and climate knowledge base.

## 8.3 LangGraph

LangGraph is used to organise the agent's execution workflow.

The integrated assistant defines a graph containing two main stages:

1. **Agent execution:** The agent processes the user's messages and invokes relevant tools.
2. **Response validation:** The resulting response is passed through a validation stage before the workflow ends.

The graph uses an asynchronous agent invocation and is responsible for coordinating the execution of these stages.

## 8.4 Tool Selection

The agent has access to selected MCP tools and the document-research tool.

The `predict_temperature` tool is excluded from the general LangChain agent tool list because prediction requests are handled through a dedicated Python-controlled MCP workflow.

This separation provides more control over the prediction input and response handling.

---

# 9. MODEL CONTEXT PROTOCOL (MCP) INTEGRATION

## 9.1 Overview

The Model Context Protocol (MCP) provides a standardised way for AI applications to interact with external tools and services.

In this project, MCP connects the Generative AI assistant to the existing Part 1 weather prediction API.

The MCP server is implemented in:

`part02_genai/src/weather_mcp_server.py`

The implementation uses FastMCP and communicates with the assistant through standard input/output (stdio) transport.

## 9.2 MCP Tools

The MCP server exposes three tools:

| Tool | Function |
|---|---|
| `get_weather_api_health` | Checks the availability and health of the prediction API |
| `get_available_cities` | Retrieves the supported prediction cities |
| `predict_temperature` | Sends prediction inputs to the Part 1 API and returns the forecast |

The tools communicate with the FastAPI service through HTTP requests.

## 9.3 API Integration

The MCP server connects to the deployed Part 1 prediction API:

`https://zimbabwe-weather-intelligence-macroedtech.onrender.com`

The relevant endpoints include:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API root |
| `/health` | GET | API health check |
| `/cities` | GET | Retrieves supported cities |
| `/predict` | POST | Generates a temperature prediction |
| `/docs` | GET | API documentation |

The supported cities are:

- Harare
- Bulawayo
- Mutare
- Gweru

## 9.4 Prediction Tool Workflow

When the integrated assistant needs to access prediction functionality:

1. The assistant identifies the need for a temperature prediction.
2. The Python-controlled MCP client starts a connection to the MCP server.
3. A client session is established.
4. The `predict_temperature` tool is called with the required input.
5. The MCP server sends the request to the FastAPI `/predict` endpoint.
6. The prediction API processes the input using the XGBoost model.
7. The prediction result is returned through MCP.
8. The assistant presents the result to the user.

## 9.5 MCP Integration Testing

The MCP integration was tested using the available tools.

The city retrieval function returned the supported cities, while the health-check function confirmed that the prediction API was operational.

The temperature prediction tool was also successfully executed and returned a one-hour-ahead forecast with model information.

The successful tool calls confirmed that the MCP server could communicate with the deployed prediction API.

The prediction test used supplied weather input values and was intended to verify integration functionality. It should not be interpreted as a complete assessment of real-world forecast accuracy.

---

# 10. CONVERSATION MEMORY

## 10.1 Overview

Conversation memory allows the assistant to retain recent messages during an interaction.

This helps maintain context when users ask follow-up questions without repeating all the details from earlier messages.

## 10.2 Implementation

The application maintains conversation history using Streamlit session state.

The conversation history is limited to the most recent ten messages, equivalent to approximately five user-assistant exchanges.

The history is passed into the relevant assistant workflow so that subsequent questions can be interpreted in context.

## 10.3 Purpose

Conversation memory supports:

- Follow-up questions.
- Contextual responses.
- Continuity across a sequence of related questions.
- Reduced repetition by the user.

The memory is session-based and is not implemented as permanent storage of conversations across separate application sessions.

---

# 11. STREAMLIT USER INTERFACE

## 11.1 Overview

Streamlit provides the interactive interface through which users access the weather prediction and climate research capabilities.

The main application is implemented in:

`part02_genai/src/weather_assistant_ui.py`

The interface brings together the existing prediction service and the Generative AI research assistant.

## 11.2 Main Application Features

The application provides two main functional areas:

### A. Temperature Prediction

The Temperature Prediction section allows users to:

- Select a supported Zimbabwean city.
- Provide the required weather input variables.
- Submit a prediction request.
- View the predicted temperature and relevant model information.

The Streamlit prediction interface communicates directly with the FastAPI `/predict` endpoint using Python's `requests` library.

This is separate from the MCP prediction workflow used by the integrated conversational assistant.

### B. Weather and Climate Research Assistant

The research assistant allows users to submit natural-language questions relating to weather, climate and environmental conditions.

It supports:

- General weather-related questions.
- Document-based climate research.
- Historical drought and agricultural impact questions.
- AFOLU and emissions-related questions.
- Annual precipitation projection questions.
- Contextual follow-up questions.

The assistant retrieves information from the local knowledge base and uses the appropriate response workflow.

## 11.3 Application Session Handling

The Streamlit application uses session state to maintain recent conversation history.

The history is limited to ten messages.

The application also uses caching for selected research resources and agent-related components to reduce unnecessary reloading during normal use.

---

# 12. CONFIGURATION AND APPLICATION EXECUTION

## 12.1 Prerequisites

Before running Part 2, ensure that the following are available:

- Python 3.12.
- Git.
- Ollama.
- Llama 3.2:3b.
- nomic-embed-text.
- The project source code and required documents.
- The Part 2 Python dependencies.

## 12.2 Virtual Environment

The project uses a Python virtual environment to isolate its dependencies.

Navigate to the project root:

```cmd
cd "%USERPROFILE%\Desktop\ROSE-TADIWA\MACROEDTECH\WEEK5_8Task\zimbabwe-weather-intelligence"
```

Activate the virtual environment:

```cmd
part02_genai\venv\Scripts\activate
```

## 12.3 Dependency Installation

If setting up the environment for the first time, install the dependencies listed in the Part 2 requirements file:

```cmd
pip install -r part02_genai\requirements.txt
```

Ensure Ollama is installed and the required models have been pulled before starting the application.

## 12.4 Rebuilding the Knowledge Base

If source documents are added or updated, rebuild the knowledge base:

```cmd
python part02_genai\src\build_knowledge_base.py
```

Review the output and ensure the FAISS index is generated successfully before using the research assistant.

## 12.5 Running the Streamlit Application

From the project root, with the virtual environment activated, execute:

```cmd
streamlit run part02_genai\src\weather_assistant_ui.py
```

Streamlit will start the local application and provide a browser address for accessing the interface.

The application requires the saved knowledge base and an available Ollama service.

## 12.6 Running the Integrated CLI Assistant

The integrated conversational assistant can also be executed separately using:

```cmd
python part02_genai\src\integrated_weather_assistant.py
```

This provides a command-line interface for testing the integrated agent, research tool and MCP functionality.

---

# 13. TESTING AND EVALUATION

## 13.1 Testing Objectives

Testing was conducted to assess whether the main system components operate as intended.

The testing activities focused on:

- Document loading and extraction.
- Knowledge-base construction.
- Semantic retrieval.
- Climate research response workflows.
- Prediction API connectivity.
- MCP tool integration.
- Streamlit application functionality.
- Conversation context handling.

## 13.2 Knowledge Base Testing

The knowledge-base construction script includes five focused retrieval smoke tests.

All five tests passed.

The tests confirmed that expected evidence could be retrieved for the selected NDC-related research questions.

These checks validate retrieval presence, rather than the factual correctness of every final answer generated by the language model.

## 13.3 Research Response Testing

Specialised research workflows were tested using questions relating to:

- Agricultural challenges and historical maize production trends.
- The impacts of the 1991/92 drought.
- Water availability and future projections.
- AFOLU's contribution to greenhouse gas emissions.

The specialised response functions successfully returned the expected evidence for the tested questions.

The general document-research workflow still requires continued evaluation to identify and reduce occasional inaccuracies, unsupported interpretations and inconsistencies in generated responses.

## 13.4 Prediction API Testing

The deployed FastAPI service was tested through its available endpoints.

The API health check, supported-city retrieval and prediction functionality were exercised.

The Streamlit prediction interface also uses the deployed API directly.

## 13.5 MCP Testing

The MCP server was tested to confirm that the assistant could access the available API tools.

The following functionality was successfully tested:

- API health retrieval.
- Supported-city retrieval.
- Temperature prediction through the MCP prediction tool.

## 13.6 Application Testing

The Streamlit application was launched to verify that the user interface and its main functional areas were accessible.

The application supports both direct API-based temperature prediction and document-based climate research.

## 13.7 Conversation Memory Testing

The application retains recent conversation messages through Streamlit session state.

The configured history limit is ten messages, equivalent to approximately five exchanges.

This supports contextual follow-up interactions within the active session.

## 13.8 Testing Summary

| Component | Testing outcome |
|---|---|
| Document extraction | Successful |
| NDC evidence verification | Successful |
| Text chunking | Successful |
| Embedding generation | Successful |
| FAISS index creation | Successful |
| Retrieval smoke tests | 5/5 passed |
| Specialised climate research | Tested successfully |
| General document research | Requires further refinement |
| Prediction API integration | Tested successfully |
| MCP health and city tools | Tested successfully |
| MCP prediction tool | Tested successfully |
| Streamlit interface | Launched and tested |
| Conversation history implementation | Implemented |

---

# 14. TROUBLESHOOTING

The following issues may occur during setup or application execution.

## 14.1 Ollama Model Not Found

**Possible cause:** The required model has not been downloaded or is not available locally.

**Resolution:**

```cmd
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Confirm the available models:

```cmd
ollama list
```

## 14.2 FAISS Index Not Found

**Possible cause:** The knowledge base has not been built, or the saved index is missing.

**Resolution:**

From the project root, execute:

```cmd
python part02_genai\src\build_knowledge_base.py
```

Review the execution output to confirm that the index has been saved.

## 14.3 Prediction API Connection Error

**Possible cause:** The deployed API is unavailable, still starting, or experiencing a connection delay.

**Resolution:**

Check the API health endpoint:

`https://zimbabwe-weather-intelligence-macroedtech.onrender.com/health`

If the service is unavailable, retry after some time.

The application uses a free-tier hosted service, which may experience cold starts and response delays.

## 14.4 MCP Connection Error

**Possible cause:** The MCP server failed to start or the client could not establish a session.

**Resolution:**

Confirm that the MCP server script is available and that its dependencies are installed.

Check the prediction API's availability and review the terminal output for connection or execution errors.

## 14.5 PDF Extraction Warnings

**Possible cause:** Some PDF documents contain embedded font encodings that require additional font parsing support.

The knowledge-base build may display a warning indicating that `fontTools` is not installed.

If text extraction works correctly, this warning does not necessarily prevent knowledge-base construction.

If encoding problems are encountered, install the optional package:

```cmd
pip install fontTools
```

## 14.6 Streamlit Application Error

**Possible cause:** Missing dependencies, an unavailable knowledge base, an inactive Ollama service or an application configuration issue.

**Resolution:**

1. Confirm that the virtual environment is activated.
2. Confirm that the required dependencies are installed.
3. Confirm that Ollama is running.
4. Confirm that both required Ollama models are available.
5. Confirm that the FAISS index exists.
6. Review the terminal error message and address the reported issue.

---

# 15. SYSTEM LIMITATIONS

The following limitations should be considered when operating and evaluating the application.

## 15.1 Knowledge Base Coverage

The knowledge base currently relies on a limited collection of Zimbabwe-specific climate and weather documents.

Its ability to answer questions is therefore influenced by the coverage, quality and availability of the source material.

## 15.2 Language Model Accuracy

Although the system uses retrieval and evidence-grounding instructions, language models may occasionally produce inaccurate interpretations or unsupported statements.

Research responses should therefore be checked against the original source documents, particularly when used for formal reporting or decision-making.

## 15.3 Retrieval Limitations

Semantic retrieval depends on the quality of the document chunks and their embeddings.

Relevant information may not always appear among the retrieved results, especially when a question is broad or requires information spread across several documents.

## 15.4 API Availability

The prediction service is hosted on a free-tier platform and may experience cold starts or temporary response delays.

The assistant's prediction functionality consequently depends on the availability of the deployed API.

## 15.5 Prediction Scope

The existing prediction model provides a one-hour-ahead temperature forecast for the supported cities.

It does not currently provide a complete multi-day weather forecast.

## 15.6 Conversation Memory

Conversation history is maintained within the active Streamlit session.

It is not designed as permanent conversation storage across separate sessions.

## 15.7 Local Computing Requirements

The application relies on locally executed language and embedding models.

Its performance therefore depends on the available computing resources, particularly memory and processing capacity.

---

# 16. FUTURE IMPROVEMENTS

The following improvements could strengthen the application:

1. **Expand the knowledge base:** Incorporate additional Zimbabwe-specific climate, weather, agricultural and environmental research documents.
2. **Improve retrieval quality:** Experiment with alternative chunking strategies, retrieval parameters and ranking methods.
3. **Strengthen response evaluation:** Develop a broader evaluation dataset to assess factual accuracy, evidence relevance and response consistency.
4. **Improve source citations:** Provide more consistent document and page references in research responses.
5. **Enhance conversation memory:** Explore persistent conversation storage where appropriate.
6. **Improve user experience:** Refine the Streamlit interface and introduce additional visualisations.
7. **Expand prediction capabilities:** Explore additional forecast horizons and weather variables.
8. **Improve service reliability:** Consider more reliable API hosting and monitoring arrangements.

---

# 17. CONCLUSION

Part 2 extends the Zimbabwe Weather Intelligence project by integrating Generative AI with the existing machine learning-based temperature prediction service.

The implementation establishes a Zimbabwe-specific climate knowledge base using document extraction, text chunking, local embeddings and FAISS vector storage. It also introduces Retrieval-Augmented Generation, local language models, AI agent workflows, LangGraph orchestration and MCP-based integration with the existing prediction API.

The system provides two complementary capabilities: temperature prediction and contextual weather and climate research.

The knowledge-base construction process was completed successfully, with 103 source records processed into 217 chunks and all five retrieval smoke tests passing. The prediction API and MCP integration were also tested successfully.

Further improvements are required in general research response accuracy, retrieval evaluation, knowledge-base coverage and service reliability.

Overall, Part 2 establishes a functional Generative AI layer that extends the original Zimbabwe Weather Intelligence solution beyond temperature prediction into document-based climate research and conversational interaction.

---
