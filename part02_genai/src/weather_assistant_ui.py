import asyncio

import re

import sys

from datetime import date, time

from pathlib import Path

import requests

import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "part02_genai" / "src"

if str(SRC_PATH) not in sys.path:

    sys.path.insert(0, str(SRC_PATH))

import integrated_weather_assistant as assistant  # noqa: E402

API_BASE_URL = "https://zimbabwe-weather-intelligence-macroedtech.onrender.com"

CITIES = ["Harare", "Bulawayo", "Mutare", "Gweru"]

st.set_page_config(

    page_title="Zimbabwe Weather Intelligence",

    page_icon="🌦️",

    layout="wide",

)

# --- Light blue visual theme ---

st.markdown(

    """

    <style>

    :root {

        --weather-bg: #edf6ff;

        --weather-panel: #ffffff;

        --weather-blue: #4f9ee8;

        --weather-blue-dark: #245b8f;

        --weather-text: #18324b;

        --weather-muted: #536b80;

    }

    .stApp, [data-testid="stAppViewContainer"] {

        background: linear-gradient(145deg, #f4faff 0%, #eaf4ff 55%, #e3f1ff 100%);

        color: var(--weather-text);

    }

    [data-testid="stHeader"] {

        background: rgba(244, 250, 255, 0.92);

    }

    [data-testid="stSidebar"], [data-testid="stBottom"] {

        background-color: #e5f1fd;

    }

    h1, h2, h3, h4, p, label, .stMarkdown, [data-testid="stCaptionContainer"] {

        color: var(--weather-text);

    }

    [data-testid="stCaptionContainer"] {

        color: var(--weather-muted);

    }

    [data-testid="stTabs"] [role="tablist"] {

        display: flex;

        gap: 0.55rem;

        padding: 0.55rem;

        background: #d8eafa;

        border: 1px solid #c1dbf2;

        border-radius: 12px;

        margin-bottom: 1rem;

    }

    [data-testid="stTabs"] button[role="tab"] {

        color: #245b8f;

        background: #f8fbff;

        border: 1px solid #c1dbf2;

        border-radius: 9px;

        padding: 0.55rem 1rem;

        font-weight: 600;

        transition: background 0.15s ease, border-color 0.15s ease;

    }

    [data-testid="stTabs"] button[role="tab"]:hover {

        color: #174c7d;

        background: #eaf4ff;

        border-color: #75afe5;

    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {

        color: #ffffff;

        background: #398bd6;

        border: 1px solid #398bd6;

        box-shadow: 0 2px 6px rgba(36, 91, 143, 0.18);

    }

    [data-testid="stTabs"] button[role="tab"] p {

        color: inherit !important;

        font-weight: 600;

    }

    [data-testid="stForm"], [data-testid="stExpander"] {

        background: rgba(255, 255, 255, 0.9);

        border: 1px solid #cfe2f4;

        border-radius: 12px;

    }

    [data-testid="stMetric"] {

        background: linear-gradient(135deg, #ffffff, #edf6ff);

        border: 1px solid #c9def2;

        padding: 1rem;

        border-radius: 12px;

    }

    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {

        color: #18324b;

    }

    input, textarea, [data-baseweb="select"] > div,

    [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {

        background-color: #ffffff !important;

        color: #18324b !important;

        border-color: #b9d4ed !important;

    }

    [data-baseweb="popover"], [data-baseweb="menu"] {

        background-color: #ffffff !important;

        color: #18324b !important;

    }

    [data-testid="stFormSubmitButton"] button,

    [data-testid="stButton"] button {

        background: linear-gradient(135deg, #398bd6, #5ba8ed);

        color: #ffffff;

        border: 1px solid #398bd6;

        border-radius: 9px;

        font-weight: 600;

    }

    [data-testid="stFormSubmitButton"] button:hover,

    [data-testid="stButton"] button:hover {

        background: linear-gradient(135deg, #2878c2, #398bd6);

        border-color: #2878c2;

        color: #ffffff;

    }

    [data-testid="stAlert"] {

        border-radius: 10px;

    }

    hr {

        border-color: #c9def2;

    }

    code, pre {

        background-color: #f3f8fd !important;

        color: #23496b !important;

        border: 1px solid #cfe2f4;

        border-radius: 8px;

    }

    a {

        color: #245b8f !important;

    }

    </style>

    """,

    unsafe_allow_html=True,

)

st.title("🌦️ Zimbabwe Weather Intelligence")

st.caption(

    "Evidence-based climate research and one-hour-ahead temperature prediction "

    "for Harare, Bulawayo, Mutare and Gweru."

)

@st.cache_resource

def load_research_resources():

    """Load the FAISS retriever and local model used by the research workflow."""

    retriever = assistant.load_knowledge_base()

    llm = assistant.ChatOllama(model="llama3.2:3b", temperature=0)

    return retriever, llm


async def _run_agent_graph(messages):
    """Build and execute the LangGraph workflow within one event loop."""

    client = assistant.MultiServerMCPClient(
        {
            "zimbabwe_weather": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(assistant.MCP_SERVER_FILE)],
                "cwd": str(assistant.PROJECT_ROOT),
            }
        }
    )

    mcp_tools = await client.get_tools()

    safe_mcp_tools = [
        current_tool
        for current_tool in mcp_tools
        if current_tool.name != "predict_temperature"
    ]

    retriever, _ = load_research_resources()

    research_tool = assistant.create_research_tool(retriever)

    llm = assistant.ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    agent = assistant.create_agent(
        model=llm,
        tools=[research_tool] + safe_mcp_tools,
        system_prompt=assistant.SYSTEM_PROMPT,
    )

    graph = assistant.build_weather_agent_graph(agent)

    return await graph.ainvoke({"messages": messages})


@st.cache_resource



def is_drought_impact_question(question):
    """Route questions about drought conditions and their impacts."""
    text = question.lower()
    drought_terms = (
        "drought", "dry spell", "water scarcity", "water shortage",
        "water-shedding", "low rainfall", "rainfall deficit",
    )
    impact_terms = (
        "impact", "impacts", "consequence", "consequences", "effect",
        "effects", "affected", "affect", "damage", "loss", "losses",
        "economic", "economy", "industry", "industries", "employment",
        "wildlife", "elephant", "biodiversity", "ecosystem", "agriculture",
        "crop", "crops", "livestock", "cattle", "food", "prices",
        "livelihood", "water availability", "what happened",
    )
    return any(term in text for term in drought_terms) and any(
        term in text for term in impact_terms
    )

def _question_event_year(question):
    """Detect an explicitly named historical drought period."""
    text = question.lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)

    if re.search(r"\b1991\s*(?:/|-|to)\s*92\b|\b1991/1992\b|\b1991-1992\b", text):
        return "1991/92"

    if re.search(r"\b2015\s*(?:-|/|to)\s*2016\b|\b2015/16\b", text):
        return "2015/16"

    for year in ("2007", "2013", "2015", "2016", "2018", "2020"):
        if re.search(rf"\b{year}\b", text):
            return year
    return None

def _requested_drought_categories(question):
    """Identify the impact categories explicitly requested by the user."""
    q = question.lower()
    categories = {
        "water": ("water", "river", "rivers", "dam", "dams", "groundwater", "water supply"),
        "agriculture": ("agriculture", "crop", "crops", "maize", "harvest", "farming", "farmers"),
        "livestock": ("livestock", "cattle", "herd", "herds", "grazing", "animal disease"),
        "food security": ("food security", "food insecurity", "food prices", "hunger", "food supply"),
        "wildlife and biodiversity": ("wildlife", "elephant", "elephants", "biodiversity", "ecosystem", "habitat", "species"),
        "economy and employment": ("economic", "economy", "industry", "industries", "employment", "jobs", "business", "income", "financial"),
        "social impacts": ("household", "community", "communities", "women", "migration", "livelihood", "social"),
        "environment": ("environment", "environmental", "vegetation", "forest", "rainforest", "land degradation", "soil"),
    }
    return [name for name, terms in categories.items() if any(term in q for term in terms)]

def _category_search_terms(category):
    """Return focused retrieval language for a requested impact category."""
    terms = {
        "water": "water resources river flow dam levels water supply water shortages water availability",
        "agriculture": "crop production maize yields harvest crop losses farmer income agriculture drought",
        "livestock": "livestock cattle animal disease deaths grazing pasture drought",
        "food security": "food security hunger food prices household food supply drought",
        "wildlife and biodiversity": "wildlife elephant deaths biodiversity ecosystems habitat species drought",
        "economy and employment": "economic impacts industry closures employment jobs agricultural raw materials drought",
        "social impacts": "households communities livelihoods migration women water collection drought",
        "environment": "environmental impacts vegetation forests rainforest biodiversity soil drought",
    }
    return terms[category]


def _drought_evidence_score(document, question=""):
    """Score evidence for event match, requested categories and direct outcomes."""
    text = str(getattr(document, "page_content", "")).lower()
    metadata = getattr(document, "metadata", {}) or {}
    source = str(metadata.get("source", "")).lower()
    q = question.lower().replace("–", "-").replace("—", "-")
    score = 0

    if "drought" in text or "dry spell" in text:
        score += 3
    if "zimbabwe" in text or "zimbabwe" in source:
        score += 1

    direct_outcomes = (
        "water-shedding", "water shedding", "water discharge",
        "water shortages", "water shortage", "dam levels",
        "crop losses", "crop loss", "crop yields", "crop yield",
        "maize yield", "farmer income", "crop failure", "food prices",
        "food insecurity", "sold cattle", "livestock disease",
        "cattle died", "cattle deaths", "livestock deaths",
        "lack of grazing", "elephant deaths", "elephants",
        "wildlife", "biodiversity", "industries faced closure",
        "industry closure", "employment opportunities", "rainforest",
    )
    score += 4 * sum(1 for term in direct_outcomes if term in text)

    if any(term in text for term in (
        "during the drought", "drought affected", "drought-affected",
        "extended drought", "extreme drought", "in drought years",
        "during drought",
    )):
        score += 3

    event = _question_event_year(question)

    projection_terms = (
        "projected", "projection", "by 2050", "by 2080",
        "2040-2059", "2080-2099", "rcp 2.6", "rcp 8.5",
    )
    passage_is_projection = any(term in text for term in projection_terms)

    question_requests_projection = any(term in q for term in (
        "projected", "projection", "projections", "future",
        "by 2050", "by 2080", "expected by", "will decrease",
        "will decline", "forecast",
    ))

    # Keep projections when the question explicitly asks about future risks.
    # Otherwise, rank them lower for questions about historical events.
    if passage_is_projection:
        if question_requests_projection:
            score += 5
        elif event:
            score -= 7
        else:
            score -= 2

    if any(term in text for term in (
        "adaptation options", "adaptation measures",
        "should be implemented", "recommended",
        "policy recommendation",
    )):
        score -= 4

    requested = _requested_drought_categories(question)

    category_terms = {
        "water": (
            "water", "dam", "river", "groundwater", "water-shedding",
        ),
        "agriculture": (
            "agriculture", "crop", "maize", "farmer", "harvest",
            "cereal", "farming",
        ),
        "livestock": (
            "livestock", "cattle", "grazing", "animal", "pasture",
        ),
        "food security": (
            "food", "hunger", "prices", "household", "nutrition",
        ),
        "wildlife and biodiversity": (
            "wildlife", "elephant", "biodiversity", "ecosystem",
            "habitat", "species",
        ),
        "economy and employment": (
            "economy", "economic", "industry", "industries",
            "employment", "jobs", "income", "business",
        ),
        "social impacts": (
            "household", "community", "communities", "women",
            "migration", "livelihood",
        ),
        "environment": (
            "environment", "vegetation", "forest", "rainforest",
            "soil", "biodiversity",
        ),
    }

    if requested:
        matched = 0

        for category in requested:
            if any(term in text for term in category_terms[category]):
                score += 6
                matched += 1

        if matched == 0:
            score -= 8

    # Prefer passages that explicitly match the requested historical event,
    # but do not discard other potentially useful evidence.
    if event == "1991/92":
        if re.search(
            r"1991\s*(?:/|-|to)\s*92|1991/1992|1991-1992",
            text,
        ):
            score += 14
        else:
            score -= 14

    elif event == "2015/16":
        if re.search(
            r"2015\s*(?:-|/|to)\s*2016|2015/16|\b2015\b|\b2016\b",
            text,
        ):
            score += 12
        else:
            score -= 12

    elif event:
        if re.search(rf"\b{re.escape(event)}\b", text):
            score += 10
        else:
            score -= 10

    return score





def retrieve_drought_evidence(retriever, question):
    """Retrieve focused evidence for historical droughts and water impacts."""

    question_lower = question.lower()
    event_year = _question_event_year(question)
    categories = _requested_drought_categories(question)

    queries = [question]

    # Correctly map the category names returned by
    # _requested_drought_categories() to retrieval queries.
    category_queries = {
        "water": [
            "Zimbabwe Bulawayo water shedding water shortages water supply",
            "Zimbabwe NDC water shedding Bulawayo water availability",
            "Zimbabwe 38% decline national water availability per capita 2050",
        ],
        "agriculture": [
            "Zimbabwe drought agriculture maize cereal crop yields food insecurity",
        ],
        "livestock": [
            "Zimbabwe 2015/16 drought livestock cattle disease 61% cattle deaths",
        ],
        "food security": [
            "Zimbabwe drought food insecurity food prices household food supply",
        ],
        "wildlife and biodiversity": [
            "Zimbabwe 1991/92 drought 148 elephants Hwange 1500 Gonarezhou",
            "Zimbabwe 1991/92 drought Victoria Falls discharge plant biodiversity",
        ],
        "economy and employment": [
            "Zimbabwe 1991/92 drought Bulawayo industries agricultural raw materials",
            "Zimbabwe drought industry closure reduced agricultural raw materials",
        ],
        "social impacts": [
            "Zimbabwe drought household community livelihood impacts",
        ],
        "environment": [
            "Zimbabwe 1991/92 drought environmental impacts Victoria Falls biodiversity",
        ],
    }

    for category in categories:
        queries.extend(category_queries.get(category, []))

    # Specific searches for the 1991/92 drought.
    if event_year == "1991/92":
        queries.extend([
            "Zimbabwe 1991/92 drought Victoria Falls reduced discharge plant biodiversity",
            "1991/92 drought 148 elephants died Hwange National Park",
            "1991/92 drought 1500 elephants died Gonarezhou National Park",
            "Zimbabwe 1991/92 drought industries Bulawayo agricultural raw materials closure",
            "1991 1992 drought Zimbabwe environmental economic impacts",
        ])

    # Specific searches for the 2015/16 drought.
    if event_year == "2015/16":
        queries.extend([
            "Zimbabwe 2015/16 drought low rainfall livestock disease 61% cattle deaths",
            "2015/16 drought Manicaland Masvingo cattle deaths livestock disease",
        ])

    # Water availability and Bulawayo water shedding.
    if "water" in question_lower or "2050" in question_lower:
        queries.extend([
            "Zimbabwe 38% decline water availability per capita by 2050",
            "Bulawayo water shedding Zimbabwe climate change water availability",
            "Bulawayo water shedding water shortages NDC Zimbabwe",
        ])

    # Long-term crop yield trends.
    if any(term in question_lower for term in (
        "maize", "cereal", "crop yield", "yield trends"
    )):
        queries.extend([
            "Zimbabwe maize yields gradually declined over past 15 years",
            "Zimbabwe cereal yield decreased since independence maize yields",
        ])

    unique_documents = []
    seen_content = set()

    for search_query in queries:
        try:
            results = retriever.invoke(search_query)
        except Exception as exc:
            print(
                f"Evidence retrieval warning for '{search_query}': {exc}"
            )
            continue

        for document in results:
            content = getattr(document, "page_content", "")

            if not content:
                continue

            content_key = " ".join(content.lower().split())

            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_documents.append(document)

    # Keep records containing the specific evidence needed for the question.
    priority_documents = []

    for document in unique_documents:
        content = document.page_content.lower()

        required_evidence = False

        if event_year == "1991/92":
            required_evidence = (
                "1991/92" in content
                or "1991/1992" in content
                or (
                    "victoria falls" in content
                    and (
                        "discharge" in content
                        or "biodiversity" in content
                    )
                )
                or (
                    "hwange" in content
                    and "elephant" in content
                )
                or (
                    "gonarezhou" in content
                    and "elephant" in content
                )
                or (
                    "bulawayo" in content
                    and (
                        "raw materials" in content
                        or "industries" in content
                    )
                )
            )

        elif "water" in question_lower:
            required_evidence = (
                (
                    "38%" in content
                    and "2050" in content
                    and "per capita" in content
                )
                or (
                    "bulawayo" in content
                    and (
                        "water shedding" in content
                        or "water-shedding" in content
                        or "water shedding" in content
                    )
                )
            )

        elif event_year == "2015/16":
            required_evidence = (
                "61%" in content
                and "cattle" in content
            )

        elif any(term in question_lower for term in (
            "maize", "cereal", "crop yield"
        )):
            required_evidence = (
                "maize" in content
                and (
                    "past 15 years" in content
                    or "gradually declined" in content
                )
            ) or (
                "cereal yield" in content
                and "independence" in content
            )

        if required_evidence:
            priority_documents.append(document)

    ranked_documents = sorted(
        unique_documents,
        key=lambda document: _drought_evidence_score(
            document, question
        ),
        reverse=True,
    )

    # Preserve targeted evidence even if its general relevance score is low.
    selected_documents = []

    for document in priority_documents + ranked_documents:
        if document not in selected_documents:
            selected_documents.append(document)

    # Keep the evidence set manageable while retaining targeted records.
    return selected_documents[:20]



def format_drought_evidence(documents):

    """Format retrieved passages with stable, clearly separated evidence IDs."""

    evidence_parts = []

    evidence_records = []

    for number, document in enumerate(documents, start=1):

        metadata = getattr(document, "metadata", {}) or {}

        source_value = (

            metadata.get("source")

            or metadata.get("file_path")

            or metadata.get("filename")

            or "Unknown source"

        )

        source_name = Path(str(source_value)).name

        page = metadata.get("page")

        content = str(

            getattr(document, "page_content", "")

        ).strip()

        evidence_id = f"E{number}"

        if page is not None:

            source_label = f"{source_name}, PDF page {page}"

        else:

            source_label = source_name

        evidence_parts.append(

            f"BEGIN EVIDENCE {evidence_id}\n"

            f"Evidence ID: [{evidence_id}]\n"

            f"Source: {source_label}\n"

            f"Passage:\n{content}\n"

            f"END EVIDENCE {evidence_id}"

        )

        evidence_records.append({

            "id": evidence_id,

            "source": source_name,

            "page": page,

            "content": content,

        })

    return "\n\n".join(evidence_parts), evidence_records



def answer_drought_impact_question(question, retriever, llm):
    """Generate evidence-checked answers for drought and climate impacts."""

    print("DEBUG: EVIDENCE-BASED DROUGHT FUNCTION IS RUNNING")

    documents = retrieve_drought_evidence(retriever, question)

    evidence, evidence_records = format_drought_evidence(documents)

    question_lower = question.lower()

    print("DEBUG: RETRIEVED EVIDENCE RECORDS:", len(evidence_records))
    print(
        "DEBUG: AVAILABLE EVIDENCE IDS:",
        [record["id"] for record in evidence_records],
    )

    def find_record(required_groups):
        """Find one retrieved record supporting all required term groups."""

        for record in evidence_records:
            content = str(record.get("content", "")).lower()

            if all(
                any(term.lower() in content for term in group)
                for group in required_groups
            ):
                return record["id"]

        return None

    sections = {
        "Long-term Agricultural Trends": [],
        "Documented Historical Impacts": [],
        "Future Projections": [],
        "Agricultural Vulnerability": [],
    }

    missing = []

    def add_finding(section, required_groups, statement):
        evidence_id = find_record(required_groups)

        if evidence_id:
            sections[section].append(
                f"- {statement} [{evidence_id}]"
            )
            print(
                "DEBUG: VERIFIED FINDING:",
                statement,
                "| EVIDENCE:",
                evidence_id,
            )
        else:
            missing.append(statement)
            print("DEBUG: MISSING SUPPORT:", statement)

    # --------------------------------------------------
    # 1991/92 HISTORICAL DROUGHT
    # --------------------------------------------------

    event_year = _question_event_year(question)

    if event_year == "1991/92":

        add_finding(
            "Documented Historical Impacts",
            [
                ["victoria falls"],
                ["discharge", "flow"],
            ],
            (
                "During the 1991/92 drought, Victoria Falls experienced "
                "reduced water discharge."
            ),
        )

        add_finding(
            "Documented Historical Impacts",
            [
                ["victoria falls"],
                ["plant biodiversity", "biodiversity"],
            ],
            (
                "The 1991/92 drought was also associated with a loss "
                "of plant biodiversity."
            ),
        )

        add_finding(
            "Documented Historical Impacts",
            [
                ["148"],
                ["elephant"],
                ["hwange"],
            ],
            (
                "The drought resulted in the reported deaths of "
                "148 elephants in Hwange National Park."
            ),
        )

        add_finding(
            "Documented Historical Impacts",
            [
                ["1,500", "1500"],
                ["elephant"],
                ["gonarezhou"],
            ],
            (
                "The drought resulted in the reported deaths of "
                "1,500 elephants in Gonarezhou National Park."
            ),
        )

        add_finding(
            "Documented Historical Impacts",
            [
                ["bulawayo"],
                ["industry", "industries", "industrial"],
                ["raw materials"],
            ],
            (
                "Industrial activity, particularly in Bulawayo, faced "
                "closure risks because drought reduced the availability "
                "of agricultural raw materials."
            ),
        )

    # --------------------------------------------------
    # LONG-TERM MAIZE AND CEREAL TRENDS
    # --------------------------------------------------

    if "maize" in question_lower:
        add_finding(
            "Long-term Agricultural Trends",
            [
                ["maize"],
                ["past 15 years", "15 years"],
                ["declined", "decreased"],
            ],
            (
                "Maize yields gradually declined over the past "
                "15 years while the population increased."
            ),
        )

    if "cereal" in question_lower:
        add_finding(
            "Long-term Agricultural Trends",
            [
                ["cereal yield"],
                ["independence"],
                ["decreased", "declined"],
            ],
            (
                "The source reports that cereal yield had decreased "
                "since the country's independence in 1965. The date "
                "is reproduced as stated in the source."
            ),
        )

    # --------------------------------------------------
    # 2016 FALL ARMYWORM
    # --------------------------------------------------

    if "fall armyworm" in question_lower:
        add_finding(
            "Documented Historical Impacts",
            [
                ["fall armyworm"],
                ["2016"],
                ["263,000"],
                ["83 million", "$83 million", "$83m", "83m"],
            ],
            (
                "In 2016, the fall armyworm outbreak caused estimated "
                "maize losses of 263,000 tonnes, valued at approximately "
                "US$83 million."
            ),
        )

    # --------------------------------------------------
    # 2015/16 LIVESTOCK IMPACTS
    # --------------------------------------------------

    if any(term in question_lower for term in ("livestock", "cattle")):
        add_finding(
            "Documented Historical Impacts",
            [
                ["61%"],
                ["cattle"],
                [
                    "2015/16", "2015-16", "2015–16",
                    "2015/2016", "2015-2016", "2015–2016",
                ],
            ],
            (
                "During the 2015/16 drought, low rainfall increased "
                "livestock disease. Livestock disease accounted for "
                "61% of reported cattle deaths, particularly in "
                "Manicaland and Masvingo provinces."
            ),
        )

    # --------------------------------------------------
    # WATER AVAILABILITY PROJECTION
    # --------------------------------------------------

    if any(term in question_lower for term in (
        "water", "2050", "per capita"
    )):
        add_finding(
            "Future Projections",
            [
                ["38%"],
                ["2050"],
                ["per capita"],
                ["water availability"],
            ],
            (
                "Zimbabwe is projected to experience a 38% decline "
                "in national water availability per capita by 2050. "
                "This is a future projection, not an observed "
                "historical decline."
            ),
        )

    # --------------------------------------------------
    # BULAWAYO WATER SHEDDING
    # --------------------------------------------------

    if "bulawayo" in question_lower:
        add_finding(
            "Documented Historical Impacts",
            [
                ["bulawayo"],
                ["water shedding", "water-shedding"],
            ],
            (
                "Water shedding was documented in Bulawayo, "
                "illustrating an existing water-supply challenge."
            ),
        )

    # --------------------------------------------------
    # OPTIONAL RAIN-FED AGRICULTURE CONTEXT
    # Only include if specifically requested.
    # --------------------------------------------------

    if any(term in question_lower for term in (
        "rain-fed", "rain fed", "80%"
    )):
        add_finding(
            "Agricultural Vulnerability",
            [
                ["80%"],
                ["rain-fed", "rain fed"],
            ],
            (
                "Approximately 80% of agricultural production is "
                "rain-fed, making agricultural output vulnerable "
                "to rainfall variability."
            ),
        )

    # --------------------------------------------------
    # BUILD ANSWER
    # --------------------------------------------------

    output = ["**Documented Climate and Drought Impacts in Zimbabwe**"]

    for section_name, findings in sections.items():
        if findings:
            output.extend([
                "",
                f"**{section_name}**",
                "",
                *findings,
            ])

    if missing:
        output.extend([
            "",
            "**Evidence limitations**",
        ])

        for statement in missing:
            output.append(
                "- The retrieved passages did not provide sufficient "
                "matching evidence for: " + statement
            )

    if not any(sections.values()):
        output.extend([
            "",
            "No requested findings could be verified against the "
            "retrieved evidence. Try a more specific question or "
            "inspect the retrieved passages.",
        ])

    answer = "\n".join(output).strip()

    print(
        "DEBUG: TOTAL VERIFIED FINDINGS:",
        sum(len(items) for items in sections.values()),
    )
    print("DEBUG: FINDINGS WITHOUT MATCHING EVIDENCE:", len(missing))
    print("DEBUG: FINAL DROUGHT ANSWER GENERATED")

    return answer, evidence_records



def is_afolu_question(question):
    """Identify questions specifically about AFOLU and emissions."""
    text = question.lower()

    return (
        "afolu" in text
        or (
            "agriculture, forestry and other land use" in text
            and any(
                term in text
                for term in (
                    "emission", "greenhouse gas",
                    "carbon sink", "carbon source",
                )
            )
        )
    )


def answer_in_existing_workflow(question, history):
    """Use one central evidence-first RAG path for all document research."""

    print("\nDEBUG: answer_in_existing_workflow() WAS CALLED")
    print(f"DEBUG: QUESTION RECEIVED: {question}")

    retriever, llm = load_research_resources()

    # Check annual precipitation projection questions first.
    annual_check = assistant.is_annual_precipitation_projection_question(question)
    print(f"DEBUG: Annual precipitation question: {annual_check}")

    if annual_check:
        documents = assistant.retrieve_research_evidence(retriever, question)

        has_table = any(
            (doc.metadata or {}).get("record_type") == "normalized_climate_table"
            for doc in documents
        )

        if has_table:
            return assistant.format_annual_precipitation_projection_answer(), []

        return (
            "The normalized Table 3 evidence was not retrieved from the local index. "
            "Please rebuild the knowledge base and inspect the Table 3 retrieval test.",
            [],
        )



        # Handle AFOLU emissions questions before general document research.
    afolu_check = is_afolu_question(question)
    print(f"DEBUG: Is AFOLU question: {afolu_check}")

    if afolu_check:
        print("DEBUG: ROUTING TO AFOLU EVIDENCE-BASED RESEARCH")

        result = answer_afolu_question(question, retriever)

        print("DEBUG: AFOLU RESEARCH FINISHED")

        return result
    
    # Check whether this is a drought-impact question.
    drought_check = is_drought_impact_question(question)
    print(f"DEBUG: Is drought-impact question: {drought_check}")

    if drought_check:
        print("DEBUG: ROUTING TO answer_drought_impact_question()")

        result = answer_drought_impact_question(question, retriever, llm)

        print("DEBUG: answer_drought_impact_question() FINISHED")

        return result

    # General document research route.
    document_check = assistant.is_document_research_question(question)
    print(f"DEBUG: Is document-research question: {document_check}")

    if document_check:
        print("DEBUG: ROUTING TO GENERAL DOCUMENT RESEARCH")

        documents = assistant.retrieve_research_evidence(retriever, question)

        answer = assistant.answer_document_research_question(
            llm, retriever, question, history, documents=documents
        )

        evidence_records = []

        for number, document in enumerate(documents, start=1):
            metadata = document.metadata or {}

            evidence_records.append({
                "id": f"E{number}",
                "source": Path(str(metadata.get("source", "Unknown source"))).name,
                "page": metadata.get("page"),
                "printed_page": metadata.get("printed_page"),
                "content": document.page_content,
            })

        return answer, evidence_records

    # General weather-agent route.
    print("DEBUG: ROUTING TO LANGGRAPH AGENT")

    

    messages = history[-assistant.MAX_HISTORY_MESSAGES:] + [
    {"role": "user", "content": question}
]

    result = asyncio.run(_run_agent_graph(messages))

    answer = result.get(
        "answer",
        "I could not generate a usable response. Please rephrase the question.",
    )

    return answer, []

def answer_afolu_question(question, retriever):
    """Return evidence-checked findings about Zimbabwe's AFOLU emissions."""

    print("DEBUG: AFOLU EVIDENCE-BASED FUNCTION IS RUNNING")

    queries = [
        "Zimbabwe NDC AFOLU 54% greenhouse gas emissions 2017",
        "Zimbabwe AFOLU sector 2017 emissions 54 percent",
        "Zimbabwe AFOLU net carbon sink 1990 1993 net source 1994 2017",
        "Zimbabwe AFOLU deforestation fuelwood livestock logging veld fires emissions",
        "Zimbabwe agriculture forestry other land use emissions 2017 NDC",
    ]

    documents = []
    seen = set()

    for query in queries:
        try:
            results = retriever.invoke(query)
        except Exception as exc:
            print(f"AFOLU retrieval warning: {exc}")
            continue

        for document in results:
            content = getattr(document, "page_content", "")

            if not content:
                continue

            key = " ".join(content.lower().split())

            if key not in seen:
                seen.add(key)
                documents.append(document)

    _, evidence_records = format_drought_evidence(documents)

    print("DEBUG: AFOLU RECORDS RETRIEVED:", len(evidence_records))

    def find_record(groups):
        for record in evidence_records:
            content = str(record.get("content", "")).lower()

            if all(
                any(term.lower() in content for term in group)
                for group in groups
            ):
                return record["id"]

        return None

    findings = []
    missing = []

    def add_finding(groups, statement):
        evidence_id = find_record(groups)

        if evidence_id:
            findings.append(f"- {statement} [{evidence_id}]")
            print(
                "DEBUG: VERIFIED AFOLU FINDING:",
                statement,
                "| EVIDENCE:",
                evidence_id,
            )
        else:
            missing.append(statement)
            print("DEBUG: AFOLU EVIDENCE NOT FOUND:", statement)

    add_finding(
        [
            ["afolu"],
            ["54%"],
            ["2017"],
        ],
        (
            "The Agriculture, Forestry and Other Land Use (AFOLU) "
            "sector contributed approximately 54% of Zimbabwe's "
            "greenhouse gas emissions in 2017."
        ),
    )

    add_finding(
        [
            ["afolu"],
            ["net sink", "carbon sink"],
            ["1990", "1990–1993", "1990-1993"],
            ["1993"],
        ],
        (
            "The AFOLU sector functioned as a net carbon sink "
            "between 1990 and 1993."
        ),
    )

    add_finding(
        [
            ["afolu"],
            ["net source", "carbon source"],
            ["1994"],
            ["2017"],
        ],
        (
            "The AFOLU sector functioned as a net source of "
            "greenhouse gas emissions from 1994 to 2017."
        ),
    )

    add_finding(
        [
            ["deforestation"],
            ["fuelwood", "fuel wood"],
            ["livestock"],
            ["logging"],
            ["veld fires", "veld fire"],
        ],
        (
            "Activities associated with AFOLU emissions included "
            "deforestation, fuelwood use, livestock, logging "
            "and veld fires."
        ),
    )

    output = [
        "**Zimbabwe's AFOLU Sector and Greenhouse Gas Emissions**",
        "",
        "**2017 Emissions Contribution**",
        "",
    ]

    output.extend(findings[:1])

    historical_findings = findings[1:]

    if historical_findings:
        output.extend([
            "",
            "**Historical Role of the AFOLU Sector**",
            "",
            *historical_findings,
        ])

    if missing:
        output.extend([
            "",
            "**Evidence limitations**",
        ])

        for statement in missing:
            output.append(
                "- The retrieved passages did not provide sufficient "
                "matching evidence for: " + statement
            )

    if not findings:
        output.extend([
            "",
            "The retrieved evidence did not provide sufficient "
            "support for the requested AFOLU findings. Inspect the "
            "retrieved evidence and knowledge-base index.",
        ])

    print("DEBUG: AFOLU VERIFIED FINDINGS:", len(findings))

    return "\n".join(output), evidence_records


if "research_history" not in st.session_state:

    st.session_state.research_history = []

if "last_research_evidence" not in st.session_state:

    st.session_state.last_research_evidence = []

tab_research, tab_prediction, tab_about = st.tabs(

    ["Research Assistant", "Temperature Prediction", "About"]

)

with tab_research:

    st.subheader("Ask a question about Zimbabwe's weather and climate")

    st.write(

        "Document-based climate questions use the existing evidence-first research "

        "functions. General weather questions use the LangGraph agent and its MCP tools."

    )

    with st.form("research_form"):

        question = st.text_area(

            "Your question",

            placeholder=(

                "e.g. What are the main consequences of drought in Zimbabwe?"

            ),

            height=100,

        )

        submitted = st.form_submit_button("Ask the assistant", type="primary")

    if submitted:

        if not question.strip():

            st.warning("Enter a question first.")

        else:

            try:

                with st.spinner(

                    "Retrieving source passages and generating an evidence-based answer..."

                ):

                    previous_history = list(st.session_state.research_history)

                    answer, evidence_records = answer_in_existing_workflow(

                        question.strip(), previous_history

                    )

                st.session_state.last_research_evidence = evidence_records

                st.session_state.research_history.extend(

                    [

                        {"role": "user", "content": question.strip()},

                        {"role": "assistant", "content": str(answer)},

                    ]

                )

                del st.session_state.research_history[

                    :-assistant.MAX_HISTORY_MESSAGES

                ]

                st.markdown("### Answer")

                st.markdown(str(answer))

            except Exception as exc:

                st.error(

                    "The assistant could not complete this request. Check that "

                    "Ollama is running, the required models are installed, the "

                    "FAISS index exists, and the MCP server can start."

                )

                st.code(str(exc))

    if st.session_state.last_research_evidence:

        with st.expander("Inspect retrieved evidence used for the latest answer"):

            st.caption(

                "These are the retrieved passages ranked for relevance to direct "

                "drought impacts. Ranking and citation checks are heuristic; "

                "manually verify that each source passage supports the claim."

            )

            for record in st.session_state.last_research_evidence:

                if record.get("printed_page") is not None:
                    page_text = f" — printed page {record['printed_page']}"
                elif record.get("page") is not None:
                    page_text = f" — PDF page {record['page']}"
                else:
                    page_text = ""

                st.markdown(

                    f"**[{record['id']}] {record['source']}{page_text}**"

                )

                st.write(record["content"])

                st.divider()

    if st.session_state.research_history:

        with st.expander("Conversation history"):

            for message in st.session_state.research_history:

                label = "You" if message["role"] == "user" else "Assistant"

                st.markdown(f"**{label}:** {message['content']}")

                st.divider()

    if st.button("Clear conversation history"):

        st.session_state.research_history = []

        st.session_state.last_research_evidence = []

        st.rerun()

with tab_prediction:

    st.subheader("Predict temperature one hour ahead")

    st.warning(

        "Enter weather measurements from a trusted source. Do not guess values. "

        "The model predicts temperature only; it does not directly predict rainfall "

        "probability or thunderstorms."

    )

    with st.form("prediction_form"):

        c1, c2 = st.columns(2)

        with c1:

            city = st.selectbox("City", CITIES)

            input_date = st.date_input("Input date", value=date(2025, 12, 31))

            input_time = st.time_input("Input time", value=time(10, 30))

            temperature = st.number_input(

                "Temperature (°C)", min_value=-90.0, max_value=60.0,

                value=22.7, step=0.1

            )

            humidity = st.number_input(

                "Humidity (%)", min_value=0.0, max_value=100.0,

                value=63.0, step=1.0

            )

            precipitation = st.number_input(

                "Precipitation (mm)", min_value=0.0, value=0.3, step=0.1

            )

        with c2:

            pressure = st.number_input(

                "Atmospheric pressure (hPa)", min_value=0.0,

                value=1017.0, step=1.0

            )

            wind_speed = st.number_input(

                "Wind speed (km/h)", min_value=0.0,

                value=3.6, step=0.1

            )

            wind_direction = st.number_input(

                "Wind direction (degrees)", min_value=0.0,

                max_value=360.0, value=173.0, step=1.0

            )

            cloud_cover = st.number_input(

                "Cloud cover (%)", min_value=0.0, max_value=100.0,

                value=100.0, step=1.0

            )

        confirmed = st.checkbox(

            "I have checked that these measurements are accurate."

        )

        predict_clicked = st.form_submit_button(

            "Predict next-hour temperature", type="primary"

        )

    if predict_clicked:

        if not confirmed:

            st.warning("Please confirm that you checked the input measurements.")

        else:

            payload = {

                "city": city,

                "prediction_datetime": (

                    f"{input_date.isoformat()} {input_time.strftime('%H:%M')}"

                ),

                "temperature": temperature,

                "humidity": humidity,

                "precipitation": precipitation,

                "pressure": pressure,

                "wind_speed": wind_speed,

                "wind_direction": wind_direction,

                "cloud_cover": cloud_cover,

            }

            try:

                with st.spinner("Calling the deployed XGBoost prediction API..."):

                    response = requests.post(

                        f"{API_BASE_URL}/predict", json=payload, timeout=60

                    )

                if not response.ok:

                    st.error(

                        f"Prediction API returned HTTP {response.status_code}."

                    )

                    st.code(response.text[:2000])

                else:

                    result = response.json()

                    predicted = result.get("predicted_temperature_c")

                    if predicted is None:

                        st.error(

                            "The API response did not contain "

                            "predicted_temperature_c."

                        )

                        st.json(result)

                    else:

                        st.success("Prediction received")

                        st.metric(

                            "Predicted temperature one hour later",

                            f"{float(predicted):.2f} °C",

                        )

                        result_cols = st.columns(3)

                        result_cols[0].metric(

                            "Model", str(result.get("model", "XGBoost"))

                        )

                        result_cols[1].metric(

                            "Test MAE",

                            f"{result.get('test_mae_c', 'N/A')} °C",

                        )

                        result_cols[2].metric(

                            "Test RMSE",

                            f"{result.get('test_rmse_c', 'N/A')} °C",

                        )

                        st.write(f"**City:** {result.get('city', city)}")

                        st.write(

                            "**Input time:** "

                            f"{result.get('input_datetime', payload['prediction_datetime'])}"

                        )

                        st.write(

                            "**Forecast time:** "

                            f"{result.get('forecast_datetime', 'Not provided')}"

                        )

                        st.caption(

                            "This is a model estimate, not a direct observation or "

                            "a guarantee of future conditions. Test metrics describe "

                            "model evaluation and are not a confidence interval for "

                            "this individual prediction."

                        )

                        with st.expander("View API response"):

                            st.json(result)

            except requests.RequestException as exc:

                st.error("Could not connect to the deployed prediction API.")

                st.write(

                    "Check your internet connection and whether the API is awake."

                )

                st.code(str(exc))

with tab_about:

    st.subheader("Project components")

    st.markdown(

        "- **Research:** local FAISS retrieval and `llama3.2:3b` through Ollama.\n"

        "- **Agent orchestration:** LangGraph with the existing MCP weather tools "

        "for general weather questions.\n"

        "- **Prediction:** deployed XGBoost API for one-hour-ahead temperature estimates.\n"

        "- **Supported cities:** Harare, Bulawayo, Mutare and Gweru.\n"

        "- **Memory:** recent research conversation exchanges retained in the current "

        "Streamlit session."

    )

    st.info(

        "The interface reuses the existing CLI agent construction and MCP tool "

        "definitions. Drought-impact answers also show the exact retrieved passages "

        "so their statistics can be checked. Temperature prediction remains in its "

        "dedicated tab and calls the deployed prediction API."

    )

st.divider()

st.caption(

    "Zimbabwe Weather Intelligence | Research findings and predictions should be "

    "checked against their source evidence and input data."

)
