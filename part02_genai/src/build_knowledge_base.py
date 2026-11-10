from pathlib import Path
import re

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_DIR / "documents"
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_pdf_pages(file_path: Path, relative_path: Path):
    documents = []
    reader = PdfReader(str(file_path))
    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(Document(
            page_content=text,
            metadata={
                "source": relative_path.as_posix(),
                "file_type": ".pdf",
                "page": page_index + 1,
                "page_index": page_index,
                "record_type": "pdf_page",
            },
        ))
    print(f"Loaded PDF: {relative_path} ({len(documents)} text pages)")
    return documents


def load_page_labelled_markdown(file_path: Path, relative_path: Path):
    """Split the curated NDC extract into records with printed-page metadata."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    if "## Printed page" not in content:
        return [Document(
            page_content=content,
            metadata={"source": relative_path.as_posix(), "file_type": ".md",
                      "record_type": "markdown_document"},
        )] if content.strip() else []

    matches = list(re.finditer(r"(?m)^## Printed pages? (\d+(?:–\d+)?)(?:\s*)$", content))
    documents = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        page_label = match.group(1)
        body = content[start:end].strip()
        if not body:
            continue
        page_start = int(re.split(r"[–-]", page_label)[0])
        page_end = int(re.split(r"[–-]", page_label)[-1])
        documents.append(Document(
            page_content=f"Source document: Zimbabwe Revised Nationally Determined Contribution (NDC), 2021.\nPrinted page(s): {page_label}.\n\n{body}",
            metadata={
                "source": relative_path.as_posix(),
                "file_type": ".md",
                "printed_page": page_label,
                "printed_page_start": page_start,
                "printed_page_end": page_end,
                "page": None,
                "page_index": None,
                "record_type": "ndc_page_extract",
            },
        ))
    return documents


def load_documents():
    documents = []
    supported_extensions = {".txt", ".md", ".pdf"}
    files = sorted(path for path in DOCUMENTS_DIR.rglob("*")
                   if path.is_file() and path.suffix.lower() in supported_extensions)
    for file_path in files:
        relative_path = file_path.relative_to(DOCUMENTS_DIR)
        try:
            if file_path.suffix.lower() == ".pdf":
                documents.extend(load_pdf_pages(file_path, relative_path))
            elif file_path.suffix.lower() == ".md":
                documents.extend(load_page_labelled_markdown(file_path, relative_path))
            else:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": relative_path.as_posix(), "file_type": ".txt",
                                  "record_type": "text_document"},
                    ))
                    print(f"Loaded text: {relative_path}")
        except Exception as error:
            print(f"WARNING: Could not load {relative_path}: {error}")
    return documents


def add_normalized_climate_table(documents):
    """Add the eight Table 3 values as one atomic, searchable record."""
    source = "zimbabwe_climate_risk_country_profile.pdf"
    matching = [d for d in documents
                if Path(str(d.metadata.get("source", ""))).name == source
                and d.metadata.get("page") == 12]
    if not matching:
        print("WARNING: Table 3 source page not found; no normalized Table 3 record added.")
        return documents
    text = "\n".join(d.page_content for d in matching).lower()
    if "table 3" not in text or "precipitation" not in text:
        print("WARNING: Table 3 was not identified confidently on PDF page 12.")
        return documents
    table = """NORMALIZED CLIMATE PROJECTION TABLE — TABLE 3
Source: zimbabwe_climate_risk_country_profile.pdf
Table: Table 3, CMIP5 ensemble projection of change in annual precipitation
PDF page number (1-based): 12
PDF page index (0-based): 11
Printed page: 8
Baseline: 1986–2005
Statistic: Median (50th percentile) projected change in annual precipitation, percent relative to baseline.
2040–2059: RCP 2.6 = -1.2%; RCP 4.5 = +1.2%; RCP 6.0 = +4.4%; RCP 8.5 = -4.4%.
2080–2099: RCP 2.6 = +2.8%; RCP 4.5 = +4.1%; RCP 6.0 = +3.0%; RCP 8.5 = -10.7%.
Negative values indicate projected decreases; positive values indicate projected increases. These are projections, not observations or rainfall amounts in millimetres."""
    documents.append(Document(page_content=table, metadata={
        "source": source, "file_type": ".pdf", "page": 12, "page_index": 11,
        "printed_page": 8, "table": "Table 3",
        "record_type": "normalized_climate_table",
    }))
    print("Added normalized Table 3 record (PDF page 12; printed page 8).")
    return documents


def verify_ndc_extract(documents):
    ndc_docs = [d for d in documents if d.metadata.get("record_type") == "ndc_page_extract"]
    if not ndc_docs:
        raise RuntimeError(
            "The structured NDC extract was not loaded. Copy zimbabwe_revised_ndc_evidence.md "
            "into part02_genai/documents before building the index."
        )
    corpus = "\n".join(d.page_content.lower() for d in ndc_docs)
    required = {
        "fall armyworm": "263,000",
        "elephant deaths": "1,500",
        "annual crop losses": "us$126 million",
        "cattle deaths": "3,000",
        "AFOLU share": "54%",
        "mitigation target": "40%",
    }
    missing = [label for label, value in required.items() if value not in corpus]
    if missing:
        raise RuntimeError(f"NDC extract is missing expected evidence markers: {missing}")
    print(f"Verified {len(ndc_docs)} page-labelled NDC evidence records.")


print("Loading local Zimbabwe weather and climate documents...")
documents = load_documents()
if not documents:
    raise RuntimeError(f"No readable TXT, Markdown or PDF documents found in {DOCUMENTS_DIR}")
print(f"Loaded {len(documents)} source records.")
documents = add_normalized_climate_table(documents)
verify_ndc_extract(documents)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1400,
    chunk_overlap=180,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)
if not chunks:
    raise RuntimeError("No text chunks were created.")
print(f"Created {len(chunks)} chunks.")

# Ensure the page-level NDC metadata survives splitting.
ndc_chunks = [c for c in chunks if c.metadata.get("record_type") == "ndc_page_extract"]
if not ndc_chunks:
    raise RuntimeError("NDC page-labelled chunks were lost during splitting.")

print("Loading Ollama embedding model: nomic-embed-text")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("Building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local(str(INDEX_DIR))
print(f"FAISS index saved: {INDEX_DIR}")

# Focused retrieval smoke tests. These verify retrieval presence, not answer correctness.
tests = [
    ("fall armyworm maize production 2016 crop yield loss", "263,000"),
    ("1991/92 drought elephant deaths Hwange Gonarezhou", "1,500"),
    ("2015–2016 drought livestock disease 61% cattle deaths", "61%"),
    ("Zimbabwe annual crop losses US$126 million", "126 million"),
    ("AFOLU share of Zimbabwe GHG emissions in 2017", "54%"),
]
print("\nNDC retrieval smoke tests:")
for query, expected in tests:
    results = vector_store.similarity_search(query, k=5)
    hits = [d for d in results if expected.lower() in d.page_content.lower()]
    status = "PASS" if hits else "CHECK"
    print(f"{status}: {query}")
    if hits:
        for doc in hits[:2]:
            print(f"  source={doc.metadata.get('source')} printed_page={doc.metadata.get('printed_page')}")
    else:
        print("  Expected marker not found in top five; inspect retrieval before deployment.")
print("\nKnowledge-base build completed. Review smoke-test output before using the index.")
