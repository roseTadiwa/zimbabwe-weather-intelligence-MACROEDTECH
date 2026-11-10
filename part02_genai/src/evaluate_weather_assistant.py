
from __future__ import annotations

import csv
import json
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# This script lives beside integrated_weather_assistant.py, so its folder is
# automatically on sys.path when executed using the instructions in the header.
import integrated_weather_assistant as assistant
from langchain_ollama import ChatOllama


@dataclass
class EvaluationRecord:
    test_id: str
    category: str
    question: str
    answer: str
    expected_terms_found: list[str]
    expected_terms_missing: list[str]
    keyword_coverage: float
    has_evidence_citations: bool
    has_source_metadata: bool
    answer_not_empty: bool
    passed: bool
    notes: str


TEST_CASES: list[dict[str, Any]] = [
    {
        "test_id": "RAG-001",
        "category": "drought_impacts",
        "question": "What are the main consequences of drought in Zimbabwe?",
        "expected_terms": ["water", "livelihood", "poverty", "agriculture"],
    },
    {
        "test_id": "RAG-002",
        "category": "climate_change",
        "question": "How is climate change expected to affect Zimbabwe?",
        "expected_terms": ["temperature", "rainfall", "drought", "flood"],
    },
    {
        "test_id": "RAG-003",
        "category": "historical_climate",
        "question": "What historical changes in temperature or rainfall have been reported in Zimbabwe?",
        "expected_terms": ["temperature", "rainfall"],
    },
    {
        "test_id": "RAG-004",
        "category": "water_resources",
        "question": "What risks does drought or reduced rainfall pose to water resources in Zimbabwe?",
        "expected_terms": ["water", "rainfall"],
    },
]

CITATION_PATTERN = re.compile(r"\[E\d+(?:;[^\]]*)?\]", re.IGNORECASE)
SOURCE_PATTERN = re.compile(r"Source\s*:", re.IGNORECASE)


def evaluate_answer(test_case: dict[str, Any], answer: str) -> EvaluationRecord:
    normalized_answer = answer.lower()
    expected_terms = [term.lower() for term in test_case["expected_terms"]]
    found = [term for term in expected_terms if term in normalized_answer]
    missing = [term for term in expected_terms if term not in normalized_answer]
    coverage = len(found) / len(expected_terms) if expected_terms else 1.0
    has_citations = bool(CITATION_PATTERN.search(answer))
    has_source_metadata = bool(SOURCE_PATTERN.search(answer))
    non_empty = bool(answer.strip())

    # These are transparent, lightweight automated checks, not a substitute for
    # human review of factual accuracy or whether each citation supports its claim.
    passed = non_empty and has_citations and has_source_metadata and coverage >= 0.5
    notes = "Automated checks passed; review factual grounding manually." if passed else (
        "Review answer: " + ", ".join(
            label for label, condition in [
                ("empty answer", not non_empty),
                ("missing evidence citation", not has_citations),
                ("missing source metadata", not has_source_metadata),
                ("low expected-term coverage", coverage < 0.5),
            ] if condition
        )
    )

    return EvaluationRecord(
        test_id=test_case["test_id"],
        category=test_case["category"],
        question=test_case["question"],
        answer=answer,
        expected_terms_found=found,
        expected_terms_missing=missing,
        keyword_coverage=round(coverage, 3),
        has_evidence_citations=has_citations,
        has_source_metadata=has_source_metadata,
        answer_not_empty=non_empty,
        passed=passed,
        notes=notes,
    )


def main() -> int:
    project_root = Path(assistant.PROJECT_ROOT)
    output_dir = project_root / "part02_genai" / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 66)
    print("ZIMBABWE WEATHER ASSISTANT — RESPONSE EVALUATION")
    print("=" * 66)
    print("Loading local FAISS knowledge base...")
    retriever = assistant.load_knowledge_base()
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    print("Knowledge base loaded. Running research question tests.\n")

    records: list[EvaluationRecord] = []
    for case in TEST_CASES:
        print(f"[{case['test_id']}] {case['question']}")
        try:
            answer = assistant.answer_document_research_question(
                llm=llm,
                retriever=retriever,
                question=case["question"],
                history=[],
            )
            record = evaluate_answer(case, answer)
        except Exception as exc:
            record = EvaluationRecord(
                test_id=case["test_id"],
                category=case["category"],
                question=case["question"],
                answer="",
                expected_terms_found=[],
                expected_terms_missing=[term.lower() for term in case["expected_terms"]],
                keyword_coverage=0.0,
                has_evidence_citations=False,
                has_source_metadata=False,
                answer_not_empty=False,
                passed=False,
                notes=f"Execution error: {type(exc).__name__}: {exc}",
            )
            print(traceback.format_exc())

        records.append(record)
        print(f"Result: {'PASS' if record.passed else 'REVIEW'}")
        print(f"Citation present: {record.has_evidence_citations}; source metadata: {record.has_source_metadata}; keyword coverage: {record.keyword_coverage:.0%}")
        print(f"Answer preview: {record.answer[:350].replace(chr(10), ' ')}")
        print()

    passed_count = sum(record.passed for record in records)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "llama3.2:3b",
        "knowledge_base": "local FAISS index",
        "tests_run": len(records),
        "tests_passed_automated_checks": passed_count,
        "tests_flagged_for_review": len(records) - passed_count,
        "pass_rate": round(passed_count / len(records), 3) if records else 0.0,
        "evaluation_note": (
            "Automated screening only. A PASS does not prove factual correctness; "
            "manually verify each claim against the cited source passage."
        ),
        "results": [asdict(record) for record in records],
    }

    json_path = output_dir / f"weather_assistant_evaluation_{timestamp}.json"
    csv_path = output_dir / f"weather_assistant_evaluation_{timestamp}.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = list(asdict(records[0]).keys()) if records else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["expected_terms_found"] = "; ".join(row["expected_terms_found"])
            row["expected_terms_missing"] = "; ".join(row["expected_terms_missing"])
            writer.writerow(row)

    print("=" * 66)
    print(f"Tests run: {len(records)}")
    print(f"Automated checks passed: {passed_count}")
    print(f"Flagged for review: {len(records) - passed_count}")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    print("Remember: manually review factual accuracy and citation support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
