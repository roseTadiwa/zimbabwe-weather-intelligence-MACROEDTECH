
import csv
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "part02_genai" / "reports" / "evaluation"

REVIEWS = [
    {
        "test_id": "RAG-001",
        "question": "What are the main consequences of drought in Zimbabwe?",
        "automated_result": "FAIL",
        "manual_grounding_result": "PARTIALLY SUPPORTED - FACTUAL ERROR",
        "source_reference": "Climate Risk Country Profile: Zimbabwe (2021), PDF page 13; printed page 9; Table 5",
        "supported_claims": "Drought-related economic losses and affected populations are discussed. Food insecurity and unemployment are discussed in the broader context of climate change and increasingly arid conditions (PDF page 16; printed page 12).",
        "unsupported_or_inaccurate_claims": "The answer incorrectly says 7,000 people were affected. The source says 7,000 was the total number of deaths across droughts, epidemics, floods and storms. It also incorrectly attributes US$950 million in total damage across all hazards to drought alone.",
        "source_verified_facts": "All hazards combined: 7,000 deaths, more than 20 million affected, US$950 million total damage. Drought-specific table: approximately 19.1 million affected and US$551 million damage. The narrative says the 2007 drought affected 6 million people and the 2013 drought caused up to US$500 million in damage while affecting over 4 million residents.",
        "recommended_action": "Keep flagged. Correct attribution of the 7,000 and US$950 million figures. Distinguish event impacts from broader climate-change impacts.",
        "final_status": "NEEDS CORRECTION",
    },
    {
        "test_id": "RAG-002",
        "question": "How is climate change expected to affect Zimbabwe?",
        "automated_result": "PASS",
        "manual_grounding_result": "MOSTLY SUPPORTED - NUMERICAL QUALIFICATION NEEDED",
        "source_reference": "Climate Risk Country Profile: Zimbabwe (2021), PDF page 14; printed page 10; and PDF page 20; printed page 16",
        "supported_claims": "Under RCP8.5, annual likelihood of severe drought is projected to increase by 21% in 2040-2059 and 47% in 2080-2099. Days per year with maximum temperature above 35°C are projected to increase by 39 and 108 days. Consecutive dry spells are projected to increase by 13 and 25 days.",
        "unsupported_or_inaccurate_claims": "The phrase 'runoff declines by 60% to 40%' is inaccurate if used in the answer. The source range is -60% to +40% across basins and scenarios, so some scenarios show increases. Avoid treating all sector impacts as equally quantified projections.",
        "source_verified_facts": "Drought, extreme-temperature and dry-spell projections appear on PDF page 14. Mean annual runoff projections across seven basins range from -60% to +40% on PDF page 20.",
        "recommended_action": "Keep automated pass but add a manual qualification flag. State the runoff range as -60% to +40% and explain direction varies by basin/scenario.",
        "final_status": "PASS WITH QUALIFICATION",
    },
    {
        "test_id": "RAG-003",
        "question": "What historical changes in temperature or rainfall have been reported in Zimbabwe?",
        "automated_result": "PASS",
        "manual_grounding_result": "SUPPORTED",
        "source_reference": "Climate Risk Country Profile: Zimbabwe (2021), PDF page 9; printed page 5",
        "supported_claims": "Mean annual precipitation has shown a decreasing trend; late onset and early cessation of the rainy/growing season have been observed across agro-ecological regions; mean annual temperature increased by roughly 0.03°C/year from 1970 to 2016.",
        "unsupported_or_inaccurate_claims": "No material discrepancy identified in the answer summary reviewed.",
        "source_verified_facts": "The three reported historical trends are explicitly stated in the source's Key Climate Trends section.",
        "recommended_action": "Keep the automated pass. Preserve the time period and describe the temperature figure as a historical trend, not a projection.",
        "final_status": "PASS",
    },
    {
        "test_id": "RAG-004",
        "question": "What risks does drought or reduced rainfall pose to water resources in Zimbabwe?",
        "automated_result": "PASS",
        "manual_grounding_result": "SUPPORTED",
        "source_reference": "Climate Risk Country Profile: Zimbabwe (2021), PDF page 19; printed page 15",
        "supported_claims": "Lower rainfall can negatively affect groundwater recharge and availability; runoff changes can limit hydropower generation; higher temperatures can increase water demand for agriculture and energy; reduced rainfall can increase water-treatment and wastewater-management costs; climate change may intensify droughts and floods.",
        "unsupported_or_inaccurate_claims": "No material discrepancy identified in the answer summary reviewed.",
        "source_verified_facts": "These impacts are directly described in the Water section on PDF page 19 (printed page 15).",
        "recommended_action": "Keep the automated pass. Distinguish physical PDF page numbers from printed page numbers in citations.",
        "final_status": "PASS",
    },
]


def latest_automated_json():
    files = sorted(
        EVALUATION_DIR.glob("weather_assistant_evaluation_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def main():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_json = latest_automated_json()
    summary = {
        "tests_reviewed": len(REVIEWS),
        "needs_correction": sum(r["final_status"] == "NEEDS CORRECTION" for r in REVIEWS),
        "pass_with_qualification": sum(r["final_status"] == "PASS WITH QUALIFICATION" for r in REVIEWS),
        "pass": sum(r["final_status"] == "PASS" for r in REVIEWS),
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "review_type": "source-grounded manual review",
        "source_document": "Climate Risk Country Profile: Zimbabwe (World Bank Group, 2021)",
        "source_automated_evaluation_file": str(source_json) if source_json else None,
        "note": "This manual review is separate from automated keyword checks. It does not rerun the model. Recheck the original answer if it has since changed.",
        "summary": summary,
        "results": REVIEWS,
    }
    json_path = EVALUATION_DIR / f"weather_assistant_manual_review_{stamp}.json"
    csv_path = EVALUATION_DIR / f"weather_assistant_manual_review_{stamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    columns = list(REVIEWS[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(REVIEWS)

    print("Manual review reports created.")
    print(f"Automated report referenced: {source_json or 'No existing automated JSON found'}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("Summary:", summary)
    print("Original results and the assistant are unchanged.")


if __name__ == "__main__":
    main()
