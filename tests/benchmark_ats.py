"""Benchmarks CVTransformer + CVQualityScorer over the sample CV corpus.

Run `python tests/fixtures/download_test_cvs.py` once to populate
tests/fixtures/sample_cvs/, then:

    python tests/benchmark_ats.py
    make benchmark

Writes a CSV report to tests/fixtures/benchmark_report.csv and prints a
summary table.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.cv_quality_scorer import CVQualityScorer
from src.services.cv_transformer import CVTransformer

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample_cvs"
OUTPUT_CSV = Path(__file__).parent / "fixtures" / "benchmark_report.csv"

FIELDNAMES = [
    "cv_name",
    "method",
    "word_count",
    "strength",
    "sections",
    "time_ms",
]


def run_benchmark(
    sample_dir: Path = SAMPLE_DIR, allow_vision_llm: bool = False
) -> list[dict]:
    transformer = CVTransformer()
    scorer = CVQualityScorer()
    rows: list[dict] = []

    if not allow_vision_llm:
        # Niveau 3 (Vision LLM) appelle l'API Claude réelle et est facturé.
        # Désactivé par défaut, comme dans la suite de tests.
        patch("src.core.config.settings.anthropic_api_key", "").start()

    pdf_paths = sorted(sample_dir.glob("*.pdf"))
    for pdf_path in pdf_paths:
        start = time.perf_counter()
        try:
            cv = transformer.transform(str(pdf_path))
            report = scorer.score(cv)
            row = {
                "cv_name": pdf_path.name,
                "method": cv.extraction_method,
                "word_count": cv.word_count,
                "strength": report.profile_strength.level,
                "sections": "|".join(report.ats_readability.sections_found),
                "time_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        except Exception as exc:
            row = {
                "cv_name": pdf_path.name,
                "method": "error",
                "word_count": 0,
                "strength": f"ERROR: {exc}",
                "sections": "",
                "time_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output_path: Path = OUTPUT_CSV) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("Aucun CV trouve dans", SAMPLE_DIR)
        print("Lancez d'abord : python tests/fixtures/download_test_cvs.py")
        return

    widths = {
        field: max(len(field), *(len(str(row[field])) for row in rows))
        for field in FIELDNAMES
    }

    header = " | ".join(field.ljust(widths[field]) for field in FIELDNAMES)
    print(header)
    print("-+-".join("-" * widths[field] for field in FIELDNAMES))
    for row in rows:
        print(" | ".join(str(row[field]).ljust(widths[field]) for field in FIELDNAMES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-vision-llm",
        action="store_true",
        help=(
            "Autorise le niveau 3 de la cascade (Vision LLM, appels API "
            "Claude réels et facturés). Désactivé par défaut."
        ),
    )
    args = parser.parse_args()

    rows = run_benchmark(allow_vision_llm=args.with_vision_llm)
    print_table(rows)
    if rows:
        write_csv(rows)
        print(f"\nRapport CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
