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


CALIBRATION_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Synth CVs with clearly tech vocabulary — used to detect false positives
# (non-tech ESCO phrases semantically matched against a tech CV).
SYNTH_TECH_CVS = [
    "synth_single_column_backend_engineer.pdf",
    "synth_two_columns_data_analyst.pdf",
]


def run_calibration(sample_dir: Path = SAMPLE_DIR) -> None:
    """Compare semantic skill match thresholds on the benchmark corpus.

    For each candidate threshold: average ProfileStrength score for HF CVs,
    average for synth CVs, and the count of semantically-matched ESCO
    phrases on SYNTH_TECH_CVS (proxy for false positives — these CVs are
    already well-scored via substring matching, so new semantic matches
    there are likely noise).
    """
    transformer = CVTransformer()
    scorer = CVQualityScorer()
    patch("src.core.config.settings.anthropic_api_key", "").start()

    cvs = {}
    for pdf_path in sorted(sample_dir.glob("*.pdf")):
        cvs[pdf_path.name] = transformer.transform(str(pdf_path))

    print(
        f"{'threshold':>9} | {'avg HF':>6} | {'avg synth':>9} | {'tech false positives':>20}"
    )
    print("-" * 9 + "-+-" + "-" * 6 + "-+-" + "-" * 9 + "-+-" + "-" * 20)

    for threshold in CALIBRATION_THRESHOLDS:
        hf_scores: list[int] = []
        synth_scores: list[int] = []
        false_positives = 0

        for name, cv in cvs.items():
            skills = transformer._parse_skills(
                cv.raw_text, semantic_threshold=threshold
            )
            cv_variant = cv.model_copy(update={"skills": skills})
            score = scorer.score(cv_variant).profile_strength.score

            if name.startswith("hf_"):
                hf_scores.append(score)
            elif name.startswith("synth_"):
                synth_scores.append(score)

            if name in SYNTH_TECH_CVS:
                false_positives += len(skills.other)

        avg_hf = round(sum(hf_scores) / len(hf_scores), 1) if hf_scores else 0.0
        avg_synth = (
            round(sum(synth_scores) / len(synth_scores), 1) if synth_scores else 0.0
        )
        print(f"{threshold:>9} | {avg_hf:>6} | {avg_synth:>9} | {false_positives:>20}")


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
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Compare les seuils de matching sémantique des compétences "
            f"({CALIBRATION_THRESHOLDS}) au lieu de lancer le benchmark normal."
        ),
    )
    args = parser.parse_args()

    if args.calibrate:
        run_calibration()
        return

    rows = run_benchmark(allow_vision_llm=args.with_vision_llm)
    print_table(rows)
    if rows:
        write_csv(rows)
        print(f"\nRapport CSV : {OUTPUT_CSV}")
        from src.services.experiment_tracker import ExperimentTracker
        ExperimentTracker().log_benchmark(
            str(OUTPUT_CSV),
            params={"allow_vision_llm": str(args.with_vision_llm)},
            metrics={
                "cv_count": float(len(rows)),
                "avg_time_ms": round(sum(r["time_ms"] for r in rows) / len(rows), 1),
                "error_count": float(sum(1 for r in rows if r["method"] == "error")),
            },
        )


if __name__ == "__main__":
    main()
