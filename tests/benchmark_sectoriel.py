"""Benchmark sectoriel — compare ProfileStrength générique vs critères adaptatifs.

Pour chaque CV du corpus (15 CVs dans sample_cvs/), compare :
  score_avant : ProfileStrength.score  (scoring universel 8 critères, pré-Phase B)
  score_apres : somme pondérée des criteria_results  (scoring adaptatif, Phase B)
  delta       : score_apres - score_avant

Génère tests/fixtures/benchmark_sectoriel.csv.

Usage :
    python tests/benchmark_sectoriel.py
    make benchmark          # lance aussi benchmark_ats.py
    make benchmark-sectoriel
"""

import csv
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Lexicons must be initialised before any service import.
from src.core.lexicons import init_lexicons

init_lexicons()

from src.services.cv_quality_scorer import CVQualityScorer
from src.services.cv_transformer import CVTransformer
from src.services.nlp_pipeline import NLPPipeline
from src.services.sector_detector import SectorDetector

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample_cvs"
OUTPUT_CSV = Path(__file__).parent / "fixtures" / "benchmark_sectoriel.csv"

FIELDNAMES = [
    "cv_name",
    "secteur_detecte",
    "confidence",
    "profile_id",
    "score_avant",
    "score_apres",
    "level_avant",
    "level_apres",
    "delta",
]

# ---------------------------------------------------------------------------
# Manual classification for the summary report
# ---------------------------------------------------------------------------

TECH_CVS = frozenset({
    "hf_data_data_science.pdf",
    "hf_data_database.pdf",
    "hf_tech_devops.pdf",
    "hf_tech_information_technology.pdf",
    "synth_single_column_backend_engineer.pdf",
    "synth_two_columns_data_analyst.pdf",
})

NON_TECH_CVS = frozenset({
    "hf_engineering_civil_engineer.pdf",
    "hf_engineering_mechanical_engineer.pdf",
    "hf_non-tech_health_and_fitness.pdf",
    "hf_non-tech_human_resources.pdf",
    "hf_sales_business_analyst.pdf",
    "hf_sales_sales.pdf",
})

OTHER_CVS = frozenset({
    "synth_reconversion_fr.pdf",
    "synth_single_column_marketing_manager.pdf",
    "synth_two_columns_product_manager.pdf",
})


def _level(score: int) -> str:
    return "Solide" if score >= 75 else ("Correct" if score >= 50 else "À renforcer")


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def run_benchmark(sample_dir: Path = SAMPLE_DIR) -> list[dict]:
    """Run the sectoriel benchmark on all PDFs in *sample_dir*."""
    # Disable Vision LLM to avoid billing (keeps extraction to pdfplumber/OCR).
    patcher = patch("src.core.config.settings.anthropic_api_key", "")
    patcher.start()

    transformer = CVTransformer()
    nlp = NLPPipeline()
    scorer = CVQualityScorer()
    detector = SectorDetector()  # MiniLM loaded lazily on first detect() call

    rows: list[dict] = []
    pdf_paths = sorted(sample_dir.glob("*.pdf"))

    for pdf_path in pdf_paths:
        start = time.perf_counter()
        try:
            cv = transformer.transform(str(pdf_path))
            parsed_cv = nlp.parse_normalized(cv)
            sector_result = detector.detect(parsed_cv)
            report = scorer.score(parsed_cv, sector_result=sector_result)

            score_avant = report.profile_strength.score
            level_avant = report.profile_strength.level
            score_apres = int(sum(r.weighted_score for r in report.criteria_results))
            level_apres = _level(score_apres)

            rows.append({
                "cv_name": pdf_path.name,
                "secteur_detecte": sector_result.sector,
                "confidence": round(sector_result.confidence, 2),
                "profile_id": sector_result.profile_id,
                "score_avant": score_avant,
                "score_apres": score_apres,
                "level_avant": level_avant,
                "level_apres": level_apres,
                "delta": score_apres - score_avant,
            })
        except Exception as exc:
            rows.append({
                "cv_name": pdf_path.name,
                "secteur_detecte": "ERREUR",
                "confidence": 0.0,
                "profile_id": "error",
                "score_avant": 0,
                "score_apres": 0,
                "level_avant": "ERREUR",
                "level_apres": f"ERREUR : {exc}",
                "delta": 0,
            })

    patcher.stop()
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], output_path: Path = OUTPUT_CSV) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _pad(s: str, w: int) -> str:
    return str(s).ljust(w)


def print_table(rows: list[dict]) -> None:
    col_w = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in FIELDNAMES}
    header = " | ".join(_pad(f, col_w[f]) for f in FIELDNAMES)
    sep = "-+-".join("-" * col_w[f] for f in FIELDNAMES)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(_pad(row[f], col_w[f]) for f in FIELDNAMES))


def print_summary(rows: list[dict]) -> None:
    ok_rows = [r for r in rows if r["profile_id"] != "error"]
    if not ok_rows:
        print("\n⚠️  Aucun CV traité sans erreur.")
        return

    tech   = [r for r in ok_rows if r["cv_name"] in TECH_CVS]
    nontech = [r for r in ok_rows if r["cv_name"] in NON_TECH_CVS]
    other  = [r for r in ok_rows if r["cv_name"] in OTHER_CVS]

    detected = [r for r in ok_rows if r["profile_id"] != "non_detecte"]
    pct_detected = len(detected) / len(ok_rows) * 100 if ok_rows else 0

    def avg(lst, key):
        return round(sum(r[key] for r in lst) / len(lst), 1) if lst else 0.0

    print("\n" + "=" * 60)
    print("RÉSUMÉ BENCHMARK SECTORIEL")
    print("=" * 60)

    print(f"\n{'Groupe':<22} {'N':>3} {'avg avant':>9} {'avg après':>9} {'avg Δ':>7} {'détectés':>9}")
    print("-" * 60)
    for label, group in [("Tech", tech), ("Non-tech", nontech), ("Autre", other), ("TOTAL", ok_rows)]:
        n = len(group)
        if n == 0:
            continue
        det = sum(1 for r in group if r["profile_id"] != "non_detecte")
        print(f"{label:<22} {n:>3} {avg(group,'score_avant'):>9} {avg(group,'score_apres'):>9} "
              f"{avg(group,'delta'):>7} {det}/{n:>3}")

    print(f"\n{'Secteur bien détecté':40} {pct_detected:.0f}% ({len(detected)}/{len(ok_rows)})")

    # Objectifs
    print("\n── Objectifs Phase D ──")

    nt_avg_apres = avg(nontech, "score_apres") if nontech else 0
    obj1 = "✅" if nt_avg_apres >= 50 else "❌"
    print(f"{obj1} CVs non-tech : score_apres moyen = {nt_avg_apres}/100 (cible ≥ 50)")

    if tech:
        regressions = [r for r in tech if r["delta"] < -10]
        obj2 = "✅" if not regressions else f"⚠️  ({len(regressions)} régressions)"
        print(f"{obj2} CVs tech : pas de régression > -10 pts")

    obj3 = "✅" if pct_detected >= 80 else f"❌ ({pct_detected:.0f}% < 80%)"
    print(f"{obj3} Détection secteur ≥ 80% des CVs : {pct_detected:.0f}%")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not SAMPLE_DIR.exists():
        print(f"⚠️  Dossier introuvable : {SAMPLE_DIR}")
        print("Lancez d'abord : python tests/fixtures/download_test_cvs.py")
        sys.exit(1)

    rows = run_benchmark()
    print_table(rows)
    write_csv(rows)
    print_summary(rows)
    print(f"\nRapport CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
