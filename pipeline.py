"""
pipeline.py — End-to-end ML pipeline orchestrator
===================================================
Run this single script to execute all 6 phases in order:

  Phase 1 → Data generation (generate_data.py)
  Phase 2 → Data cleaning   (preprocess.py)
  Phase 3 → EDA             (03_eda.py)
  Phase 4 → Feature engineering (build_features.py)
  Phase 5 → Model training  (05_modeling.py)
  Phase 6 → Evaluation & dashboard export (06_evaluation.py)

Usage:
  python pipeline.py           # run all phases
  python pipeline.py --from 4  # resume from phase 4
  python pipeline.py --to   3  # run only phases 1-3

After completion:
  streamlit run dashboard/app.py
  open dashboard/dashboard_index.html  (requires a local HTTP server, see README)
"""

import argparse
import sys
import time
import traceback
from pathlib import Path


def banner(phase: int, title: str):
    print(f"\n{'='*65}")
    print(f"  PHASE {phase} — {title}")
    print(f"{'='*65}")


def run_phase(phase_num: int, title: str, fn) -> bool:
    """Run a phase, catch errors, return True on success."""
    banner(phase_num, title)
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n  ✅ Phase {phase_num} complete in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\n  ❌ Phase {phase_num} FAILED: {e}")
        traceback.print_exc()
        return False


def phase1_generate():
    from src.data.generate_data import main as gen
    gen()


def phase2_clean():
    from notebooks.n02_data_cleaning import main as clean
    clean()


def phase3_eda():
    from notebooks.n03_eda import main as eda
    eda()


def phase4_features():
    from notebooks.n04_feature_engineering import main as feats
    feats()


def phase5_model():
    from notebooks.n05_modeling import main as model
    model()


def phase6_evaluate():
    from notebooks.n06_evaluation import main as evaluate
    evaluate()


def main():
    # Direct imports (not module-style) so we can run from project root
    import importlib.util, types

    def load_nb(path: str, alias: str):
        """Load a notebook .py file as a module by path (avoids naming conflicts)."""
        spec   = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module

    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="E-Commerce ML Pipeline")
    parser.add_argument("--from", dest="from_phase", type=int, default=1,
                        help="Start from this phase (1-6, default 1)")
    parser.add_argument("--to",   dest="to_phase",   type=int, default=6,
                        help="Stop after this phase  (1-6, default 6)")
    args = parser.parse_args()

    phases = [
        (1, "Data Generation",    lambda: load_nb(str(root/"src"/"data"/"generate_data.py"), "gen").main()),
        (2, "Data Cleaning",      lambda: load_nb(str(root/"notebooks"/"02_data_cleaning.py"),  "nb02").main()),
        (3, "Exploratory Analysis",lambda: load_nb(str(root/"notebooks"/"03_eda.py"),           "nb03").main()),
        (4, "Feature Engineering",lambda: load_nb(str(root/"notebooks"/"04_feature_engineering.py"),"nb04").main()),
        (5, "Model Development",  lambda: load_nb(str(root/"notebooks"/"05_modeling.py"),       "nb05").main()),
        (6, "Evaluation & Export",lambda: load_nb(str(root/"notebooks"/"06_evaluation.py"),     "nb06").main()),
    ]

    overall_start = time.time()
    failures = []

    for num, title, fn in phases:
        if num < args.from_phase or num > args.to_phase:
            continue
        ok = run_phase(num, title, fn)
        if not ok:
            failures.append(num)

    elapsed = time.time() - overall_start
    print(f"\n{'='*65}")
    if not failures:
        print(f"  🎉 All phases complete in {elapsed:.1f}s")
        print()
        print("  Next steps:")
        print("    streamlit run dashboard/app.py")
        print("    Open dashboard/dashboard_index.html in a browser")
        print(f"{'='*65}\n")
    else:
        print(f"  ⚠️  Pipeline finished with failures in phases: {failures}")
        print(f"{'='*65}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
