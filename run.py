"""Cross-platform task runner (works on Windows without `make`).

    python run.py data    # fetch ~10K recent cs.LG abstracts from the arXiv API
    python run.py index   # chunk + embed abstracts into artifacts/
    python run.py eval    # known-item evaluation of 4 retrievers -> reports/
    python run.py significance  # paired bootstrap CIs from saved per-query ranks
    python run.py test    # lint + unit tests
    python run.py app     # launch the Streamlit demo
    python run.py all     # data + index + eval
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def main(cmd: str) -> int:
    if cmd in ("data", "all"):
        from paperqa.ingest import main as ingest

        ingest()
    if cmd in ("index", "all"):
        from paperqa.index import build_index

        build_index()
    if cmd in ("eval", "all"):
        from paperqa.evaluate import run_evaluation

        run_evaluation()
    if cmd in ("eval", "all", "significance"):
        from paperqa.significance import run as significance

        significance()
    if cmd == "test":
        lint = subprocess.call([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT)
        return lint or subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if cmd == "app":
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py")],
            cwd=ROOT,
        )
    if cmd not in ("data", "index", "eval", "all", "significance"):
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
