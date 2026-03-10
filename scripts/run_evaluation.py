"""Official CLI entrypoint for RAGAS benchmark evaluations."""

import sys
from pathlib import Path

# Make script execution robust regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ragas_evaluator import main


if __name__ == "__main__":
    main()
