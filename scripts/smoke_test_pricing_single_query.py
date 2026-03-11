"""Run a single-query pricing smoke test for one RAG + one model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.model_provider import MODELS_REGISTRY, create_llm
from src.rag.simple import query_for_evaluation as simple_query_for_evaluation
from src.rag.hybrid import query_for_evaluation as hybrid_query_for_evaluation
from src.rag.hybrid_rrf import query_for_evaluation as hybrid_rrf_query_for_evaluation
from src.rag.hyde import query_for_evaluation as hyde_query_for_evaluation
from src.rag.rewriter import query_for_evaluation as rewriter_query_for_evaluation
from src.rag.pageindex import query_for_evaluation as pageindex_query_for_evaluation


RAG_QUERY_FUNCTIONS = {
    "simple": lambda q, llm: simple_query_for_evaluation(q, custom_llm=llm),
    "hybrid": lambda q, llm: hybrid_query_for_evaluation(q, custom_llm=llm),
    "hybrid-rrf": lambda q, llm: hybrid_rrf_query_for_evaluation(q, custom_llm=llm),
    "hyde": lambda q, llm: hyde_query_for_evaluation(q, custom_hyde_llm=llm, custom_answer_llm=llm),
    "rewriter": lambda q, llm: rewriter_query_for_evaluation(q, custom_rewriter_llm=llm, custom_answer_llm=llm),
    "pageindex": lambda q, llm: pageindex_query_for_evaluation(q, custom_llm=llm),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-query pricing smoke test")
    parser.add_argument("--rag", choices=sorted(RAG_QUERY_FUNCTIONS.keys()), default="simple")
    parser.add_argument("--model", choices=sorted(MODELS_REGISTRY.keys()), default="gpt-5")
    parser.add_argument(
        "--question",
        default="Cual es la semana ideal para iniciar los controles prenatales",
    )
    parser.add_argument(
        "--require-positive-cost",
        action="store_true",
        help="Fail with exit code 1 if total_cost <= 0.",
    )
    args = parser.parse_args()

    llm = create_llm(MODELS_REGISTRY[args.model])
    result = RAG_QUERY_FUNCTIONS[args.rag](args.question, llm)

    metadata = result.get("metadata", {})
    output = {
        "rag": args.rag,
        "model": args.model,
        "question": args.question,
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "total_cost": metadata.get("total_cost"),
        "cost_source": metadata.get("cost_source"),
        "execution_time": metadata.get("execution_time"),
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

    total_cost = float(metadata.get("total_cost", 0.0) or 0.0)
    if args.require_positive_cost and total_cost <= 0:
        print("[FAIL] total_cost <= 0.0")
        return 1

    print("[OK] smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
