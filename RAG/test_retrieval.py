"""Evaluate retrieval with DeepEval contextual precision and recall using FreeLLMAPI.

Run from this directory with: python test_retrieval.py
Optionally set the number of retrieved nodes: python test_retrieval.py --top-k 5
"""

import argparse
import json
import os
import asyncio
from pathlib import Path
from deepeval.evaluate import AsyncConfig
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel
from openai import OpenAI  # Use standard OpenAI client to communicate with your proxy base URL

from rag_engine import HRAgentEngine


class FreeLLMAPIJudge(DeepEvalBaseLLM):
    """Use FreeLLMAPI's local proxy endpoint as DeepEval's LLM judge."""

    def __init__(self, model="auto"):
        # Pull API credentials mirroring your running HRAgentEngine configurations
        api_key = os.environ.get("FREELLMAPI_API_KEY", "freellmapi-e198ca14b4b28e505d50662534e3c8de8a0d590a7344ed47")
        api_base = os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
        
        self.model = model
        # Configure standard OpenAI client wrapper targeting local proxy endpoints
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def load_model(self):
        return self.client

    def get_model_name(self):
        return f"FreeLLMAPI {self.model}"

    def generate(self, prompt: str, schema: BaseModel = None):
        # Format the user instruction block cleanly for JSON schemas
        if schema is not None:
            schema_json = (schema.model_json_schema() if hasattr(schema, "model_json_schema")
                           else schema.schema())
            prompt += (
                "\n\nYou MUST respond with a valid JSON object matching the following structure schema. "
                "Do NOT wrap it in markdown code blocks or backticks. Return ONLY the raw JSON string text:\n" 
                + json.dumps(schema_json)
            )

        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": f"You are a strict data validation agent. Always respond with pure, valid JSON objects. Here is your evaluation task:\n{prompt}"
                }
            ],
            "temperature": 0,
        }
        
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
            
        response = self.load_model().chat.completions.create(**request)
        
        # Safely index the choice array from the live network response stream
        output = response.choices[0].message.content or ""
        
        if schema is None:
            return output

        # Parse string block safely to JSON
        parsed = json.loads(output.strip())
        if hasattr(schema, "model_validate"):
            return schema.model_validate(parsed)
        return schema.parse_obj(parsed)

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return await asyncio.to_thread(self.generate, prompt, schema)


def node_text(result_node):
    """Return the text of a LlamaIndex retrieval result."""
    node = getattr(result_node, "node", result_node)
    if hasattr(node, "get_content"):
        return node.get_content()
    return str(getattr(node, "text", node))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of nodes to retrieve per query (default: 3)")
    parser.add_argument(
        "--golden-file",
        type=Path,
        default=Path(__file__).parent / "data" / "golden_retrieval.jsonl",
        help="path to the golden JSONL file",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not args.golden_file.is_file():
        parser.error(f"golden file not found: {args.golden_file}")

    cases = [
        json.loads(line)
        for line in args.golden_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    missing_evidence = [case.get("id", "unknown") for case in cases
                        if case.get("relevant_documents") and not case.get("evidence")]
    if missing_evidence:
        parser.error("in-scope golden cases need an 'evidence' value: "
                     + ", ".join(missing_evidence))

    engine = HRAgentEngine()
    # Instantiate custom evaluation judge routing traffic through your proxy container
    judge = FreeLLMAPIJudge()
    precision_metric = ContextualPrecisionMetric(model=judge)
    recall_metric = ContextualRecallMetric(model=judge)
    rows = []
    test_cases = []
    scored_case_ids = []

    print(f"Golden set: {args.golden_file}")
    print(f"Retrieving top {args.top_k} nodes per query\n")

    for case in cases:
        # Retrieve once and reuse these nodes for both evaluation and synthesis.
        retrieval = engine.retrieve(case["query"], similarity_top_k=args.top_k)
        retrieved_nodes = retrieval["nodes"]
        contexts = [node_text(node) for node in retrieved_nodes]

        relevant_documents = case.get("relevant_documents", [])

        if relevant_documents:
            generated = engine.generate(case["query"], retrieved_nodes)
            test_case = LLMTestCase(
                input=case["query"],
                actual_output=generated.get("answer", ""),
                expected_output=case["evidence"],
                retrieval_context=contexts,
            )
            test_cases.append(test_case)
            scored_case_ids.append(case.get("id", ""))
            precision = recall = None
            p_reason = r_reason = ""
        else:
            precision = recall = None
            p_reason = r_reason = ""

        rows.append({
            "id": case.get("id", ""),
            "precision": precision,
            "recall": recall,
            "retrieved_nodes": len(contexts),
            "precision_reason": p_reason,
            "recall_reason": r_reason,
        })
    print(f"Processed {len(rows)} test cases, {len(test_cases)} in-scope for evaluation.\n")
    if test_cases:
        

        # Configure the execution pacing parameters
        async_config = AsyncConfig(
            run_async=True,       # Keeps async processing active
            max_concurrent=10,    # Limits maximum parallel test cases to 10 at any given time
            throttle_value=2      # Introduces a 2-second sleep delay between case starts to prevent API rate limits
        )
        evaluation = evaluate(
            test_cases=test_cases,
            metrics=[precision_metric, recall_metric],
            async_config=async_config,
        )
        rows_by_id = {row["id"]: row for row in rows}
        for case_id, test_result in zip(scored_case_ids, evaluation.test_results):
            row = rows_by_id[case_id]
            for metric_data in test_result.metrics_data or []:
                metric_name = metric_data.name.lower()
                if "precision" in metric_name:
                    row["precision"] = metric_data.score
                    row["precision_reason"] = metric_data.reason or ""
                elif "recall" in metric_name:
                    row["recall"] = metric_data.score
                    row["recall_reason"] = metric_data.reason or ""

    for row in rows:
        if row["precision"] is None:
            status = "N/A (no reference evidence)"
        else:
            status = f"P={row['precision']:.1%} R={row['recall']:.1%}"
        print(f"{row['id'] or 'unknown'}: {status}")

    results = pd.DataFrame(rows)
    scored = results.dropna(subset=["precision", "recall"])
    print("\nSummary (DeepEval contextual metrics)")
    if scored.empty:
        print("  No in-scope cases to score.")
    else:
        print(f"  Mean contextual precision: {scored['precision'].mean():.1%}")
        print(f"  Mean contextual recall:    {scored['recall'].mean():.1%}")
        print(f"  Scored cases: {len(scored)}/{len(results)}")


if __name__ == "__main__":
    main()
