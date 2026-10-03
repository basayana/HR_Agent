"""Evaluate retrieval with DeepEval contextual precision and recall using FreeLLMAPI.

Run from this directory with: python test_retrieval.py
Set TOP_K and GOLDEN_FILE below to change the retrieval settings.
"""

import asyncio
import json
import os
from statistics import mean
from pathlib import Path
from deepeval.evaluate import AsyncConfig
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel
from openai import OpenAI  # Use standard OpenAI client to communicate with your proxy base URL

from rag_engine import HRAgentEngine


TOP_K = 3
GOLDEN_FILE = Path(__file__).parent / "data" / "golden_retrieval.jsonl"


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
        # pass


def node_text(result_node):
    """Return the text of a LlamaIndex retrieval result."""
    node = getattr(result_node, "node", result_node)
    if hasattr(node, "get_content"):
        return node.get_content()
    return str(getattr(node, "text", node))


def main():
    if not GOLDEN_FILE.is_file():
        raise FileNotFoundError(f"golden file not found: {GOLDEN_FILE}")

    cases = [
        json.loads(line)
        for line in GOLDEN_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    missing_evidence = [case.get("id", "unknown") for case in cases
                        if case.get("relevant_documents") and not case.get("evidence")]
    if missing_evidence:
        raise ValueError("in-scope golden cases need an 'evidence' value: "
                         + ", ".join(missing_evidence))

    engine = HRAgentEngine()
    # Instantiate custom evaluation judge routing traffic through your proxy container
    judge = FreeLLMAPIJudge()
    precision_metric = ContextualPrecisionMetric(model=judge)
    recall_metric = ContextualRecallMetric(model=judge)
    test_cases = []
    reports = []

    print(f"Golden set: {GOLDEN_FILE}")
    print(f"Retrieving top {TOP_K} nodes per query\n")

    for case in cases:
        # Retrieve once and reuse these nodes for both evaluation and synthesis.
        retrieval = engine.retrieve(case["query"], similarity_top_k=TOP_K)
        retrieved_nodes = retrieval["nodes"]
        contexts = [node_text(node) for node in retrieved_nodes]

        relevant_documents = case.get("relevant_documents", [])

        if relevant_documents:
            test_case = LLMTestCase(
                input=case["query"],
                actual_output='Ignore this as we are only evaluating retrieval, not generation.',
                expected_output=case["evidence"],
                retrieval_context=contexts,
            )
            test_cases.append(test_case)

        reports.append({
            "id": case.get("id", ""),
            "precision": None,
            "recall": None,
        })
    print(f"Processed {len(reports)} test cases, {len(test_cases)} in-scope for evaluation.\n")
    if test_cases:
        async_config = AsyncConfig(
            run_async=True,
            max_concurrent=10,
            throttle_value=2,
        )
        evaluation = evaluate(
            test_cases=test_cases,
            metrics=[precision_metric, recall_metric],
            async_config=async_config,
        )
        scored_reports = (report for report, case in zip(reports, cases)
                          if case.get("relevant_documents"))
        for report, test_result in zip(scored_reports, evaluation.test_results):
            for metric_data in test_result.metrics_data or []:
                metric_name = metric_data.name.lower()
                if "precision" in metric_name:
                    report["precision"] = metric_data.score
                elif "recall" in metric_name:
                    report["recall"] = metric_data.score

    for report in reports:
        if report["precision"] is None:
            status = "N/A (no reference evidence)"
        else:
            status = f"P={report['precision']:.1%} R={report['recall']:.1%}"
        print(f"{report['id'] or 'unknown'}: {status}")

    scored = [report for report in reports
              if report["precision"] is not None and report["recall"] is not None]
    print("\nSummary (DeepEval contextual metrics)")
    if not scored:
        print("  No in-scope cases to score.")
    else:
        print(f"  Mean contextual precision: {mean(r['precision'] for r in scored):.1%}")
        print(f"  Mean contextual recall:    {mean(r['recall'] for r in scored):.1%}")
        print(f"  Scored cases: {len(scored)}/{len(reports)}")


if __name__ == "__main__":
    main()
