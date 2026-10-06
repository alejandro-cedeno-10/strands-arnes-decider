"""Local test of the Lambda handler with a STUB engine (does not load the model or torch).

Checks: direct invocation, HTTP event (Function URL), 422 on schema errors, 422 on engine ValueError,
an error on an invalid direct invocation, and that the engine is loaded only once.

    python deploy/lambda/test_handler.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import handler  # noqa: E402
from strands_decider.schema import NoulAnswer, SystemOneResponse, Usage  # noqa: E402

QUESTIONS = {"q": {"type": "noul", "instructions": "Urgent?"}}


class StubEngine:
    loads = 0

    def evaluate(self, request):
        if "boom" in str(request.state):
            raise ValueError("option span has no tokens left")
        return SystemOneResponse(model="STUB", answers={k: NoulAnswer(noul=0.5) for k in request.questions},
                                 usage=Usage(input_tokens=1, output_tokens=len(request.questions)))


def fake_engine():
    if handler._ENGINE is None:
        StubEngine.loads += 1
        handler._ENGINE = StubEngine()
    return handler._ENGINE


def main() -> None:
    handler._engine = fake_engine
    direct = handler.lambda_handler({"state": "hi", "questions": QUESTIONS}, None)
    assert direct["answers"]["q"]["noul"] == 0.5 and "latency_ms" in direct and direct["cold_start_load_ms"] >= 0, direct
    http = handler.lambda_handler({"body": json.dumps({"state": "hi", "questions": QUESTIONS})}, None)
    assert http["statusCode"] == 200 and json.loads(http["body"])["cold_start_load_ms"] == 0.0, http
    schema_error = handler.lambda_handler(
        {"body": json.dumps({"state": "hi", "questions": {"c": {"type": "choice", "instructions": "x"}}})}, None)
    assert schema_error["statusCode"] == 422, schema_error
    engine_error = handler.lambda_handler({"body": json.dumps({"state": "boom", "questions": QUESTIONS})}, None)
    assert engine_error["statusCode"] == 422 and "no tokens left" in engine_error["body"], engine_error
    try:
        handler.lambda_handler({"state": "", "questions": QUESTIONS}, None)
        raise AssertionError("an invalid direct invocation should raise")
    except ValueError:
        pass
    assert StubEngine.loads == 1
    print("handler OK: direct, Function URL, schema 422, engine 422, invalid direct invocation -> error, 1 load")


if __name__ == "__main__":
    main()
