"""Serve Strands Decider on a CPU instance (EC2 Graviton) with the official FastAPI app.

Two changes over `strands-decider serve`:
- The torso stays in bf16 on CPU instead of being upcast to fp32. On Graviton3 (c7g/m7g) bf16 was
  about 2x faster than fp32 and used 3.6 GB instead of 11.7 GB, with the same answers.
- Every evaluation runs under one lock. strands-decider 0.1.0 keeps per-request state in a shared
  attribute (upstream issue #17): concurrent requests can return wrong answers with HTTP 200.
  FastAPI runs sync endpoints in a thread pool, so a single worker alone does not prevent it.

Usage:
    python deploy/ec2/serve.py [--checkpoint StrandsAgents/strands-decider-2B-hobson-v19]
                               [--host 127.0.0.1] [--port 8099] [--threads N]
Set DECIDER_FP32=1 to keep the library default (fp32 on CPU).
"""

from __future__ import annotations

import argparse
import os
import threading

import torch


def patch_engine() -> None:
    from strands_decider.infer import SystemOneEngine

    if os.getenv("DECIDER_FP32") != "1":
        SystemOneEngine._upcast_torso_for_cpu = lambda self: None  # type: ignore[method-assign]
    lock = threading.Lock()
    evaluate = SystemOneEngine.evaluate

    def serialized(self, request):
        with lock:
            return evaluate(self, request)

    SystemOneEngine.evaluate = serialized  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.getenv("DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-hobson-v19"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    patch_engine()

    import uvicorn
    from strands_decider.server import create_app

    app = create_app(args.checkpoint, device="cpu")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
