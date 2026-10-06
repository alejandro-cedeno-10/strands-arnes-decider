"""Amazon Bedrock models shared by the examples (the article's System 2).

- `nova_lite()` is the routine model (`us.amazon.nova-lite-v1:0`).
- `nova_pro()` is the advanced model (`us.amazon.nova-pro-v1:0`).
- Credentials and region come from the environment: `AWS_PROFILE` and `AWS_REGION` (default us-east-1).
- `strands.models.BedrockModel` is used as is; the subclass only adds a usage ledger: when
  `BEDROCK_USAGE_LEDGER` points to a file, every model call appends one JSON line with the token
  counts Bedrock returned in `usage` (not tokenizer estimates). Without that variable it behaves
  exactly like `BedrockModel`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from strands.models import BedrockModel

NOVA_LITE = "us.amazon.nova-lite-v1:0"
NOVA_PRO = "us.amazon.nova-pro-v1:0"
REGION = os.getenv("AWS_REGION", "us-east-1")


class LedgerBedrockModel(BedrockModel):
    """BedrockModel that writes the real `usage` of every call to `BEDROCK_USAGE_LEDGER`."""

    async def stream(self, *args: Any, **kwargs: Any):
        async for event in super().stream(*args, **kwargs):
            usage = event.get("metadata", {}).get("usage") if isinstance(event, dict) else None
            if usage:
                self._record(usage)
            yield event

    def _record(self, usage: dict[str, Any]) -> None:
        path = os.getenv("BEDROCK_USAGE_LEDGER")
        if not path:
            return
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "",
            "tag": os.getenv("BEDROCK_USAGE_TAG", ""),
            "model_id": self.config["model_id"],
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")


def nova_lite(**config: Any) -> BedrockModel:
    return LedgerBedrockModel(model_id=NOVA_LITE, region_name=REGION, **config)


def nova_pro(**config: Any) -> BedrockModel:
    return LedgerBedrockModel(model_id=NOVA_PRO, region_name=REGION, **config)
