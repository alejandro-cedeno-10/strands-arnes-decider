"""Fine-tune only the readout head of Strands Decider v19 on your own labelled examples.

The torso and its LoRA adapter stay frozen, so their outputs never change during training. The
script therefore runs the 2B torso ONCE over the training examples (with shuffled option order),
caches the pooled `<answer>` state and the option states, and trains the small pointer head on
that cache for many epochs in seconds. The head starts from the v19 weights, not from scratch.
The calib split selects the best epoch. The result is a full checkpoint directory (v19 files plus
the new head.safetensors) that `load_engine` and `strands-decider serve` accept.
`strands-decider calibrate` needs CUDA in 0.1.0; on CPU, choose thresholds from finetune/evaluate.py.

Usage:
    python finetune/tune_head.py --out results/finetune/checkpoint [--epochs 40] [--lr 3e-4]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"
DATA = Path(__file__).resolve().parents[1] / "results" / "finetune"


def cache_features(model, loader) -> list[dict]:
    """Run the frozen torso once and keep what the head reads: pooled state and option states."""
    from strands_decider.modeling import gather_options, pool_last_token

    cached = []
    with torch.no_grad():
        for batch in loader:
            hidden = model.encode(batch["input_ids"], batch["attention_mask"])
            cached.append({
                "pooled": pool_last_token(hidden, batch["attention_mask"]).to(torch.float32),
                "options": gather_options(hidden, batch["opt_idx"]).to(torch.float32),
                "n_slots": batch["n_slots"],
                "labels": batch["labels"],
            })
    return cached


def batch_loss(head, batch: dict) -> torch.Tensor:
    from strands_decider.modeling import masked_log_softmax

    log_probs = masked_log_softmax(head(batch["pooled"], batch["options"]), batch["n_slots"])
    return torch.nn.functional.nll_loss(log_probs, batch["labels"])


def mean_loss(head, cached: list[dict]) -> float:
    head.eval()
    with torch.no_grad():
        losses = [float(batch_loss(head, b)) * b["labels"].numel() for b in cached]
    return sum(losses) / sum(b["labels"].numel() for b in cached)


def main() -> None:
    from safetensors.torch import save_file
    from strands_decider.data.collate import CollatorConfig, SystemOneCollator
    from strands_decider.data.format import load_examples
    from strands_decider.modeling import StrandsDeciderModel, build_head, checkpoint_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    source = checkpoint_dir(CHECKPOINT)
    model = StrandsDeciderModel.load(source).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = CollatorConfig(max_length=1024, num_slots=model.config.num_slots, head_type=model.config.head_type)
    started = time.perf_counter()
    train = cache_features(model, DataLoader(load_examples([str(DATA / "train.jsonl")]), batch_size=args.batch,
                                             shuffle=True, collate_fn=SystemOneCollator(model.tokenizer, config, train=True)))
    calib = cache_features(model, DataLoader(load_examples([str(DATA / "calib.jsonl")]), batch_size=args.batch,
                                             shuffle=False, collate_fn=SystemOneCollator(model.tokenizer, config, train=False)))
    print(json.dumps({"cached_batches": len(train) + len(calib), "seconds": round(time.perf_counter() - started)}), flush=True)

    head = build_head(model.config, StrandsDeciderModel.hidden_size(model.torso))
    head.load_state_dict({k: v.clone() for k, v in model.head.state_dict().items()})
    head.to(torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    best = mean_loss(head, calib)
    best_epoch = 0
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    print(json.dumps({"epoch": 0, "calib_loss": round(best, 4), "train_loss": round(mean_loss(head, train), 4)}), flush=True)
    for epoch in range(1, args.epochs + 1):
        head.train()
        random.shuffle(train)
        for batch in train:
            optimizer.zero_grad()
            batch_loss(head, batch).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        calib_loss = mean_loss(head, calib)
        if epoch % 5 == 0 or calib_loss < best:
            print(json.dumps({"epoch": epoch, "calib_loss": round(calib_loss, 4),
                              "train_loss": round(mean_loss(head, train), 4)}), flush=True)
        if calib_loss < best:
            best, best_epoch = calib_loss, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(source, out_dir, ignore=shutil.ignore_patterns("eval", ".gitattributes", "MANIFEST.sha256"))
    save_file({k: v.contiguous().to(torch.float32) for k, v in best_state.items()}, str(out_dir / "head.safetensors"))
    print(json.dumps({"saved": str(out_dir), "best_epoch": best_epoch, "best_calib_loss": round(best, 4)}), flush=True)


if __name__ == "__main__":
    main()
