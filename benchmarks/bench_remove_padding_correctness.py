"""Correctness benchmark for remove_padding (sequence packing).

Mimics on-policy PPO: runs two identical forward passes on the same data and
verifies that exp(lp_run2 - lp_run1) == 1.0 for every token.  If packing
introduces cross-sequence attention leakage, logprobs will differ between
runs (batch composition changes the result), causing ratios != 1.

Usage:
    # Start server with remove_padding + mbs >= 2:
    python -m hosted_tinker.api \
        --base-model Qwen/Qwen3-0.6B \
        --backend fsdp2 \
        --backend-config '{"n_train_gpus": 4, "micro_batch_size": 2, "remove_padding": true}'

    # Run correctness check:
    python benchmarks/bench_remove_padding_correctness.py --url http://localhost:8000

    # Compare against mbs=1 baseline (optional):
    python benchmarks/bench_remove_padding_correctness.py --url http://localhost:8000 --compare-mbs1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
import tinker


def make_data(
    n_examples: int, min_len: int, max_len: int, seed: int
) -> tuple[list[tinker.Datum], list[int]]:
    """Generate random sequences with varied lengths."""
    rng = np.random.RandomState(seed)
    lengths = rng.randint(min_len, max_len + 1, size=n_examples)
    data = []
    for seq_len in lengths:
        seq_len = int(seq_len)
        tokens = rng.randint(100, 150000, size=seq_len).tolist()
        target_tokens = tokens[1:] + [0]
        # Weight only the second half (mimics prompt+completion masking)
        train_start = seq_len // 2
        weights = [0.0] * train_start + [1.0] * (seq_len - train_start)
        zeros = [0.0] * seq_len
        data.append(
            tinker.Datum(
                model_input=tinker.ModelInput(
                    chunks=[tinker.EncodedTextChunk(tokens=tokens)]
                ),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(data=target_tokens, dtype="int64"),
                    "weights": tinker.TensorData(data=weights, dtype="float32"),
                    "logprobs": tinker.TensorData(data=zeros, dtype="float32"),
                    "advantages": tinker.TensorData(data=[1.0] * seq_len, dtype="float32"),
                },
            )
        )
    return data, lengths.tolist()


def extract_logprobs(result) -> list[list[float]]:
    """Extract per-example logprob lists from a forward/forward_backward result."""
    return [out["logprobs"].data for out in result.loss_fn_outputs]


def run_forward(tc, data: list[tinker.Datum], timeout: int = 600) -> list[list[float]]:
    """Run forward pass and return logprobs per example."""
    result = tc.forward(data, loss_fn="cross_entropy").result(timeout=timeout)
    return extract_logprobs(result)


def check_ratios(
    lp_a: list[list[float]],
    lp_b: list[list[float]],
    weights: list[list[float]],
    label: str,
    atol: float = 1e-4,
) -> bool:
    """Check exp(lp_b - lp_a) == 1.0 on weighted tokens. Return True if passed."""
    all_ratios = []
    max_dev = 0.0
    for i, (a, b, w) in enumerate(zip(lp_a, lp_b, weights)):
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        w_arr = np.array(w, dtype=np.float64)
        ratios = np.exp(b_arr - a_arr)
        # Only check tokens with nonzero weight
        mask = w_arr > 0
        if mask.sum() == 0:
            continue
        masked_ratios = ratios[mask]
        dev = np.max(np.abs(masked_ratios - 1.0))
        if dev > max_dev:
            max_dev = dev
        all_ratios.extend(masked_ratios.tolist())

    all_ratios = np.array(all_ratios)
    mean_ratio = all_ratios.mean()
    std_ratio = all_ratios.std()
    passed = max_dev < atol

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    print(f"    n_tokens={len(all_ratios)}  mean_ratio={mean_ratio:.8f}  "
          f"std={std_ratio:.2e}  max_dev={max_dev:.2e}  atol={atol:.0e}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Remove-padding correctness benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--min-len", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Number of repeat forward passes to compare")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for ratio deviation from 1.0")
    parser.add_argument("--compare-mbs1", action="store_true",
                        help="Also run with mbs=1 via admin endpoint and compare")
    args = parser.parse_args()

    service = tinker.ServiceClient(base_url=args.url)
    model_name = args.model or service.base_model
    tc = service.create_lora_training_client(base_model=model_name, rank=args.lora_rank)

    data, lengths = make_data(args.n_examples, args.min_len, args.max_len, args.seed)
    total_tokens = sum(lengths)
    weights = [d.loss_fn_inputs["weights"].data for d in data]

    print(f"Config: {args.n_examples} seqs, lengths {args.min_len}-{args.max_len}, "
          f"total {total_tokens} tokens, seed={args.seed}")
    print()

    # --- Test 1: Self-consistency (two identical forward passes) ---
    print("Test 1: Self-consistency (same batch, repeated forward passes)")
    print("  If packing is correct, logprobs should be identical across runs.")
    all_lps = []
    for i in range(args.n_repeats):
        t0 = time.time()
        lps = run_forward(tc, data)
        elapsed = time.time() - t0
        all_lps.append(lps)
        print(f"  forward {i+1}/{args.n_repeats}: {elapsed:.2f}s")

    all_passed = True
    for i in range(1, args.n_repeats):
        ok = check_ratios(all_lps[0], all_lps[i], weights,
                          f"run1 vs run{i+1}", atol=args.atol)
        all_passed = all_passed and ok
    print()

    # --- Test 2: Different batch compositions ---
    print("Test 2: Different batch composition (subset vs full batch)")
    print("  If packing is correct, logprobs should NOT depend on what other")
    print("  sequences are in the batch.")

    # Run full batch
    lp_full = run_forward(tc, data)

    # Run first half and second half separately
    mid = args.n_examples // 2
    lp_first_half = run_forward(tc, data[:mid])
    lp_second_half = run_forward(tc, data[mid:])
    lp_separate = lp_first_half + lp_second_half

    ok = check_ratios(lp_full, lp_separate, weights,
                      "full_batch vs split_batches", atol=args.atol)
    all_passed = all_passed and ok
    print()

    # --- Test 3: Reversed order ---
    print("Test 3: Reversed batch order")
    print("  Logprobs for each sequence should be the same regardless of order.")
    lp_reversed = run_forward(tc, list(reversed(data)))
    lp_reversed_reorder = list(reversed(lp_reversed))
    weights_orig = weights

    ok = check_ratios(lp_full, lp_reversed_reorder, weights_orig,
                      "original_order vs reversed_order", atol=args.atol)
    all_passed = all_passed and ok
    print()

    # --- Test 4 (optional): Compare against mbs=1 baseline ---
    if args.compare_mbs1:
        import requests as req
        print("Test 4: Compare mbs=current vs mbs=1 (admin endpoint)")
        # Get current mbs, set to 1, run, restore
        resp = req.post(f"{args.url}/admin/set_micro_batch_size",
                        json={"n": 1}, timeout=10)
        if resp.status_code != 200:
            print(f"  SKIP: admin endpoint returned {resp.status_code}")
        else:
            lp_mbs1 = run_forward(tc, data)
            # Restore original mbs (we don't know it, but tests are done)
            ok = check_ratios(lp_full, lp_mbs1, weights,
                              "mbs=current vs mbs=1", atol=args.atol)
            all_passed = all_passed and ok
        print()

    # --- Summary ---
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
