#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a dummy GPT-2-based causal LM that deterministically generates one fixed
sequence and then a valid end-of-sequence token:
    "OpenVINO is an open-source toolkit created by Intel to speed up and run AI models efficiently."

With normal generation settings the model stops right after that end-of-sequence
token, like any well-behaved LM. Passing `ignore_eos=True` (openvino_genai
GenerationConfig) makes the pipeline feed EOS back in as if it were any other
non-sequence token, which this model maps to the first token of the sequence -
so the sentence repeats forever instead of stopping.

Weight design (all using standard GPT-2 architecture so optimum/OV export just works):

  wte  : sequence token t_i  → standard basis vector e_i  (1 at dim i, 0 elsewhere)
         any other token (incl. EOS) → e_{DEFAULT_DIM}  (same "default" vector for all)

  wpe  : all zeros  (no positional interference)

  attn/MLP weights : all zeros  (each transformer block becomes a pure residual = identity)

  ln_f : default weight=1, bias=0

  lm_head:  for each i < N-1, W[t_{i+1}, i] = SCALE      →  logit ≈ 100 after LayerNorm
            for i = N-1 (last token), W[EOS, i] = SCALE   →  correctly ends the sequence
            W[t_0, DEFAULT_DIM] = SCALE                    →  non-sequence tokens (incl. EOS
            fed back in under ignore_eos=True) predict t_0

  SCALE is chosen so that ln_scale * SCALE ≈ 100, where ln_scale = LayerNorm(e_i)[i].
  This makes the correct next-token logit (~100) dominate over all others (~0 or negative).
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

TARGET_SEQUENCE = "OpenVINO is an open-source toolkit created by Intel to speed up and run AI models efficiently."
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_model")

# Handles system/user/assistant turns, multi-part content blocks, and reasoning_content.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "System: {{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}"
    "User: "
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "\n"
    "{% elif message['role'] == 'assistant' %}"
    "{% if message.get('reasoning_content') %}"
    "Thinking: {{ message['reasoning_content'] }}\n"
    "{% endif %}"
    "Assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)


def layernorm_scale_for_basis_vector(hidden_dim: int, eps: float = 1e-5) -> float:
    """Return the value that LayerNorm(e_i)[i] equals, for a standard basis vector e_i in R^H."""
    H = hidden_dim
    mean = 1.0 / H
    # Population variance: E[(x - mean)^2] over H elements, one of which is 1, rest are 0
    var = ((H - 1) * mean ** 2 + (1.0 - mean) ** 2) / H
    return (1.0 - mean) / math.sqrt(var + eps)


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dummy cyclic LLM (GPT-2 based)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the HuggingFace model")
    parser.add_argument("--sequence", default=TARGET_SEQUENCE,
                        help="Cyclic text sequence the model will generate endlessly")
    args = parser.parse_args()

    print(f"Target sequence : {args.sequence!r}")
    print(f"Output directory: {args.output_dir}")

    # ── Tokenize ────────────────────────────────────────────────────────────────
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    token_ids: list[int] = tokenizer.encode(args.sequence, add_special_tokens=False)
    token_texts: list[str] = [tokenizer.decode([t]) for t in token_ids]
    N = len(token_ids)

    print(f"Sequence has {N} tokens:")
    for i, (tid, txt) in enumerate(zip(token_ids, token_texts)):
        print(f"  [{i:2d}] id={tid:6d}  {txt!r}")

    # Guard: duplicate token IDs break the bijective embedding↔position mapping
    if len(set(token_ids)) != N:
        dupes = [tid for tid in token_ids if token_ids.count(tid) > 1]
        print(f"\nERROR: Duplicate token IDs in sequence: {dupes}", file=sys.stderr)
        print("Choose a sequence where every GPT-2 token appears at most once.", file=sys.stderr)
        sys.exit(1)

    # ── Dimensions ──────────────────────────────────────────────────────────────
    # Need H >= N+1: N dims for sequence positions, 1 dim for the "default" vector.
    # Round up to the next power of two (no artificial floor: wte/lm_head are O(vocab×H),
    # so keeping H small is the biggest size lever).
    H = next_power_of_two(N + 2)
    DEFAULT_DIM = N            # dimension index for "any non-sequence token"
    VOCAB_SIZE = tokenizer.vocab_size   # 50257 for GPT-2

    ln_scale = layernorm_scale_for_basis_vector(H)
    # Raw lm_head weight so that logit = ln_scale * LM_WEIGHT ≈ 100
    LM_WEIGHT = 100.0 / ln_scale

    print(f"\nHidden dim     : {H}")
    print(f"Default dim    : {DEFAULT_DIM}")
    print(f"LayerNorm scale: {ln_scale:.4f}  →  lm_head weight: {LM_WEIGHT:.4f}")

    # ── GPT-2 config ─────────────────────────────────────────────────────────────
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=131072,  # 128k - lets tests exercise huge-context behavior
        n_embd=H,
        n_layer=1,       # zeroed-out blocks are identity; one is enough
        n_head=1,        # head_dim = H; satisfies GPU PA kernel requirement head_size >= 16
        n_inner=4 * H,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id or 50256,
        eos_token_id=tokenizer.eos_token_id or 50256,
        # MUST be False so wte and lm_head carry independent weights
        tie_word_embeddings=False,
    )

    print("\nCreating model...")
    model = GPT2LMHeadModel(config)
    token_id_set = set(token_ids)

    with torch.no_grad():

        # ── Token embeddings ────────────────────────────────────────────────────
        # Sequence token t_i  → e_i  (1 at position i, 0 everywhere else)
        # All other tokens    → e_{DEFAULT_DIM}
        wte = torch.zeros(VOCAB_SIZE, H)
        for i, tok_id in enumerate(token_ids):
            wte[tok_id, i] = 1.0
        for tok_id in range(VOCAB_SIZE):
            if tok_id not in token_id_set:
                wte[tok_id, DEFAULT_DIM] = 1.0
        model.transformer.wte.weight.copy_(wte)

        # ── Positional embeddings ───────────────────────────────────────────────
        model.transformer.wpe.weight.zero_()

        # ── Transformer blocks → identity ───────────────────────────────────────
        # Zero attention projections and MLP weights; layer norms stay at defaults.
        # With zero attn/MLP output, each block reduces to:  output = input (residual).
        for block in model.transformer.h:
            for p in block.attn.parameters():
                p.zero_()
            for p in block.mlp.parameters():
                p.zero_()

        # ── lm_head ─────────────────────────────────────────────────────────────
        # Transition rule: after LayerNorm, hidden state for t_i has large value at
        # dimension i and small (~-0.13) values at all other dims.
        # Setting W[t_{i+1}, i] = LM_WEIGHT makes logit[t_{i+1}] ≈ 100 >> others ≈ 0.
        lm_head_w = torch.zeros(VOCAB_SIZE, H)
        for i in range(N):
            # Last token always predicts EOS; use ignore_eos=True in sampling for looping
            next_tok = tokenizer.eos_token_id if i == N - 1 else token_ids[i + 1]
            lm_head_w[next_tok, i] = LM_WEIGHT
        # Non-sequence tokens use the default vector (dim DEFAULT_DIM) → predict t_0
        lm_head_w[token_ids[0], DEFAULT_DIM] = LM_WEIGHT

        # Untie and replace lm_head weights (config.tie_word_embeddings=False ensures no re-tying)
        model.lm_head.weight = nn.Parameter(lm_head_w)

        assert model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr(), \
            "lm_head and wte are still sharing storage – check tie_word_embeddings"

    # ── Sanity check ─────────────────────────────────────────────────────────────
    print("\nSanity check (greedy single-token forward passes):")
    model.eval()
    failures = 0
    with torch.no_grad():
        for i, tok_id in enumerate(token_ids):
            inp = torch.tensor([[tok_id]])
            predicted = model(inp).logits[0, -1].argmax().item()
            expected  = tokenizer.eos_token_id if i == N - 1 else token_ids[i + 1]
            ok = (predicted == expected)
            failures += 0 if ok else 1
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {token_texts[i]!r:20s} → predicted {tokenizer.decode([predicted])!r:20s}"
                  f" (expected {tokenizer.decode([expected])!r})")

        # Non-sequence token (EOS) should predict t_0
        eos = tokenizer.eos_token_id
        inp = torch.tensor([[eos]])
        predicted = model(inp).logits[0, -1].argmax().item()
        expected  = token_ids[0]
        ok = (predicted == expected)
        failures += 0 if ok else 1
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] <EOS>                 → predicted {tokenizer.decode([predicted])!r:20s}"
              f" (expected {tokenizer.decode([expected])!r})")

    if failures:
        print(f"\nWARNING: {failures} sanity-check failure(s). Inspect the weight setup.", file=sys.stderr)
    else:
        print("\nAll transitions correct!")

    # ── Save ─────────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving model to {args.output_dir!r} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "target_sequence": args.sequence,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "hidden_dim": H,
        "default_dim": DEFAULT_DIM,
        "layernorm_scale": ln_scale,
    }
    with open(os.path.join(args.output_dir, "dummy_llm_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
