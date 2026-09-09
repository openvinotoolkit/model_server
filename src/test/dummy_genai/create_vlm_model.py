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
Create a dummy LLaVA-style VLM that always generates the cyclic sequence:
    "For the night is dark and full of terrors..."
regardless of the image and text prompt.

Architecture:
  LlavaForConditionalGeneration
    vision_tower      : tiny CLIPVisionModel (random weights, non-NaN)
    multi_modal_projector : random weights (maps vision → text hidden space)
    language_model    : LlamaForCausalLM with the same cyclic weight trick

Why images do not disturb the cyclic generation:
  The language model has all attention and MLP weights zeroed, so every
  position's hidden state = its own input embedding, independent of all other
  positions.  Image patch embeddings (injected by the projector) appear at
  early positions; generation only reads the logit of the *last* text token,
  which still follows the cyclic rule.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    GPT2TokenizerFast,
)

TARGET_SEQUENCE = "For the night is dark and full of terrors..."
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Handles system/user/assistant turns, image/image_url content blocks, and reasoning_content.
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
    "{% if item['type'] in ['image', 'image_url'] %}"
    "<image>\n"
    "{% elif item['type'] == 'text' %}"
    "{{ item['text'] }}"
    "{% endif %}"
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
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "vlm_hf_model")


def rms_norm_scale_for_basis_vector(hidden_dim: int, eps: float = 1e-5) -> float:
    """Value that RMSNorm(e_i)[i] equals for a standard basis vector e_i in R^H.

    RMS(e_i) = sqrt(1/H)  →  scale = 1 / sqrt(1/H + eps) ≈ sqrt(H).
    Off-diagonal components are exactly 0, which is cleaner than LayerNorm.
    """
    return 1.0 / math.sqrt(1.0 / hidden_dim + eps)


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dummy cyclic VLM (LLaVA + GPT-2)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence", default=TARGET_SEQUENCE)
    args = parser.parse_args()

    print(f"Target sequence : {args.sequence!r}")
    print(f"Output directory: {args.output_dir}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print("\nLoading GPT-2 tokenizer, adding <image> token...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # <image> is inserted into the text prompt where the image pixels will be injected.
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    IMAGE_TOKEN_ID: int = tokenizer.convert_tokens_to_ids("<image>")
    tokenizer.pad_token = tokenizer.eos_token

    token_ids: list[int] = tokenizer.encode(args.sequence, add_special_tokens=False)
    token_texts: list[str] = [tokenizer.decode([t]) for t in token_ids]
    N = len(token_ids)
    print(f"Sequence has {N} tokens:  {token_texts}")
    print(f"<image> token id: {IMAGE_TOKEN_ID}")

    if len(set(token_ids)) != N:
        print("ERROR: duplicate token IDs in the target sequence.", file=sys.stderr)
        sys.exit(1)

    # ── Dimensions ──────────────────────────────────────────────────────────
    H = next_power_of_two(N + 2)            # LM hidden dim
    DEFAULT_DIM = N                          # embedding slot for all non-sequence tokens
    VOCAB_SIZE = len(tokenizer)              # 50257 (gpt2) + 1 (<image>) = 50258
    # RMSNorm(e_i)[i] ≈ sqrt(H); off-diagonal = 0 (exact), so LM_WEIGHT = 100/sqrt(H)
    rms_scale = rms_norm_scale_for_basis_vector(H)
    LM_WEIGHT = 100.0 / rms_scale

    print(f"LM hidden dim  : {H}")
    print(f"Vocab size     : {VOCAB_SIZE}")

    # ── Vision config: tiny CLIP ─────────────────────────────────────────────
    # patch_size=32 over 224×224 → 7×7 = 49 patches per image.
    # Random (default) weights are fine: vision features only fill positions that
    # precede the last text token, which is never sampled during generation.
    vision_config = CLIPVisionConfig(
        hidden_size=32,
        image_size=224,
        intermediate_size=128,
        num_attention_heads=2,
        num_channels=3,
        num_hidden_layers=1,
        patch_size=32,
    )

    # ── Text (LM) config: LlamaForCausalLM ──────────────────────────────────
    # LLaVA-1.5 uses LLaMA as its language model; this is the well-tested export
    # path in optimum-intel.  Using GPT-2 here risks mismatched attribute paths
    # in the VLM export patching code (which navigates model.model.layers etc.).
    text_config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=H,
        intermediate_size=4 * H,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,  # head_dim = H; satisfies GPU PA kernel head_size >= 16
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        bos_token_id=tokenizer.bos_token_id or 50256,
        eos_token_id=tokenizer.eos_token_id or 50256,
        tie_word_embeddings=False,
    )

    # ── LLaVA config ─────────────────────────────────────────────────────────
    llava_config = LlavaConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=IMAGE_TOKEN_ID,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,   # standard LLaVA-1.5 setting
        pad_token_id=tokenizer.pad_token_id,
    )

    # ── Create model ─────────────────────────────────────────────────────────
    print("\nCreating LLaVA model...")
    model = LlavaForConditionalGeneration(llava_config)

    # Locate the LLaMA backbone and lm_head regardless of transformers version.
    # Old (<~4.47): model.language_model  is a LlamaForCausalLM
    #               backbone = model.language_model.model   (LlamaModel)
    # New (≥~4.47): model.model.language_model  is a LlamaModel directly
    #               lm_head lives at model.lm_head
    if hasattr(model, 'language_model'):
        backbone = model.language_model.model
        lm_head  = model.language_model.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        backbone = model.model.language_model
        lm_head  = model.lm_head
    else:
        children = [n for n, _ in model.named_children()]
        raise RuntimeError(
            f"Cannot locate LLaMA backbone in this LlavaForConditionalGeneration. "
            f"Top-level children: {children}. Inspect print(model) and update attribute paths."
        )

    token_id_set = set(token_ids)
    with torch.no_grad():

        # Token embeddings: sequence token t_i → e_i; all others → e_{DEFAULT_DIM}
        wte = torch.zeros(VOCAB_SIZE, H)
        for i, tok_id in enumerate(token_ids):
            wte[tok_id, i] = 1.0
        for tok_id in range(VOCAB_SIZE):
            if tok_id not in token_id_set:
                wte[tok_id, DEFAULT_DIM] = 1.0
        backbone.embed_tokens.weight.copy_(wte)

        # LLaMA uses RoPE (no positional embedding table to zero out).
        # Transformer blocks → identity: zero all self-attn and MLP projections.
        # RMSNorms (input_layernorm, post_attention_layernorm, norm) keep defaults.
        for layer in backbone.layers:
            for p in layer.self_attn.parameters():
                p.zero_()
            for p in layer.mlp.parameters():
                p.zero_()

        # lm_head: W[t_{i+1}, i] = LM_WEIGHT
        # RMSNorm(e_i)[i] = rms_scale ≈ sqrt(H); off-diagonal = 0 (exact).
        # → logit[t_{i+1}] = rms_scale * LM_WEIGHT = 100; all others = 0.
        lm_head_w = torch.zeros(VOCAB_SIZE, H)
        for i in range(N):
            # Last token always predicts EOS; use ignore_eos=True in sampling for looping
            next_tok = tokenizer.eos_token_id if i == N - 1 else token_ids[i + 1]
            lm_head_w[next_tok, i] = LM_WEIGHT
        lm_head_w[token_ids[0], DEFAULT_DIM] = LM_WEIGHT
        lm_head.weight = nn.Parameter(lm_head_w)

        assert lm_head.weight.data_ptr() != backbone.embed_tokens.weight.data_ptr()

    # ── Sanity check (text tokens only) ─────────────────────────────────────
    # Manual forward: embed → final_norm → lm_head (attn/MLP are zero so blocks = identity)
    print("\nSanity check (single-token forward passes on text tokens):")
    model.eval()
    failures = 0
    with torch.no_grad():
        for i, tok_id in enumerate(token_ids):
            inp = torch.tensor([tok_id])
            h = backbone.embed_tokens(inp)   # [1, H]
            h = backbone.norm(h)             # [1, H]
            predicted = lm_head(h)[0].argmax().item()
            expected  = tokenizer.eos_token_id if i == N - 1 else token_ids[i + 1]
            ok = predicted == expected
            failures += 0 if ok else 1
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {token_texts[i]!r:20s} → {tokenizer.decode([predicted])!r:20s}"
                  f" (expected {tokenizer.decode([expected])!r})")

    if failures:
        print(f"\nWARNING: {failures} failure(s).", file=sys.stderr)
    else:
        print("\nAll text transitions correct!")

    # ── Image processor + processor ──────────────────────────────────────────
    image_processor = CLIPImageProcessor(
        image_size=224,
        crop_size={"height": 224, "width": 224},
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving model to {args.output_dir!r} ...")
    model.save_pretrained(args.output_dir)
    processor.tokenizer.chat_template = CHAT_TEMPLATE
    processor.save_pretrained(args.output_dir)

    metadata = {
        "target_sequence": args.sequence,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "hidden_dim": H,
        "image_token_id": IMAGE_TOKEN_ID,
    }
    with open(os.path.join(args.output_dir, "dummy_vlm_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
