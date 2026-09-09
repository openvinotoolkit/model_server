---
license: apache-2.0
tags:
  - openvino
  - test
  - dummy
  - mock
  - ci
  - gpt2
pipeline_tag: text-generation
library_name: openvino
---

# dummy-cyclic-gpt2-ov

**This is not a real language model.** It is a synthetic, hand-crafted GPT-2
whose weights were built to deterministically emit one fixed sentence followed
by a valid end-of-sequence token, regardless of the input prompt. It exists
purely as a fast, tiny, deterministic drop-in for testing OVMS / OpenVINO
GenAI integration code. Do not use it for anything that expects real language
understanding.

## What it outputs

Regardless of the input prompt, this model deterministically generates:

> OpenVINO is an open-source toolkit created by Intel to speed up and run AI models efficiently.

With normal generation settings (the default - `ignore_eos` unset/`False`),
the model emits this sentence once and then a valid end-of-sequence token, so
generation stops there, exactly like a normal, well-behaved language model. It
does **not** loop by default.

If the caller explicitly sets `ignore_eos=True` (e.g. in an OpenVINO GenAI
`GenerationConfig`) together with a large enough `max_new_tokens` budget,
decoding does not stop at the end-of-sequence token - the model treats it like
any other out-of-sequence token and deterministically predicts `Open` again,
so the sentence repeats forever. This is useful for stress-testing
streaming/very-long-generation code paths.

For any input that doesn't already contain a prefix of this exact sentence,
the very first generated token is always `Open`, and generation continues the
sentence above token-by-token until it emits the end-of-sequence token (or,
with `ignore_eos=True`, wraps back to `Open` and repeats). If the input
already ends with some prefix of the sentence, generation continues the cycle
from that point instead of restarting - the model doesn't "understand" the
phrase, it just always maps each token deterministically to the next one.

## How it behaves

- Architecture: `GPT2LMHeadModel`, 1 transformer layer, 1 attention head,
  hidden size 32 - intentionally minimal, this is not a "small but usable"
  model.
- All attention and MLP weights are zeroed, so every transformer block is a
  pure residual/identity: the hidden state at each position equals its own
  token embedding.
- The token embedding table maps each token of the sentence above (and the
  end-of-sequence token) to its own orthogonal direction; every other
  vocabulary token maps to one shared "default" direction.
- `lm_head` maps each direction to the *next* token in the sentence with a
  logit of ~100 (all other tokens ~0), so decoding is effectively always
  deterministic, even at nonzero temperature/top-k/top-p. The **last** token
  of the sentence deterministically predicts a valid end-of-sequence token,
  not a wraparound to the first token - normal generation therefore stops
  correctly after one pass.
- The "default" direction (used by any out-of-sequence token, including EOS
  if it's fed back in) deterministically predicts the first token of the
  sentence. This is what makes the sentence repeat forever if a caller sets
  `ignore_eos=True` - the pipeline keeps feeding the still-emitted EOS token
  back in as input instead of stopping.
- Position embeddings are zeroed too, so output depends only on the current
  token - not on prompt length, prompt content, or position.

## Supported OVMS pipeline types

`LM`, `LM_CB` (`text_generation` task, with or without continuous batching).

## Source / reproducibility

Built with the scripts in
[`src/test/dummy_genai`](https://github.com/openvinotoolkit/model_server/tree/main/src/test/dummy_genai)
in the OpenVINO Model Server repository (`create_model.py` + `export_to_ov.py`,
run via plain `./build.sh` - the sentence above is the tool's built-in default
sequence). See that directory for the exact weight-construction logic and how
to regenerate/verify this model.

## License

Apache 2.0. No pretrained weights are used anywhere in this model - all
weights are hand-crafted from scratch, so no upstream model license applies.
