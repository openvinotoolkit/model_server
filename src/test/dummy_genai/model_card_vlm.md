---
license: apache-2.0
tags:
  - openvino
  - test
  - dummy
  - mock
  - ci
  - llava
  - vision-language
pipeline_tag: image-text-to-text
library_name: openvino
---

# dummy-cyclic-llava-ov

**This is not a real vision-language model.** It is a synthetic, hand-crafted
LLaVA whose text backbone was built to deterministically emit one fixed
sentence followed by a valid end-of-sequence token, regardless of the input
prompt *or the input image*. It exists purely as a fast, tiny, deterministic
drop-in for testing OVMS / OpenVINO GenAI integration code. Do not use it for
anything that expects real vision or language understanding.

## What it outputs

Regardless of the prompt and regardless of the image passed in, this model
deterministically generates the exact same sentence as its LLM sibling model
([`dummy-cyclic-gpt2-ov`](https://huggingface.co/mzeglars/dummy-cyclic-gpt2-ov)):

> OpenVINO is an open-source toolkit created by Intel to speed up and run AI models efficiently.

With normal generation settings (the default - `ignore_eos` unset/`False`),
the model emits this sentence once and then a valid end-of-sequence token, so
generation stops there. It only repeats forever if the caller explicitly sets
`ignore_eos=True` with a large enough `max_new_tokens` budget - see
`dummy-cyclic-gpt2-ov`'s model card for exactly why.

Image tokens are injected at early sequence positions by the vision
encoder/projector, but because the text backbone's attention is zeroed (see
below), they never influence the hidden state of the last text token - so the
generated output is completely image-agnostic. Text-side behavior (where
generation starts/resumes depending on the prompt, and whether it stops or
loops) is identical to `dummy-cyclic-gpt2-ov`.

## How it behaves

- Architecture: `LlavaForConditionalGeneration` - a LLaMA text backbone (1
  layer, 1 attention head, hidden size 32, RMSNorm) plus a small CLIP vision
  encoder and multimodal projector.
- The text backbone uses the identical deterministic mechanism as
  `dummy-cyclic-gpt2-ov`: token embeddings map to unique orthogonal
  directions, attention/MLP weights are zeroed (making every block a pure
  residual/identity), and `lm_head` deterministically predicts the next token
  of the fixed sentence above with a dominant (~100) logit - ending, on the
  last token, in a valid end-of-sequence prediction rather than a wraparound.
- Unlike GPT-2, LLaMA has no learned position-embedding table - it uses RoPE
  instead. Since attention is zeroed anyway, RoPE has no effect on the
  output, so generation is independent of position exactly like the LLM
  sibling model.
- The CLIP vision encoder and multimodal projector keep their random,
  untrained default weights. They still run and produce valid image
  embeddings, but those embeddings never affect the generated text.

## Supported OVMS pipeline types

`VLM`, `VLM_CB` (`image-text-to-text` task, with or without continuous
batching).

## Source / reproducibility

Built with the scripts in
[`src/test/dummy_genai`](https://github.com/openvinotoolkit/model_server/tree/main/src/test/dummy_genai)
in the OpenVINO Model Server repository (`create_vlm_model.py` +
`optimum-cli`/`convert_tokenizer`, run via plain `./build_vlm.sh` - the
sentence above is the tool's built-in default sequence). See that directory
for the exact weight-construction logic and how to regenerate/verify this
model.

## License

Apache 2.0. No pretrained weights are used in the text backbone - those
weights are hand-crafted from scratch. The CLIP vision encoder and
multimodal projector keep their randomly-initialized (untrained) default
weights, so no upstream pretrained-model license applies to those either.
