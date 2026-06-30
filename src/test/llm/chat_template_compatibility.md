# Chat Template Engine Compatibility

| Model | Family Detection | Auto Tool Parser Detection | Auto Reasoning Parser Detection | Minja Mode | Jinja Mode | Notes |
|-------|--------|-----------------|-------------|-------|-------|--------|
| [gpt-oss-20b](https://huggingface.co/OpenVINO/gpt-oss-20b-int4-ov) | gptoss | gptoss | gptoss | ✅ Tools work | ✅ Tools work | The chat template we have uploaded contain many workarounds, including one that relates to `string2obj`. Without workaround, it translates to `{"key":"val"}`. To keep supporting that, we dont run `str2obj` workaround. |
| [Qwen3.6-35B](https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov) | qwen3coder | qwen3coder | qwen3 | ✅ Tools work | ✅ Tools work  | XML `<parameter=...>` format; dry-run with jinja shows that its impossible to pass string so it auto converts to object; for minja it is patched inside minja so it works either way |
| [Gemma4](https://huggingface.co/OpenVINO/gemma-4-26b-a4b-it-int4-ov) | gemma4 | gemma4 | gemma4 | ✅ Tools work | ✅ Tools work  |  `key:<\|"\|>val<\|"\|>` format; dry-run shows we need to convert to object in both cases minja and jinja. Unit test contains unpatched chat template, even for `,response:` glitch which should not affect any patched template  |
| [Qwen3 Coder](https://huggingface.co/OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov) | qwen3coder | qwen3coder |  | ❌ Silent failure | ✅ Tools work | Minja lacks `from_json` filter → dumps raw JSON; Jinja has it. We pushed chat template long time ago which does not work with minja. **TODO: How to fix that?** |
| [Phi4-mini](https://huggingface.co/OpenVINO/Phi-4-mini-instruct-int4-ov) | phi4 | phi4 |  | ✅ Tools work | ✅ Tools work | It needs to be remembered to use chat template from our repo. It is not uploaded to HuggingFaces. Detection is based on existence of `functools[` keyword in comment of chat template. Caps detections showcase that it is better to not apply `str2obj` as it would produce pythonic `'key': 'val` text. Same as gpt-oss in this regard. |
| [Qwen3-4B](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | hermes3 | hermes3 | qwen3 | ✅ Tools work | ✅ Tools work | both: string and dict handled in chat template. Since both work `str2obj` is preferred. Chat template is taken from tokenizer_config.json. |
| [Mistral-7B-v0.3](https://huggingface.co/OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov) | mistral | mistral |  | ✅ Tools work | ✅ Tools work | for some reason it is detected as mistral, but it works> |
| [LFM2-24B](https://huggingface.co/OpenVINO/LFM2-24B-A2B-int4-ov) | *(undetected)* | |  | ⚠️ No tool support detected | ⚠️ No tool support detected | Template only injects tools into system prompt, no `tool_call` rendering |
| LFM2.5 | lfm2 | |  | ❌ Silent failure | ✅ Tools work | Minja can't handle this chat template for some reason, jinja works - autodetects str2obj needed. The chat template was taken from Pawel |
| [Qwen3-VL-8B](https://huggingface.co/OpenVINO/Qwen3-VL-8B-Instruct-int4-ov) | hermes3 | hermes3 |  | ✅ Tools work | ✅ Tools work | Template handles both string and object args natively (has `is string` check); probe detects `requiresObjectArguments=true` |
| [Qwen3-30B-A3B-2507](https://huggingface.co/OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov) | hermes3 | hermes3 |  | ✅ Tools work | ✅ Tools work | Same `<tool_call>` format as Qwen3; handles both string and object args natively; probe detects `requiresObjectArguments=true` |

## Key Differences

- **Qwen3 Coder** and **LFM2.5** only work with Jinja — minja silently fails (detected by probe, tool support disabled)
- All other tool-capable models work identically on both engines
- Phi4-mini and LFM2 lack tool_call rendering in the template itself, so neither engine detects tool support
