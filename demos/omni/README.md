# Omni Pipeline Demo Clients

Example clients demonstrating Qwen3-Omni multimodal capabilities with OpenVINO Model Server.

**Requirements:** `pip install openai numpy sounddevice`

## Demo Clients

### Chat Completions API

#### Audio+Text → Text (unary)
```bash
python3 chat_completions_unary_at_t.py recording.wav
```

#### Text → Audio+Text (unary)
```bash
python3 chat_completions_unary_t_at.py --prompt "Hello, how are you?" --voice f04
```

#### Audio+Text+Image → Audio+Text (unary)
```bash
python3 chat_completions_unary_ati_at.py --audio recording.wav --image photo.jpg --voice m02
```

### Responses API

#### Audio+Text → Text (unary)
```bash
python3 responses_unary_at_t.py recording.wav
```

#### Text → Audio+Text (streaming)
```bash
python3 responses_stream_t_at.py --prompt "Tell me a short story" --voice f04
```

#### Multi-turn Voice Chat (streaming, with mic recording)
```bash
python3 responses_multiturn_voice_chat.py --voice f04
python3 responses_multiturn_voice_chat.py --voice m02 --no-stream --debug
```

## Naming Convention

`{api}_{mode}_{input}_{output}.py`

- **API:** `chat_completions` or `responses`
- **Mode:** `unary`, `stream`, or `multiturn`
- **Input:** `t` = text, `a` = audio, `i` = image (e.g., `at` = audio+text, `ati` = audio+text+image)
- **Output:** `t` = text, `at` = audio+text

## Available Voices

Model-dependent. For Qwen3-Omni Dense: `f04`, `f245`, `f37`, `m02`, `m31`, `m36`, `br_f019`
