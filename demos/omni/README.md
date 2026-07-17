# Omni Pipeline Demo

Example clients for Qwen3-Omni multimodal model served by OpenVINO Model Server.

**Requirements:** `pip install openai numpy sounddevice`

## Chat Completions API

```bash
# Text → text
python3 chat_completions.py --prompt "What is OpenVINO?"

# Text → text + audio
python3 chat_completions.py --prompt "Say 3 sentences about France. Dont use smileys, pure text." --audio-output --voice m36 --save output.wav

# Text + audio → text
python3 chat_completions.py --audio recording.wav --prompt "what is in the recording?"

# Audio + image + text → text + audio
python3 chat_completions.py --audio recording.wav --image ./../common/static/images/gorilla.jpeg --prompt "Detect which animal is in the image and tell me product name which is described in the audio." --audio-output --voice m02 --save output.wav
```

## Responses API

```bash
# Text → text
python3 responses.py --prompt "What is OpenVINO?"

# Text → text + audio (streaming with playback)
python3 responses.py --prompt "Say 3 sentences about France. Dont use smileys, pure text." --audio-output --voice m31 --save output.wav

# Audio → text (unary)
python3 responses.py --audio recording.wav --prompt "what is in the recording?"

# Image + text → text + audio, streaming
python3 responses.py --image ./../common/static/images/gorilla.jpeg --prompt "explain to me what is in the image" --audio-output --stream --voice m02 --save output.wav

# Audio + text → text + audio, streaming
python3 responses.py --audio recording.wav --prompt "what is in the recording?" --audio-output --stream --voice m02 --save output.wav

```

## Available Voices

`f04`, `f245`, `f37`, `m02`, `m31`, `m36`, `br_f019` (model-dependent)
