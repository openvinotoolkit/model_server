# Voxtral Realtime over WebRTC

This setup sends incoming Opus audio directly to a per-session Voxtral ASR stream. It does not wait for silence and does not load Qwen Omni or Whisper. Transcription text is returned on the WebRTC data channel named `transcript`; no audio is generated.

The model directory must contain `openvino_audio_encoder.xml`, `openvino_text_decoder.xml`, the corresponding `.bin` files, `config.json`, `preprocessor_config.json`, and `tekken.json`. OVMS must be built against an OpenVINO GenAI library that provides `ASRPipeline::create_stream()` and uses a compatible OpenVINO/Tokenizers runtime.

For the local builder image and Bazel build, run from the OVMS checkout:

```bash
docker run --rm --network host \
  -v "$PWD:/ovms" \
  -v "$HOME/.cache/ovms-bazel:/root/.cache/bazel" \
  -v "$HOME/models:/models" \
  -w /ovms --entrypoint /bin/bash \
  openvino/model_server-build:voxtral-local-20261006 \
  -c 'exec "$(bazel --output_user_root=/root/.cache/bazel/bazel info bazel-bin)/src/ovms" --rest_port 19112 --source_model OpenVINO/voxtral-realtime-2602 --model_repository_path /models --model_name voxtral-asr --task speech2text --target_device CPU'
```

Create a browser WebRTC offer with an Opus audio track and a data channel named `transcript`. POST `{"sdp":"...","type":"offer"}` to `/v1/webrtc/sessions`, set the returned answer as the remote description, and exchange ICE candidates through `/v1/webrtc/sessions/{session_id}/candidates`. The server uses UDP port 52000 for media. The data channel receives JSON messages of type `user_delta`, `user_final`, or `user_error`. Close the session with `/v1/webrtc/sessions/{session_id}/close` to flush the last audio chunk. Voxtral accepts 16 kHz mono audio internally; OVMS converts the incoming 48 kHz Opus frames.

WebRTC speech recognition is configured automatically by `--source_model` with `--task speech2text`. It uses the downloaded model under `--model_repository_path` and the device specified by `--target_device` (or the normal recommended device when omitted). The exported `config.json` selects Voxtral Realtime streaming or utterance-based ASR such as Whisper; neither the repository name nor the serving alias controls backend selection. No WebRTC speech-model path or device environment variables are needed or read. Omni environment settings are ignored in this speech-only mode.

Use `/v1/webrtc/sessions` for SDP signaling, not `/v3/audio/transcriptions` or another V3 inference endpoint. A signaling request does not need a `model` field: the speech model is selected at server startup.