# Voxtral Realtime over WebRTC

This setup sends incoming Opus audio directly to a per-session Voxtral ASR stream. It does not wait for silence and does not load Qwen Omni or Whisper. Transcription text is returned on the WebRTC data channel named `transcript`; no audio is generated.

The model directory must contain `openvino_audio_encoder.xml`, `openvino_text_decoder.xml`, the corresponding `.bin` files, `config.json`, `preprocessor_config.json`, and `tekken.json`. OVMS must be built against an OpenVINO GenAI library that provides `ASRPipeline::create_stream()` and uses a compatible OpenVINO/Tokenizers runtime.

For the local builder image and Bazel build, run from the OVMS checkout:

```bash
docker run --rm --network host \
  -v "$PWD:/ovms" \
  -v "$HOME/.cache/ovms-bazel:/root/.cache/bazel" \
  -v "$HOME/models:/models:ro" \
  -e OVMS_WEBRTC_VOXTRAL_MODEL_PATH=/models/voxtral-realtime-2602 \
  -e OVMS_WEBRTC_STT_DEVICE=CPU \
  -w /ovms --entrypoint /bin/bash \
  openvino/model_server-build:voxtral-local-20261006 \
  -c 'exec "$(bazel --output_user_root=/root/.cache/bazel/bazel info bazel-bin)/src/ovms" --rest_port 19112 --config_path /ovms/demos/audio/voxtral_webrtc/config.json'
```

Create a browser WebRTC offer with an Opus audio track and a data channel named `transcript`. POST `{"sdp":"...","type":"offer"}` to `/v1/webrtc/sessions`, set the returned answer as the remote description, and exchange ICE candidates through `/v1/webrtc/sessions/{session_id}/candidates`. The server uses UDP port 52000 for media. The data channel receives JSON messages of type `user_delta`, `user_final`, or `user_error`. Close the session with `/v1/webrtc/sessions/{session_id}/close` to flush the last audio chunk. Voxtral accepts 16 kHz mono audio internally; OVMS converts the incoming 48 kHz Opus frames.

Do not set `OVMS_WEBRTC_OMNI_MODEL_PATH` or `OVMS_WEBRTC_STT_MODEL_PATH` in this mode. The existing Omni/Whisper WebRTC setup remains unchanged when `OVMS_WEBRTC_VOXTRAL_MODEL_PATH` is unset.