# Voxtral Realtime over WebRTC

This setup sends incoming Opus audio directly to a per-session Voxtral ASR stream without waiting for silence. Transcription text is returned on a WebRTC data channel; no audio is generated.

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

Create a browser WebRTC offer with an Opus audio track and a data channel named `transcript`. POST `{"sdp":"...","type":"offer","model":"voxtral-asr"}` to `/v1/webrtc/sessions`, where `model` is the served speech-to-text graph name, set the returned answer as the remote description, and exchange ICE candidates through `/v1/webrtc/sessions/{session_id}/candidates`. The server uses UDP port 52000 for media. The data channel receives JSON messages of type `user_delta`, `user_final`, or `user_error`. Close the session with `/v1/webrtc/sessions/{session_id}/close` to flush the last audio chunk. Voxtral accepts 16 kHz mono audio internally; OVMS converts the incoming 48 kHz Opus frames.

### OpenAI Realtime compatible signaling

OVMS also accepts the OpenAI Realtime WebRTC flow used by OpenAI-compatible clients. POST the OpenAI unified `multipart/form-data` request to `/v1/realtime/calls` with an `sdp` field holding the SDP offer and a `session` field holding the session configuration JSON. The served speech-to-text graph is taken from `session.audio.input.transcription.model`, or from `session.model` when no transcription model is set, for example `{"type":"transcription","audio":{"input":{"transcription":{"model":"voxtral-asr"}}}}`. A plain `application/sdp` body is rejected with `412 Precondition Failed` because it does not name a model. The response body is the SDP answer (`application/sdp`) with all server ICE candidates embedded, so no candidate exchange is needed. Create the data channel with the label `oai-events`. After it opens, the server sends `session.created`. Stream audio and send `{"type":"input_audio_buffer.commit"}` to end an utterance. The server replies with `input_audio_buffer.committed`, then `conversation.item.input_audio_transcription.delta` events and a final `conversation.item.input_audio_transcription.completed` event that carries `item_id` and `transcript`. Audio sent after a commit starts a new item. The `Location` response header holds the call path (`/v1/realtime/calls/{call_id}`). End the call by closing the peer connection, or with `POST /v1/realtime/calls/{call_id}/hangup`. If OVMS runs with an API key, every WebRTC endpoint (`/v1/webrtc/sessions*` and `/v1/realtime/calls*`) requires it as `Authorization: Bearer <key>`. `session.update` events sent over the data channel are ignored: the model is fixed when the call is created.

The end-to-end tests in `tests/webrtc_e2e` exercise both signaling modes against a running server.

### Connecting from another machine (Docker)

Inside a container, OVMS only knows its bridge address (for example `172.17.0.8`), so the ICE candidates it advertises are unreachable for clients on other machines. Publish the media port 1:1 (`-p 52000:52000/udp`) and set `OVMS_WEBRTC_ANNOUNCED_IP` to the host's LAN or public IP address:

```bash
docker run ... -p 9112:9112 -p 52000:52000/udp -e OVMS_WEBRTC_ANNOUNCED_IP=192.168.1.50 ...
```

For every host ICE candidate, OVMS then also advertises a copy with the announced address and the same port. Both `/v1/webrtc/sessions` candidates and the `/v1/realtime/calls` SDP answer include it, and the original candidate is kept for clients on the same host. The value must be an IPv4 or IPv6 address of the same family as the container's candidates; invalid values are logged and ignored. Clients must be able to reach UDP port 52000 on that address; clients behind restrictive firewalls need a TURN relay.

WebRTC sessions use the pipeline of a served speech-to-text graph (`--task speech2text`, or a graph with an `S2tCalculator` node in the configuration file), so no second copy of the model is loaded and the device is the one the graph was deployed with. The exported `config.json` selects Voxtral Realtime streaming or utterance-based ASR such as Whisper; neither the repository name nor the serving alias controls backend selection. Every Voxtral session opens its own stream; Whisper transcriptions share the pipeline with `/v3/audio/transcriptions` and run one at a time. A request without a model is rejected with `412 Precondition Failed`, and a model name that is not a served speech-to-text graph with `404 Not Found`.

Utterance-based models such as Whisper buffer the incoming audio and transcribe it after 800 ms of silence, or immediately on `input_audio_buffer.commit` in the OpenAI Realtime flow. Each transcribed utterance becomes a separate item with its own delta and final events.

Use `/v1/webrtc/sessions` for SDP signaling, not `/v3/audio/transcriptions` or another V3 inference endpoint. The `model` field of a signaling request selects which served speech-to-text graph transcribes the session.