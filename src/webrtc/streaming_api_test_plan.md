# WebRTC Streaming ASR API Test Plan

## Purpose and compatibility boundary

This plan tracks tests for OVMS's WebRTC speech-to-text endpoint at
`/v1/realtime/calls`. The target is the client-visible OpenAI Realtime
transcription contract, not merely the behavior of the current implementation.
Tests should assert documented event types, required fields, and item
association; they should avoid depending on internal implementation details or
incidental event ordering.

OpenAI's [Realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription)
is the baseline for transcription event behavior: audio turns are committed,
clients receive transcript deltas and a completion event, and `item_id` is used
to associate events with their audio item. The guide also documents
`session.update` for transcription configuration. OVMS serves a configured
speech-to-text graph rather than loading arbitrary model names requested by a
client, so tests must distinguish supported configuration updates from
unsupported model changes and require a protocol-shaped response for rejected
requests.

**Status meanings**

- **Implemented**: a test exists in the repository. This does not mean it was
  run in every environment.
- **Unimplemented**: a proposed contract test that still needs to be written.
- **HIGH / MID / LOW**: relative implementation priority for unimplemented
  tests.

## Existing tests

| Status | Test coverage | Location and notes |
|---|---|---|
| Implemented | Adapter rejects malformed audio chunks; drains queued audio; calls delta/final/error callbacks; handles repeated `finish()`/`close()`; ignores audio after finish; enforces queue capacity; creates a separate inference stream per audio stream. | `src/webrtc/voxtral_audio_adapter_test.cpp`. Uses a fake inference stream; validates adapter behavior, not OpenAI protocol compatibility or actual model quality. |
| Implemented | Empty signaling offer rejection; complete SDP answer and ICE candidates; announced-IP candidate behavior; speech model selection from metadata and invalid metadata rejection. | `src/webrtc/webrtc_session_controller_test.cpp`. C++ controller tests. |
| Implemented, model-dependent | Voxtral streaming without waiting for silence; Voxtral commit/multiple-item transcript association; Whisper silence and explicit-commit transcription; adapter emits text before input ends. | `src/webrtc/webrtc_session_controller_test.cpp`. Requires model/audio environment variables; these tests are skipped when those are absent. The multi-item Voxtral test is real-model coverage, not deterministic protocol-only coverage. |
| Implemented, execution not yet verified | OVMS transcription through multipart and plain-SDP signaling; SDP response content type/candidates; transcript keyword; transcription event schema and item association; call `Location` and hangup; invalid signaling; authentication; malformed data-channel event; empty commit. Contract-first tests also cover live delta before commit, multiple committed turns, transcription-context update acknowledgment, rejection of an update to an unserved model while preserving the active call, and protocol-shaped errors for invalid/unknown client events. | `tests/webrtc_e2e/test_webrtc_transcription.py`. Requires a running OVMS instance; pytest/e2e dependencies were unavailable in the prior validation environment. The model-update, context-update, and unknown-event tests are expected to expose current compatibility gaps until the endpoint implements the tested contract. The live-delta test is gated on a Voxtral Realtime model. |
| Implemented, external-service dependent | A transcription request against OpenAI's endpoint. | `tests/webrtc_e2e/test_webrtc_transcription.py`. Requires `OPENAI_API_KEY` and network/service availability; skipped otherwise. |

## Proposed tests

### HIGH priority

| Status | Test | Contract to assert |
|---|---|---|
| Implemented; e2e execution pending | Live delta before commit | Stream audio from a configured Voxtral Realtime model and require a non-empty delta before the audio source ends and before sending `input_audio_buffer.commit`. This is gated on `OVMS_REALTIME_MODEL`; it is intentionally not asserted for utterance-based models. |
| Implemented; e2e execution pending | Committed-turn event schema and association | Send audio, then commit it. Validate JSON event envelopes and required fields for committed, delta, and completed events. Verify deltas and completion reference committed item IDs, `content_index` is valid, and transcript content is a string. |
| Implemented; e2e execution pending | Multiple committed turns | Commit once while the audio fixture is still streaming and again at end of input. Require distinct item IDs and one completion per committed item; verify per-event item associations without assuming cross-turn completion ordering. |
| Implemented; expected to expose a compatibility gap | `session.update` for transcription context | Send a valid transcription prompt update and require a `session.updated` acknowledgment that reflects the new prompt. Verify actual inference influence separately if a stable model-backed fixture is established. The current server ignores updates. |
| Implemented; expected to expose a compatibility gap | Unsupported session/model update | Request a model not served by this call. Require a structured protocol error, no update acknowledgment, and successful transcription on the already selected model afterward. The current server silently ignores updates. |
| Implemented; expected to expose a compatibility gap | Protocol-shaped errors for invalid client events | Cover malformed JSON, missing/non-string `type`, and an unsupported event type. Assert the documented error event envelope and required fields, not OVMS-specific English message text. The current server silently ignores unknown event types. |

### MID priority

| Status | Test | Contract to assert |
|---|---|---|
| Unimplemented | Buffer clear behavior | If OVMS supports `input_audio_buffer.clear` for its WebRTC media-track flow, send audio then clear and verify the cleared audio is not transcribed on a subsequent commit. Otherwise require a structured unsupported-event response. Confirm applicability against the exact OpenAI WebRTC transcription contract before setting the expected result. |
| Unimplemented | Commit edge cases | Test commit before any audio, repeated commit, and commit while a prior turn is still completing. Assert protocol-consistent errors or item lifecycle behavior and ensure items are not merged or duplicated. Keep behavior aligned with the documented endpoint contract. |
| Unimplemented | Session-created event contract | Validate the initial server event's required envelope and session fields (including event ID and session object/type where applicable), rather than merely checking that the string `session.created` appeared. |
| Unimplemented | Signaling/session validation | Cover missing/invalid multipart fields, malformed session JSON, missing model selection, and unsupported session type. Assert documented HTTP status and JSON error shape, and that failed requests do not advertise a successful call `Location`. |
| Unimplemented | Call teardown and resource lifecycle | Close the peer connection and separately invoke the hangup endpoint. Verify the call is cleaned up, no further transcription events are emitted, and repeated/late hangup follows a defined response. |
| Unimplemented | Delta/completion content consistency | For each completed item, compare accumulated deltas for that same item with the final transcript according to the documented semantics (for example, allow final correction if the protocol permits it). Do not assume all models emit a delta for every non-empty transcript unless the contract guarantees it. |

### LOW priority

| Status | Test | Contract to assert |
|---|---|---|
| Unimplemented | Differential OpenAI reference run | Run a shared client scenario against OpenAI and OVMS and compare protocol-level invariants (event schema, item association, error envelope), not identical transcript text or model timing. Keep credentials opt-in and avoid making routine CI depend on a paid external service. |
| Unimplemented | Broad audio interoperability matrix | Exercise multiple supported input rates/channels, long sessions, silence, and representative codec/audio conditions. Assert successful protocol behavior and transcription quality using tolerances appropriate to different ASR models. |
| Unimplemented | Remote ICE/network topology | Exercise announced IP, remote-client connectivity, and any supported relay/TURN setup in a controlled network. Existing unit tests check candidate rewriting but do not establish end-to-end remote reachability. |
| Unimplemented | Authentication matrix | Expand current API-key checks to malformed bearer values, valid credentials, and hangup authorization for an existing call; verify status and error response consistency. |

## Test design guidance

- Prefer protocol assertions against decoded JSON objects over substring matching
  serialized messages.
- Assert required fields and per-item relationships; avoid exact transcript
  strings except for stable, model-backed fixtures with an appropriate
  tolerance.
- Treat event arrival order as unspecified across separate items unless the
  documented contract guarantees ordering.
- Separate deterministic protocol tests from model-backed inference tests.
  Model-backed tests should be marked/skipped explicitly when assets are absent,
  and their execution status should be reported separately from target-level
  build/test success.
- When an expected behavior is unclear for the OpenAI WebRTC media-track flow,
  confirm it from the applicable API documentation before making it a required
  conformance assertion.
