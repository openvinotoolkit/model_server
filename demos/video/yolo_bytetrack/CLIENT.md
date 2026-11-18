# Web client integration: YOLOX + ByteTrack

Reference client: [index.html](index.html). This is a POC protocol extension,
not an OpenAI-standard video API.

## Connection and channels

| Channel | Purpose |
| --- | --- |
| HTTP(S) | Exchange SDP and end the call. Base URL defaults to `http://localhost:6666`. |
| WebRTC video track | Browser sends VP8 video to OVMS; no processed video is returned. |
| WebRTC data channel `oai-events` | Server sends JSON text messages containing tracking results, session information, and errors. |
| Optional WebRTC audio track | Microphone audio for a separately configured ASR graph; transcripts share `oai-events`. |

Render the local camera stream in a `<video>` element and draw returned boxes
on a canvas overlay. Do not send images as JSON, base64, or data-channel messages.
No WebSocket or polling connection is required.

The page must run on localhost or HTTPS for camera access. Configure OVMS's
`--allowed_origins` for the page's origin. For remote deployment, make the REST
endpoint and UDP media range (`52000-52100` by default) reachable, and set
`OVMS_WEBRTC_ANNOUNCED_IP` to the server's client-reachable IP. An HTTPS page
also needs an HTTPS signaling endpoint to avoid mixed-content blocking.
The demo uses direct ICE connectivity, not a configured STUN/TURN service.

## Connect

1. Create `new RTCPeerConnection({iceServers: []})`.
2. Before creating the offer, create `peer.createDataChannel("oai-events")`
   with the default reliable, ordered settings. Handle `message` with
   `JSON.parse(event.data)` and dispatch by `type`.
3. Request camera access with `getUserMedia({video: true, audio: false})`.
   Add the video track to the peer. The demo requests 640x480 at 8-10 fps.
4. Select only `video/VP8` codecs using
   `RTCRtpSender.getCapabilities("video")` and the video transceiver's
   `setCodecPreferences()`. Fail visibly if VP8 is unavailable.
5. Create an offer, call `setLocalDescription()`, then wait for
   `iceGatheringState === "complete"` with a timeout. This POC exchanges
   candidates in SDP; do not rely on trickle-ICE requests.
6. POST a browser `FormData` body to `/v1/realtime/calls`:
   - `sdp`: `peer.localDescription.sdp`
   - `session`: the following JSON serialized as a string:

   ```json
   {"type":"transcription","model":"YoloByteTrackWebRTC"}
   ```

   Let the browser set the multipart `Content-Type` and boundary.
   Video selection is server-side via `OVMS_WEBRTC_VIDEO_GRAPH`; the request
   model must match it for this video-only flow.
7. Check the HTTP response before consuming it. Success is `200`, with an
   `application/sdp` answer body. Pass it to
   `peer.setRemoteDescription({type: "answer", sdp: answer})`.
8. Save the response's `Location` call path if accessible. Also handle
   `session.created` on `oai-events`: its `session.id` yields the fallback path
   `/v1/realtime/calls/<session.id>`. This avoids depending on cross-origin
   access to the `Location` header.

No client data-channel commands or per-frame acknowledgments are needed for
video tracking. Display signaling failures, `peer.connectionState` failures,
and server events with `type: "error"` (`event.error.message`).

## Tracking output

Each graph-processed frame produces a JSON text message:

```json
{
  "type": "conversation.item.input_video_tracking.completed",
  "event_id": "event_20",
  "item_id": "video_3_13",
  "frame": 13,
  "rtp_timestamp": 3715954780,
  "width": 640,
  "height": 480,
  "objects": [
    {
      "id": 4,
      "label": "person",
      "score": 0.91,
      "box": {"x": 0.12, "y": 0.20, "w": 0.30, "h": 0.60}
    }
  ],
  "transcript": "person#4 [0.12,0.20,0.30,0.60]"
}
```

- `event_id` and `item_id`: opaque identifiers; do not parse their components.
- `frame`: increasing per-call submission counter, not a camera frame index.
  Frames may be dropped under backpressure; do not expect consecutive values
  or one result per captured frame.
- `rtp_timestamp`: original 32-bit, 90 kHz video RTP timestamp. It wraps and is
  not a wall-clock timestamp or a timestamp in milliseconds.
- `width`, `height`: original decoded frame dimensions.
- `objects`: the tracks for this result, **not incremental updates**. Replace
  the displayed set each time; `[]` means clear all boxes.
- `id`: numeric tracker ID, scoped to this call. IDs can change after a track
  is lost; they are not persistent identities across calls.
- `label`, `score`: class name and numeric detection confidence.
- `box`: normalized top-left `x`, `y`, width `w`, and height `h`, clamped to
  the image bounds. Multiply horizontal values by the displayed image width
  and vertical values by its height. Account for letterboxing, cropping, and
  mirroring in the UI.
- `transcript`: human-readable summary; use `objects`, not this string, for
  application logic.

Results arrive after inference and are not synchronized to the currently
displayed local preview frame.

## Optional microphone and shutdown

To add ASR, send a microphone track and add
`audio.input.transcription.model` to the session JSON, naming a served ASR
graph. Audio transcripts use
`conversation.item.input_audio_transcription.delta` (`delta`) and
`conversation.item.input_audio_transcription.completed` (`transcript`).
The provided video-only server config does not include an ASR model.

On stop, stop all local media tracks, close the peer connection, and POST to
`<saved call path>/hangup` on the same REST server to release the server session.
Check the hangup response and surface failures.
