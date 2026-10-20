#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""End-to-end WebRTC transcription tests mirroring references/index.html.

Connection types covered:
  * OVMS WebRTC              - JSON signaling on /v1/webrtc/sessions, "transcript" data channel
  * OpenAI Realtime API      - SDP over https://api.openai.com/v1/realtime/calls, "oai-events" data channel
  * OVMS OpenAI Compatible   - same protocol as OpenAI, served by OVMS on /v1/realtime/calls; the SDP offer and
                               the session naming the transcription model are sent as a multipart form

Environment:
  OVMS_URL             OVMS REST base URL (default: http://localhost:9112)
  OVMS_MODEL           served speech-to-text graph name (default: voxtral-asr)
  OVMS_ANNOUNCED_IP    optional; set to the OVMS_WEBRTC_ANNOUNCED_IP value the server runs with. OVMS candidates
                       with any other address are dropped, simulating a client on another machine that can
                       only reach the announced (1:1 NAT) address
  OPENAI_API_KEY       required for the OpenAI Realtime API test (skipped otherwise)
  OVMS_API_KEY         optional bearer token sent to OVMS /v1/realtime/calls
  WEBRTC_AUDIO_FILE    audio file to stream (default: references/SayLargestAnimal.webm)
  EXPECTED_KEYWORD     word expected in the transcript (default: animal)

Run (OVMS serving a speech2text model, e.g. `ovms --rest_port 9112 --source_model OpenVINO/voxtral-realtime-2602 --model_repository_path models --model_name voxtral-asr --task speech2text`):
  pip install -r tests/webrtc_e2e/requirements.txt
  pytest -v tests/webrtc_e2e
"""

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from aiortc.sdp import candidate_from_sdp

REPO_ROOT = Path(__file__).resolve().parents[2]
OVMS_URL = os.environ.get("OVMS_URL", "http://localhost:9112").rstrip("/")
OVMS_MODEL = os.environ.get("OVMS_MODEL", "voxtral-asr")
OVMS_ANNOUNCED_IP = os.environ.get("OVMS_ANNOUNCED_IP", "")
OVMS_API_KEY = os.environ.get("OVMS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_CLIENT_SECRET_ENDPOINT = "https://api.openai.com/v1/realtime/client_secrets"
OPENAI_CALLS_ENDPOINT = "https://api.openai.com/v1/realtime/calls"
AUDIO_FILE = Path(os.environ.get("WEBRTC_AUDIO_FILE", REPO_ROOT / "references" / "SayLargestAnimal.webm"))
EXPECTED_KEYWORD = os.environ.get("EXPECTED_KEYWORD", "animal").lower()

CONNECT_TIMEOUT_S = 20
TRANSCRIPT_TIMEOUT_S = 90

DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"


def ovms_http():
    # Proxies from the environment must not intercept localhost signaling.
    session = requests.Session()
    session.trust_env = False
    return session


def ovms_ready():
    try:
        return ovms_http().get(f"{OVMS_URL}/v2/health/ready", timeout=3).status_code == 200
    except requests.RequestException:
        return False


requires_ovms = pytest.mark.skipif(not ovms_ready(), reason=f"OVMS is not reachable at {OVMS_URL}")


class TranscriptCollector:
    """Accumulates data channel events from either the legacy OVMS or the OpenAI protocol."""

    def __init__(self):
        self.events = []
        self.deltas = []
        self.completed = []
        self.legacy_finals = []
        self.errors = []
        self.call_location = None
        self.updated = asyncio.Event()

    def on_message(self, raw):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        self.events.append(message)
        event_type = message.get("type")
        if event_type == DELTA_EVENT:
            self.deltas.append(message.get("delta", ""))
        elif event_type == COMPLETED_EVENT:
            self.completed.append(message.get("transcript", ""))
        elif event_type == "user_delta":
            self.deltas.append(message.get("text", ""))
        elif event_type == "user_final":
            self.legacy_finals.append(message.get("text", ""))
        elif event_type in ("error", "user_error"):
            self.errors.append(message)
        self.updated.set()

    def delta_text(self):
        return "".join(self.deltas)

    async def wait_for(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.updated.clear()
            try:
                await asyncio.wait_for(self.updated.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return bool(predicate())
        return True


class AudioClient:
    """aiortc peer streaming an audio file over a sendrecv transceiver, like the reference page."""

    def __init__(self, channel_label):
        self.pc = RTCPeerConnection()
        self.player = MediaPlayer(str(AUDIO_FILE))
        self.audio_ended = asyncio.Event()
        self.collector = TranscriptCollector()
        self.channel = self.pc.createDataChannel(channel_label)
        self.channel_open = asyncio.Event()

        @self.channel.on("open")
        def _on_open():
            self.channel_open.set()

        @self.channel.on("message")
        def _on_message(message):
            self.collector.on_message(message)

        track = self.player.audio

        @track.on("ended")
        def _on_ended():
            self.audio_ended.set()

        self.pc.addTrack(track)

    async def create_offer(self):
        # aiortc gathers every ICE candidate in setLocalDescription, so the offer is complete (no trickle).
        await self.pc.setLocalDescription(await self.pc.createOffer())
        return self.pc.localDescription.sdp

    async def apply_answer(self, sdp):
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def add_remote_candidate(self, candidate, mid):
        """Adds an OVMS candidate; returns False when it is dropped by the OVMS_ANNOUNCED_IP filter."""
        fields = candidate_fields(candidate)
        if not is_reachable_ovms_candidate(fields):
            return False
        ice_candidate = candidate_from_sdp(fields)
        ice_candidate.sdpMid = mid
        await self.pc.addIceCandidate(ice_candidate)
        return True

    async def wait_connected(self):
        await asyncio.wait_for(self.channel_open.wait(), timeout=CONNECT_TIMEOUT_S)

    async def wait_audio_sent(self):
        await asyncio.wait_for(self.audio_ended.wait(), timeout=120)

    def send_event(self, event):
        self.channel.send(json.dumps(event))

    async def close(self):
        await self.pc.close()


def candidate_fields(candidate):
    fields = candidate.strip()
    if fields.startswith("a="):
        fields = fields[2:]
    if fields.startswith("candidate:"):
        fields = fields[len("candidate:"):]
    return fields


def is_reachable_ovms_candidate(fields):
    parts = fields.split()
    return not OVMS_ANNOUNCED_IP or (len(parts) >= 6 and parts[4] == OVMS_ANNOUNCED_IP)


def keep_reachable_sdp_candidates(sdp):
    """Drops OVMS SDP candidates a remote client could not reach; returns (sdp, kept candidate count)."""
    lines = []
    kept = 0
    for line in sdp.splitlines():
        if line.startswith("a=candidate:"):
            if not is_reachable_ovms_candidate(candidate_fields(line)):
                continue
            kept += 1
        lines.append(line)
    return "\r\n".join(lines) + "\r\n", kept


def assert_transcript(text, collector):
    assert not collector.errors, f"Server reported errors: {collector.errors}"
    assert EXPECTED_KEYWORD in text.lower(), (
        f"Expected '{EXPECTED_KEYWORD}' in transcript {text!r}; received events: {collector.events}")


def transcription_session(model):
    return {
        "type": "transcription",
        "audio": {"input": {"transcription": {"model": model}, "turn_detection": None}},
    }


async def run_openai_protocol(calls_url, auth_token, http_session, ovms_server=False, session_config=None):
    """Shared flow of the OpenAI Realtime API and the OVMS OpenAI compatible endpoint.

    With session_config the offer is posted as the OpenAI unified multipart form (sdp + session fields),
    otherwise as a plain application/sdp body configured by the ephemeral token.
    """
    client = AudioClient("oai-events")
    try:
        offer_sdp = await client.create_offer()
        headers = {}
        form = None
        if session_config is None:
            headers["Content-Type"] = "application/sdp"
        else:
            form = {"sdp": (None, offer_sdp), "session": (None, json.dumps(session_config))}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        response = await asyncio.to_thread(
            http_session.post, calls_url, data=None if form else offer_sdp, files=form, headers=headers, timeout=30)
        assert response.ok, f"Call request failed: {response.status_code} {response.text}"
        answer_sdp = response.text
        assert answer_sdp.startswith("v=0"), f"Expected SDP answer, got: {answer_sdp[:200]!r}"
        assert "a=candidate:" in answer_sdp, "SDP answer must embed ICE candidates (client does not trickle)"
        client.collector.call_location = response.headers.get("Location")
        if ovms_server and OVMS_ANNOUNCED_IP:
            answer_sdp, kept = keep_reachable_sdp_candidates(answer_sdp)
            assert kept, f"SDP answer has no candidate with announced IP {OVMS_ANNOUNCED_IP}"
        await client.apply_answer(answer_sdp)
        await client.wait_connected()

        await client.wait_audio_sent()
        client.send_event({"type": "input_audio_buffer.commit"})

        completed = await client.collector.wait_for(
            lambda: client.collector.completed or client.collector.errors, TRANSCRIPT_TIMEOUT_S)
        assert completed, f"No '{COMPLETED_EVENT}' received; events: {client.collector.events}"
        assert_transcript(" ".join(client.collector.completed), client.collector)
        return client.collector
    finally:
        await client.close()


@requires_ovms
@pytest.mark.asyncio
async def test_ovms_webrtc_transcription():
    http = ovms_http()
    client = AudioClient("transcript")
    session_id = None
    try:
        offer_sdp = await client.create_offer()
        response = await asyncio.to_thread(
            http.post, f"{OVMS_URL}/v1/webrtc/sessions", json={"type": "offer", "sdp": offer_sdp, "model": OVMS_MODEL},
            timeout=30)
        assert response.ok, f"Session request failed: {response.status_code} {response.text}"
        answer = response.json()
        assert answer["type"] == "answer"
        session_id = answer["session_id"]
        await client.apply_answer(answer["sdp"])

        known = set()
        added = 0

        async def add_candidates(candidates):
            nonlocal added
            for candidate in candidates:
                key = (candidate["candidate"], candidate["mid"])
                if key not in known:
                    known.add(key)
                    if await client.add_remote_candidate(candidate["candidate"], candidate["mid"]):
                        added += 1

        await add_candidates(answer.get("candidates", []))
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while not client.channel_open.is_set() and time.monotonic() < deadline:
            polled = await asyncio.to_thread(
                http.get, f"{OVMS_URL}/v1/webrtc/sessions/{session_id}/candidates", timeout=5)
            assert polled.ok, polled.text
            await add_candidates(polled.json().get("candidates", []))
            try:
                await asyncio.wait_for(client.channel_open.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass
        assert added, f"No OVMS candidate with announced IP {OVMS_ANNOUNCED_IP}; received: {sorted(known)}"
        await client.wait_connected()

        await client.wait_audio_sent()
        # Voxtral streams deltas while audio flows; wait until the keyword shows up.
        await client.collector.wait_for(
            lambda: EXPECTED_KEYWORD in client.collector.delta_text().lower() or client.collector.errors,
            TRANSCRIPT_TIMEOUT_S)
        assert_transcript(client.collector.delta_text(), client.collector)
    finally:
        if session_id is not None:
            await asyncio.to_thread(http.post, f"{OVMS_URL}/v1/webrtc/sessions/{session_id}/close", timeout=60)
        await client.close()


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY is not set")
@pytest.mark.asyncio
async def test_openai_realtime_transcription():
    http = requests.Session()
    # Same ephemeral token request as references/openai_backend.py.
    session_config = transcription_session(os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "gpt-live-transcribe"))
    token_response = await asyncio.to_thread(
        http.post, OPENAI_CLIENT_SECRET_ENDPOINT,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"session": session_config}, timeout=15)
    assert token_response.status_code in (200, 201), token_response.text
    token = token_response.json().get("value")
    assert token, f"No ephemeral token in response: {token_response.text}"

    await run_openai_protocol(OPENAI_CALLS_ENDPOINT, token, http)


@requires_ovms
@pytest.mark.asyncio
async def test_ovms_openai_compatible_transcription():
    collector = await run_openai_protocol(f"{OVMS_URL}/v1/realtime/calls", OVMS_API_KEY, ovms_http(), ovms_server=True,
                                          session_config=transcription_session(OVMS_MODEL))
    item_ids = {event.get("item_id") for event in collector.events if event.get("type") == COMPLETED_EVENT}
    assert all(item_ids), f"Completed events must carry item_id: {collector.events}"
    location = collector.call_location
    assert location and location.startswith("/v1/realtime/calls/"), f"Unexpected Location header: {location!r}"
    hangup = await asyncio.to_thread(ovms_http().post, f"{OVMS_URL}{location}/hangup", timeout=30)
    assert hangup.ok, f"Hangup failed: {hangup.status_code} {hangup.text}"
