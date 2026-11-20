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
"""End-to-end WebRTC transcription tests for OpenAI Realtime signaling.

Connection types covered:
  * OpenAI Realtime API      - SDP over https://api.openai.com/v1/realtime/calls, "oai-events" data channel
  * OVMS OpenAI Compatible   - same protocol as OpenAI, served by OVMS on /v1/realtime/calls

Environment:
  OVMS_URL             OVMS REST base URL (default: http://localhost:9112)
  OVMS_MODEL           served speech-to-text graph name (default: voxtral-asr)
  OVMS_REALTIME_MODEL  served Voxtral Realtime graph name for live-delta testing
                       (defaults to OVMS_MODEL only when it is voxtral-asr)
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

REPO_ROOT = Path(__file__).resolve().parents[2]
OVMS_URL = os.environ.get("OVMS_URL", "http://localhost:9112").rstrip("/")
OVMS_MODEL = os.environ.get("OVMS_MODEL", "voxtral-asr")
OVMS_REALTIME_MODEL = os.environ.get(
    "OVMS_REALTIME_MODEL", OVMS_MODEL if OVMS_MODEL == "voxtral-asr" else "")
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
COMMITTED_EVENT = "input_audio_buffer.committed"


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
requires_realtime_model = pytest.mark.skipif(
    not OVMS_REALTIME_MODEL,
    reason="Set OVMS_REALTIME_MODEL to a served Voxtral Realtime graph to run live-delta coverage",
)


class TranscriptCollector:
    """Accumulates OpenAI Realtime transcription events from the data channel."""

    def __init__(self):
        self.events = []
        self.deltas = []
        self.completed = []
        self.errors = []
        self.invalid_messages = []
        self.call_location = None
        self.updated = asyncio.Event()

    def on_message(self, raw):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self.invalid_messages.append(raw)
            self.updated.set()
            return
        if not isinstance(message, dict):
            self.invalid_messages.append(raw)
            self.updated.set()
            return
        self.events.append(message)
        event_type = message.get("type")
        if event_type == DELTA_EVENT:
            self.deltas.append(message.get("delta", ""))
        elif event_type == COMPLETED_EVENT:
            self.completed.append(message.get("transcript", ""))
        elif event_type == "error":
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
            if predicate():
                return True
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

    async def wait_connected(self):
        await asyncio.wait_for(self.channel_open.wait(), timeout=CONNECT_TIMEOUT_S)

    async def wait_audio_sent(self):
        await asyncio.wait_for(self.audio_ended.wait(), timeout=120)

    def send_event(self, event):
        self.channel.send(json.dumps(event))

    async def close(self):
        await self.pc.close()


class ControlClient:
    """WebRTC peer with a negotiated audio section but no media source."""

    def __init__(self):
        self.pc = RTCPeerConnection()
        self.pc.addTransceiver("audio", direction="sendonly")
        self.channel = self.pc.createDataChannel("oai-events")
        self.channel_open = asyncio.Event()
        self.collector = TranscriptCollector()

        @self.channel.on("open")
        def _on_open():
            self.channel_open.set()

        @self.channel.on("message")
        def _on_message(message):
            self.collector.on_message(message)

    async def create_offer(self):
        await self.pc.setLocalDescription(await self.pc.createOffer())
        return self.pc.localDescription.sdp

    async def apply_answer(self, sdp):
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def wait_connected(self):
        await asyncio.wait_for(self.channel_open.wait(), timeout=CONNECT_TIMEOUT_S)

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


def assert_event_envelope(event, expected_type):
    assert event.get("type") == expected_type, event
    assert isinstance(event.get("event_id"), str) and event["event_id"], event


def assert_transcription_event_contract(events, expected_items=None):
    assert not events.invalid_messages, f"Non-JSON data-channel messages: {events.invalid_messages}"
    assert not events.errors, f"Server reported errors: {events.errors}"

    committed = [event for event in events.events if event.get("type") == COMMITTED_EVENT]
    deltas = [event for event in events.events if event.get("type") == DELTA_EVENT]
    completed = [event for event in events.events if event.get("type") == COMPLETED_EVENT]
    assert committed, f"No committed-item event received: {events.events}"
    assert completed, f"No completed-transcription event received: {events.events}"

    for event in committed:
        assert_event_envelope(event, COMMITTED_EVENT)
        assert isinstance(event.get("item_id"), str) and event["item_id"], event
        assert "previous_item_id" in event, event
        assert event["previous_item_id"] is None or isinstance(event["previous_item_id"], str), event
    for event in deltas:
        assert_event_envelope(event, DELTA_EVENT)
        assert isinstance(event.get("item_id"), str) and event["item_id"], event
        assert isinstance(event.get("content_index"), int) and event["content_index"] >= 0, event
        assert isinstance(event.get("delta"), str) and event["delta"], event
    for event in completed:
        assert_event_envelope(event, COMPLETED_EVENT)
        assert isinstance(event.get("item_id"), str) and event["item_id"], event
        assert isinstance(event.get("content_index"), int) and event["content_index"] >= 0, event
        assert isinstance(event.get("transcript"), str), event

    committed_ids = {event["item_id"] for event in committed}
    completed_ids = {event["item_id"] for event in completed}
    delta_ids = {event["item_id"] for event in deltas}
    expected_ids = set(expected_items) if expected_items is not None else committed_ids
    assert expected_ids <= committed_ids, f"Missing committed item IDs: {events.events}"
    assert completed_ids == expected_ids, f"Committed/completed item mismatch: {events.events}"
    assert len(completed) == len(expected_ids), f"Expected one completion per item: {events.events}"
    assert delta_ids <= committed_ids, f"Delta refers to an uncommitted item: {events.events}"
    return committed, deltas, completed


def assert_protocol_error(event):
    assert_event_envelope(event, "error")
    error = event.get("error")
    assert isinstance(error, dict), event
    assert isinstance(error.get("type"), str) and error["type"], event
    assert isinstance(error.get("message"), str) and error["message"], event


def transcription_session(model):
    return {
        "type": "transcription",
        "audio": {"input": {"transcription": {"model": model}, "turn_detection": None}},
    }


async def connect_ovms_client(client, session_config=None):
    offer_sdp = await client.create_offer()
    form = {"sdp": (None, offer_sdp)}
    if session_config is not None:
        form["session"] = (None, json.dumps(session_config))
    headers = {"Authorization": f"******"} if OVMS_API_KEY else {}
    response = await asyncio.to_thread(
        ovms_http().post, f"{OVMS_URL}/v1/realtime/calls", files=form, headers=headers, timeout=30)
    assert response.ok, f"Call request failed: {response.status_code} {response.text}"
    assert response.headers["Content-Type"].split(";")[0] == "application/sdp"
    assert response.text.startswith("v=0"), f"Expected SDP answer, got: {response.text[:200]!r}"
    assert "a=candidate:" in response.text, "SDP answer must embed ICE candidates (client does not trickle)"
    client.collector.call_location = response.headers.get("Location")
    answer_sdp = response.text
    if OVMS_ANNOUNCED_IP:
        answer_sdp, kept = keep_reachable_sdp_candidates(answer_sdp)
        assert kept, f"SDP answer has no candidate with announced IP {OVMS_ANNOUNCED_IP}"
    await client.apply_answer(answer_sdp)
    await client.wait_connected()
    return client


async def hangup_ovms_client(client):
    try:
        location = client.collector.call_location
        if location:
            headers = {"Authorization": f"******"} if OVMS_API_KEY else {}
            response = await asyncio.to_thread(
                ovms_http().post, f"{OVMS_URL}{location}/hangup", headers=headers, timeout=30)
            assert response.ok, f"Hangup failed: {response.status_code} {response.text}"
    finally:
        await client.close()


async def run_openai_protocol(calls_url, auth_token, http_session, ovms_server=False, session_config=None):
    """Shared flow of the OpenAI Realtime API and the OVMS OpenAI compatible endpoint.

    With session_config the offer is posted as the OpenAI unified multipart form (sdp + session fields),
    otherwise as a plain application/sdp body (OpenAI uses an ephemeral token; OVMS selects its sole STT graph).
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
        assert response.headers["Content-Type"].split(";")[0] == "application/sdp"
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


async def create_ovms_control_call():
    client = ControlClient()
    try:
        return await connect_ovms_client(client, transcription_session(OVMS_MODEL))
    except Exception:
        await client.close()
        raise


async def close_ovms_control_call(client):
    await hangup_ovms_client(client)


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
@pytest.mark.parametrize("signaling", ["multipart", "plain_sdp"])
async def test_ovms_openai_compatible_transcription(signaling):
    collector = await run_openai_protocol(f"{OVMS_URL}/v1/realtime/calls", OVMS_API_KEY, ovms_http(), ovms_server=True,
                                          session_config=transcription_session(OVMS_MODEL) if signaling == "multipart" else None)
    item_ids = {event.get("item_id") for event in collector.events if event.get("type") == COMPLETED_EVENT}
    assert all(item_ids), f"Completed events must carry item_id: {collector.events}"
    delta_events = [event for event in collector.events if event.get("type") == DELTA_EVENT]
    assert any(event.get("type") == "session.created" for event in collector.events), collector.events
    assert_transcription_event_contract(collector)
    assert delta_events, f"No transcription delta received: {collector.events}"
    location = collector.call_location
    assert location and location.startswith("/v1/realtime/calls/"), f"Unexpected Location header: {location!r}"
    headers = {"Authorization": f"Bearer {OVMS_API_KEY}"} if OVMS_API_KEY else {}
    hangup = await asyncio.to_thread(ovms_http().post, f"{OVMS_URL}{location}/hangup", headers=headers, timeout=30)
    assert hangup.ok, f"Hangup failed: {hangup.status_code} {hangup.text}"


@requires_ovms
@pytest.mark.asyncio
@requires_realtime_model
async def test_ovms_realtime_emits_delta_before_audio_is_committed():
    client = AudioClient("oai-events")
    try:
        await connect_ovms_client(client, transcription_session(OVMS_REALTIME_MODEL))
        got_delta = await client.collector.wait_for(
            lambda: bool(client.collector.deltas) or bool(client.collector.errors),
            TRANSCRIPT_TIMEOUT_S,
        )
        assert got_delta, f"No transcription delta while audio was streaming: {client.collector.events}"
        assert not client.collector.errors, f"Server reported errors: {client.collector.errors}"
        assert client.collector.deltas, f"No delta received before commit: {client.collector.events}"
        assert not client.audio_ended.is_set(), (
            "Audio ended before the first delta; this test requires a live Voxtral Realtime model "
            "and an audio fixture long enough to observe incremental output"
        )

        await client.wait_audio_sent()
        client.send_event({"type": "input_audio_buffer.commit"})
        await client.collector.wait_for(
            lambda: bool(client.collector.completed) or bool(client.collector.errors),
            TRANSCRIPT_TIMEOUT_S,
        )
        assert_transcription_event_contract(client.collector)
    finally:
        await hangup_ovms_client(client)


@requires_ovms
@pytest.mark.asyncio
@requires_realtime_model
async def test_ovms_realtime_associates_multiple_committed_turns():
    client = AudioClient("oai-events")
    try:
        await connect_ovms_client(client, transcription_session(OVMS_REALTIME_MODEL))
        delta_received = await client.collector.wait_for(
            lambda: bool(client.collector.deltas) or bool(client.collector.errors),
            TRANSCRIPT_TIMEOUT_S,
        )
        assert delta_received, f"No live delta before the first commit: {client.collector.events}"
        assert not client.collector.errors, f"Server reported errors: {client.collector.errors}"
        assert not client.audio_ended.is_set(), (
            "Audio ended before a live delta arrived; use a longer fixture or a faster Voxtral Realtime setup"
        )
        client.send_event({"type": "input_audio_buffer.commit"})
        first_commit_received = await client.collector.wait_for(
            lambda: sum(event.get("type") == COMMITTED_EVENT for event in client.collector.events) >= 1,
            10,
        )
        assert first_commit_received, f"No first committed event: {client.collector.events}"

        await client.wait_audio_sent()
        client.send_event({"type": "input_audio_buffer.commit"})
        completed_twice = await client.collector.wait_for(
            lambda: sum(event.get("type") == COMPLETED_EVENT for event in client.collector.events) >= 2
            or bool(client.collector.errors),
            TRANSCRIPT_TIMEOUT_S,
        )
        assert completed_twice, f"Did not complete both turns: {client.collector.events}"
        committed, _, completed = assert_transcription_event_contract(client.collector)
        assert len(committed) == 2, f"Expected two committed turns: {client.collector.events}"
        assert len(completed) == 2, f"Expected two completed turns: {client.collector.events}"
        assert len({event["item_id"] for event in committed}) == 2, client.collector.events
    finally:
        await hangup_ovms_client(client)


@requires_ovms
@pytest.mark.asyncio
@pytest.mark.parametrize("raw_event", [
    "{",
    '{"event":"input_audio_buffer.commit"}',
    '{"type":42}',
    '{"type":"client.event.not_supported"}',
])
async def test_ovms_realtime_returns_protocol_error_for_invalid_client_events(raw_event):
    client = await create_ovms_control_call()
    try:
        client.channel.send(raw_event)
        received = await client.collector.wait_for(
            lambda: bool(client.collector.errors) or bool(client.collector.invalid_messages), 10)
        assert received, f"No protocol response received: {client.collector.events}"
        assert not client.collector.invalid_messages, (
            f"Server sent malformed JSON instead of a protocol error: {client.collector.invalid_messages}"
        )
        assert client.collector.errors, f"No structured protocol error received: {client.collector.events}"
        assert_protocol_error(client.collector.errors[0])
    finally:
        await close_ovms_control_call(client)


@requires_ovms
@pytest.mark.asyncio
async def test_ovms_realtime_rejects_empty_audio_commit():
    client = await create_ovms_control_call()
    try:
        client.channel.send(json.dumps({"type": "input_audio_buffer.commit"}))
        received = await client.collector.wait_for(lambda: bool(client.collector.errors), 10)
        assert received, f"No error event received: {client.collector.events}"
        assert_protocol_error(client.collector.errors[0])
    finally:
        await close_ovms_control_call(client)


@requires_ovms
@pytest.mark.asyncio
async def test_ovms_realtime_acknowledges_transcription_session_update():
    client = await create_ovms_control_call()
    try:
        prompt = "Transcribe this as a customer support call."
        client.channel.send(json.dumps({
            "type": "session.update",
            "session": {"audio": {"input": {"transcription": {"prompt": prompt}}}},
        }))
        received = await client.collector.wait_for(
            lambda: any(event.get("type") in ("session.updated", "error") for event in client.collector.events),
            10,
        )
        assert received, f"No response to session.update: {client.collector.events}"
        assert not client.collector.errors, f"Valid transcription update was rejected: {client.collector.errors}"
        updated = next(event for event in client.collector.events if event.get("type") == "session.updated")
        assert_event_envelope(updated, "session.updated")
        session = updated.get("session")
        assert isinstance(session, dict), updated
        transcription = session.get("audio", {}).get("input", {}).get("transcription", {})
        assert transcription.get("prompt") == prompt, updated
    finally:
        await close_ovms_control_call(client)


@requires_ovms
@pytest.mark.asyncio
async def test_ovms_realtime_rejects_update_to_unserved_model():
    client = AudioClient("oai-events")
    try:
        await connect_ovms_client(client, transcription_session(OVMS_MODEL))
        client.channel.send(json.dumps({
            "type": "session.update",
            "session": {"audio": {"input": {"transcription": {"model": "model-not-served"}}}},
        }))
        received = await client.collector.wait_for(
            lambda: any(event.get("type") in ("session.updated", "error") for event in client.collector.events),
            10,
        )
        assert received, f"No response to unsupported model update: {client.collector.events}"
        assert client.collector.errors, (
            "An update requesting an unserved model must not be silently accepted or ignored: "
            f"{client.collector.events}"
        )
        assert_protocol_error(client.collector.errors[0])
        assert not any(event.get("type") == "session.updated" for event in client.collector.events), (
            f"Unsupported model update was acknowledged: {client.collector.events}"
        )
        await client.wait_audio_sent()
        client.send_event({"type": "input_audio_buffer.commit"})
        completed = await client.collector.wait_for(
            lambda: bool(client.collector.completed) or len(client.collector.errors) > 1,
            TRANSCRIPT_TIMEOUT_S,
        )
        assert completed, f"Call did not continue after rejecting the model update: {client.collector.events}"
        assert len(client.collector.errors) == 1, (
            f"Unexpected errors after rejecting the model update: {client.collector.errors}"
        )
        assert any(event.get("type") == COMPLETED_EVENT for event in client.collector.events), (
            f"Previously selected model did not continue transcribing: {client.collector.events}"
        )
    finally:
        await hangup_ovms_client(client)


@requires_ovms
@pytest.mark.parametrize(("content_type", "body", "expected_error"), [
    ("application/json", "{}", "Content-Type"),
    ("application/sdp-invalid", "v=0\r\n", "Content-Type"),
    ("application/sdp", "", "SDP offer"),
    ("application/sdp", "not SDP", "SDP offer"),
    ("multipart/form-data; boundary=x", "not a multipart form", "SDP offer"),
])
def test_ovms_realtime_invalid_signaling(content_type, body, expected_error):
    headers = {"Content-Type": content_type}
    if OVMS_API_KEY:
        headers["Authorization"] = f"Bearer {OVMS_API_KEY}"
    response = ovms_http().post(f"{OVMS_URL}/v1/realtime/calls", data=body, headers=headers, timeout=10)
    assert response.status_code == 400, response.text
    assert response.headers["Content-Type"].split(";")[0] == "application/json"
    assert expected_error in response.json()["error"]
    assert "Location" not in response.headers


@requires_ovms
@pytest.mark.skipif(not OVMS_API_KEY, reason="OVMS_API_KEY is not set")
@pytest.mark.parametrize("path", ["/v1/realtime/calls", "/v1/realtime/calls/missing/hangup"])
def test_ovms_realtime_requires_authentication(path):
    response = ovms_http().post(f"{OVMS_URL}{path}", data="v=0\r\n",
                                headers={"Content-Type": "application/sdp"}, timeout=10)
    assert response.status_code == 401, response.text
