# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VP8 -> served YOLOX + ByteTrack -> oai-events smoke test."""

import asyncio
import json
import os
from pathlib import Path

import av
import pytest
import requests
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


VIDEO_URL = os.environ.get("OVMS_YOLO_URL", "").rstrip("/")
VIDEO_FILE = Path(os.environ.get("WEBRTC_YOLO_VIDEO_FILE", ""))
VIDEO_GRAPH = os.environ.get("OVMS_WEBRTC_VIDEO_GRAPH_NAME", "YoloByteTrackWebRTC")
TRACKING_EVENT = "conversation.item.input_video_tracking.completed"
SIGNALING_TIMEOUT_SECONDS = 30
VIDEO_TIMEOUT_SECONDS = 60


@pytest.mark.skipif(
    not VIDEO_URL or not VIDEO_FILE.is_file(),
    reason="Set OVMS_YOLO_URL and WEBRTC_YOLO_VIDEO_FILE to run the YOLO ByteTrack POC test",
)
@pytest.mark.asyncio
async def test_native_vp8_yolox_bytetrack_tracking():
    with av.open(str(VIDEO_FILE)) as video:
        expected_frames = sum(1 for _ in video.decode(video=0))
    assert expected_frames > 0

    peer = RTCPeerConnection()
    player = MediaPlayer(str(VIDEO_FILE))
    assert player.video is not None
    channel = peer.createDataChannel("oai-events")
    tracking_events = []
    errors = []
    video_ended = asyncio.Event()
    player.video.on("ended", video_ended.set)
    http = requests.Session()
    http.trust_env = False
    location = None

    @channel.on("message")
    def on_message(raw):
        event = json.loads(raw)
        if event["type"] == "error":
            errors.append(event)
        elif event["type"] == TRACKING_EVENT:
            tracking_events.append(event)

    try:
        transceiver = peer.addTransceiver(player.video, direction="sendonly")
        transceiver.setCodecPreferences([
            codec for codec in RTCRtpSender.getCapabilities("video").codecs
            if codec.mimeType.lower() == "video/vp8"
        ])
        await peer.setLocalDescription(await peer.createOffer())
        response = await asyncio.to_thread(
            http.post, f"{VIDEO_URL}/v1/realtime/calls",
            files={
                "sdp": (None, peer.localDescription.sdp),
                "session": (None, json.dumps({"type": "transcription", "model": VIDEO_GRAPH})),
            },
            timeout=SIGNALING_TIMEOUT_SECONDS,
        )
        assert response.status_code == 200, response.text
        location = response.headers["Location"]
        assert "VP8/90000" in response.text
        assert "H264/90000" not in response.text
        await peer.setRemoteDescription(RTCSessionDescription(sdp=response.text, type="answer"))
        await asyncio.wait_for(video_ended.wait(), timeout=VIDEO_TIMEOUT_SECONDS)
        await asyncio.sleep(3)

        assert not errors, errors
        assert tracking_events
        assert all(event["width"] > 0 and event["height"] > 0 for event in tracking_events)
        assert all(isinstance(event["objects"], list) for event in tracking_events)
        assert all(
            0 <= obj["box"]["x"] <= 1 and
            0 <= obj["box"]["y"] <= 1 and
            0 <= obj["box"]["w"] <= 1 and
            0 <= obj["box"]["h"] <= 1 and
            obj["box"]["x"] + obj["box"]["w"] <= 1.000001 and
            obj["box"]["y"] + obj["box"]["h"] <= 1.000001 and
            0 <= obj["score"] <= 1
            for event in tracking_events for obj in event["objects"]
        )
        persistent_track = any(
            {obj["id"] for obj in left["objects"]} &
            {obj["id"] for obj in right["objects"]}
            for left, right in zip(tracking_events, tracking_events[1:])
        )
        assert persistent_track, "No tracker ID persisted across consecutive processed frames"
        assert all(
            left["frame"] < right["frame"]
            for left, right in zip(tracking_events, tracking_events[1:])
        )
        print(json.dumps({
            "source_frames": expected_frames,
            "tracking_events": len(tracking_events),
            "last_event": tracking_events[-1],
        }))
    finally:
        if player.video:
            player.video.stop()
        await peer.close()
        if location:
            response = await asyncio.to_thread(
                http.post, f"{VIDEO_URL}{location}/hangup", timeout=SIGNALING_TIMEOUT_SECONDS,
            )
            assert response.status_code in (200, 204), response.text
        http.close()
