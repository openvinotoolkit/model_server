pip install -r tests/webrtc_e2e/requirements.txt
OVMS_URL=http://localhost:8443 \
OVMS_MODEL=voxtral-asr \
OVMS_REALTIME_MODEL=voxtral-asr \
.venv/bin/python3 -m pytest -v  test_webrtc_transcription.py
