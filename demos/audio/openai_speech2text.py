#
# Copyright (c) 2025 Intel Corporation
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

from pathlib import Path
from openai import OpenAI

filename = "speech.wav"
url="http://localhost/v3"


speech_file_path = Path(__file__).parent / filename
client = OpenAI(base_url=url, api_key="not_used")

audio_file = open(filename, "rb")
transcript = client.audio.transcriptions.create(
  model="whisper",
  file=audio_file
)

print(transcript)