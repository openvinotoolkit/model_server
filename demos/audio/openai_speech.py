#
# Copyright (c) 2021 Intel Corporation
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

prompt = "Intel Corporation is an American multinational technology company headquartered in Santa Clara, California.[3] Intel designs, manufactures, and sells computer components such as central processing units (CPUs) and related products for business and consumer markets. It was the world's third-largest semiconductor chip manufacturer by revenue in 2024[4] and has been included in the Fortune 500 list of the largest United States corporations by revenue since 2007. It was one of the first companies listed on Nasdaq. Since 2025, it is partially owned by the United States government."
filename = "speech.wav"
url="http://localhost:80/v3"


speech_file_path = Path(__file__).parent / "speech.wav"
client = OpenAI(base_url=url, api_key="not_used")

# with client.audio.speech.with_streaming_response.create(
#   model="whisper",
#   voice="alloy",
#   input=prompt
# ) as response:
#   response.stream_to_file(speech_file_path)

audio_file = open("speech.wav", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper",
  file=audio_file
)


print(transcript)