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