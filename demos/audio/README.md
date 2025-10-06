# Audio endpoints


## Audio synthesis

python export_model.py speech --source_model microsoft/speecht5_tts --vocoder microsoft/speecht5_hifigan --weight-format fp16

docker run -p 8000:8000 -d -v $(pwd)/models/:/models openvino/model_server --model_name speecht5_tts --model_path /models/microsoft/speecht5_tts --rest_port 8000

curl http://localhost/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"speecht5_tts\", \"input\": \"The quick brown fox jumped over the lazy dog.\"}" -o audio.wav





## Audio transcription

python export_model.py transcription --source_model openai/whisper-large-v2  --weight-format fp16


docker run -p 8000:8000 -it -v $(pwd)/models/:/models openvino/model_server --model_name whisper --model_path /models/openai/whisper-large-v2 --rest_port 8000


curl http://localhost/v3/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@audio.wav" -F model="whisper"




