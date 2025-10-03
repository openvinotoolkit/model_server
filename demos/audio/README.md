# Audio endpoints


## Audio synthesis

python export_model.py speech --source_model microsoft/speecht5_tts --vocoder microsoft/speecht5_hifigan


docker run -p 8000:8000 -it -v $(pwd)/models/:/models openvino/model_server --model_name speecht5_tts --model_path /models/microsoft/speecht5_tts --rest_port 8000

curl http://mclx-23.sclab.intel.com/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"speecht5_tts\", \"input\": \"The quick brown fox jumped over the lazy dog.\"}" -o audio.mp3
