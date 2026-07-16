### Chat Completions API, Unary, AUDIO+TEXT -> TEXT
```
python3 omni_ccompletion_unary_AT_T.py kokoro.wav
```
```
OpenVINO is an open-source toolkit for deploying high-performance AI solutions across cloud, AI PCs, edge devices, and physical AI-like. Develop your applications with both generative and conventional AI models, coming from the most popular model frameworks. Convert, optimize, and run inference, utilizing the full potential of Intel hardware. There are four main tools in OpenVINO to meet all your deployment needs.
```

### Responses API, Unary, AUDIO+TEXT -> TEXT
```
python3 omni_responses_unary_AT_T.py kokoro.wav
```
```
OpenVINO is an open-source toolkit for deploying high-performance AI solutions across cloud, AI PCs, edge devices, and physical AI-like. Develop your applications with both generative and conventional AI models coming from the most popular model frameworks. Convert, optimize, and run inference, utilizing the full potential of Intel hardware. There are four main tools in OpenVINO to meet all your deployment needs.
```

### Chat Completions API, Unary, AUDIO+TEXT+IMAGE -> AUDIO+TEXT
```
python3 omni_ccompletion_unary_ATI_AT.py --audio kokoro.wav --image ../common/static/images/snail.jpeg --prompt "In the recording replace OpenVINO with animal depicted in this image and produce new audio with it."
```
```
Audio input: kokoro.wav (format: wav)
Image input: ../common/static/images/snail.jpeg

Text: A snail is an open-source toolkit for deploying high-performance AI solutions across cloud, AI PCs, edge devices, and physical AI-like. Develop your applications with both generative and conventional AI models, coming from the most popular model frameworks. Convert, optimize, and run inference, utilizing the full potential of Intel hardware. There are four main tools in snail to meet all your deployment needs.
Audio saved to: output.wav (2355584 bytes)
```

### Chat Completions API, Unary, TEXT -> AUDIO+TEXT
```
python3 omni_ccompletion_unary_AT_T.py --prompt "List 3 biggest cities of Japan, pure text no signs or special letters." --voice "br_f019"
```
```
Text: Tokyo Osaka Nagoya
Audio saved to: output.wav (228224 bytes)
Transcript: Tokyo Osaka Nagoya
```

### Responses API, Stream, TEXT -> AUDIO+TEXT
```
python3 omni_responses_unary_AT_T.py --prompt "In  20 words, what is OpenVINO?" --voice m02
```

```
Prompt: In  20 words, what is OpenVINO?
Voice: m02
Streaming from http://localhost:11338/v3/responses ...

--- Text ---
OpenVINO is an open-source toolkit for deploying high-performance AI solutions across cloud, edge, and devices.Playing audio...


--- Audio streaming ---
Audio stream complete.

--- Performance ---
Audio duration: 5.18 s (124215 samples)
Wall-clock time: 9.02 s
Real-time factor: 1.74x (1.0 = real-time, lower = faster)
```