import openvino_genai as genai
pipe = genai.LLMPipeline('ov_model', 'CPU')
print(pipe.generate('Hello', max_new_tokens=1000, ignore_eos=True, do_sample=False))

pipe = genai.VLMPipeline('vlm_ov_model', 'CPU')
print(pipe.generate('Hello', max_new_tokens=1000, ignore_eos=True, do_sample=False))

