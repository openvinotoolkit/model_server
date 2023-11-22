from optimum.intel.openvino import OVModelForCausalLM

MODEL_PATH = './model'
OV_CONFIG = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}

print('Downloading and converting...')
ov_model = OVModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    device='CPU',
    compile=False,
    ov_config=OV_CONFIG)

print(f'Saving to {MODEL_PATH} ...')
ov_model.save_pretrained(MODEL_PATH)
print('Done.')
