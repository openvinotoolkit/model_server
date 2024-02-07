import tritonclient.grpc as grpcclient
from tritonclient.utils import serialize_byte_tensor
import numpy as np

def serialize_prompts(prompts):
    infer_input = grpcclient.InferInput("pre_prompt", [len(prompts)], "BYTES")
    if len(prompts) == 1:
        # Single batch serialized directly as bytes
        infer_input._raw_content = prompts[0].encode()
        return infer_input
    # Multi batch serialized in tritonclient 4byte len format
    infer_input._raw_content = serialize_byte_tensor(
        np.array(prompts, dtype=np.object_)).item()
    return infer_input
