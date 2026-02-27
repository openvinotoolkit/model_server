#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import torch
import json
import time
from pathlib import Path
from kokoro.model import KModel
from kokoro import KPipeline
import openvino as ov
import shutil

MAX_SEQ_LENGTH = 500


class KokoroTTSPipeline:
    def __init__(self):
        model_id = "hexgrad/Kokoro-82M"
        self.pipeline = KPipeline(lang_code="a", repo_id=model_id)

    def __call__(self, text: str, voice: str = "af_heart"):
        with torch.no_grad():
            generator = self.pipeline(text, voice=voice)
            result = next(generator)
        return result.audio


class OVKModel(KModel):
    def __init__(self, model_dir: Path, device: str, plugin_config: dict = {}):
        torch.nn.Module.__init__(self)

        core = ov.Core()

        self.repo_id = model_id
        with (model_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        self.vocab = config["vocab"]
        print("Starting to compile OpenVINO model on device:", device)

        start = time.time()
        self.model = core.compile_model(model_dir / "openvino_model.xml", device.upper(), config=plugin_config)
        print(f"Model compiled successfully in {time.time() - start:.2f}s.")
        self.context_length = config["plbert"]["max_position_embeddings"]

    @property
    def device(self):
        return torch.device("cpu")

    def forward_with_tokens(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1) -> tuple[torch.FloatTensor, torch.LongTensor]:
        text_len = input_ids.shape[-1]

        if text_len < MAX_SEQ_LENGTH:
            # 0 in this model context is acting as BOS/EOS/PAD.
            # Since 0 causes artifacts, we might consider space (16) or period (4).
            padding_value = 16
            input_ids = torch.nn.functional.pad(input_ids, (0, MAX_SEQ_LENGTH - text_len), value=padding_value)

        start = time.time()
        print("Running inference on OpenVINO model...")
        outputs = self.model([input_ids, ref_s, torch.tensor(speed)])
        print(f"Inference completed in {time.time() - start:.2f}s.")

        audio = torch.from_numpy(outputs[0])
        pred_dur = torch.from_numpy(outputs[1])

        if text_len < MAX_SEQ_LENGTH:
            pred_dur = pred_dur[:text_len]
            # Approximate audio trimming based on duration ratio
            total_dur = outputs[1].sum()
            valid_dur = pred_dur.sum()
            if total_dur > 0:
                audio_keep = int(audio.shape[-1] * (valid_dur / total_dur))
                audio = audio[:audio_keep]

        return audio, pred_dur

    @staticmethod
    def download_and_convert(model_dir: Path, repo_id: str, ttsPipeline: KokoroTTSPipeline):
        import openvino as ov
        from huggingface_hub import hf_hub_download
        import gc

        if not (model_dir / "openvino_model.xml").exists():
            print(f"Converting Kokoro model to OpenVINO format at {model_dir}...")
            model = ttsPipeline.pipeline.model
            model.forward = model.forward_with_tokens
            input_ids = torch.randint(1, 100, (48,)).numpy()
            input_ids = torch.LongTensor([[0, *input_ids, 0]])
            style = torch.randn(1, 256)
            speed = torch.randint(1, 10, (1,), dtype=torch.float32)

            ov_model = ov.convert_model(model, example_input=(input_ids, style, speed), input=[
                                        ov.PartialShape("[1, 2..]"), ov.PartialShape([1, -1])])
            ov.save_model(ov_model, model_dir / "openvino_model.xml")
            hf_hub_download(repo_id=model_id, filename="config.json", local_dir=model_dir)
        else:
            print(f"OpenVINO model already exists at {model_dir}, skipping conversion.")

        gc.collect()

    @staticmethod
    def convert_to_static(input_model_dir: Path, output_model_dir: Path):
        import openvino as ov

        print(f"Converting OpenVINO model to static shapes at {input_model_dir}...")
        core = ov.Core()
        model = core.read_model(input_model_dir / "openvino_model.xml")
        static_shape = {"input_ids": [1, MAX_SEQ_LENGTH], "ref_s": [1, 256], "speed": [1], }
        model.reshape(static_shape)
        print("Reshaped model inputs:", model.inputs)
        ov.save_model(model, output_model_dir / "openvino_model.xml")
        print("Conversion to static shapes completed.")
        # Copy config file
        shutil.copy(input_model_dir / "config.json", output_model_dir / "config.json")


if __name__ == "__main__":

    model_id = "hexgrad/Kokoro-82M"

    # Download model from Hugging Face and convert to OpenVINO format.
    pipeline = KokoroTTSPipeline()

    # Convert and save the Kokoro model to OpenVINO format
    OVKModel.download_and_convert(Path("./kokoro_openvino_model"), repo_id=model_id, ttsPipeline=pipeline)

    # To run inference on NPU, model must have static input shapes
    OVKModel.convert_to_static(Path("./kokoro_openvino_model"), Path("./kokoro_static_openvino_model"))
    # # Execution on NPU require config file
    # config = {
    #     "NPU": {
    #         "NPU_USE_NPUW": "YES",
    #         "NPUW_DEVICES": "NPU,CPU",
    #         "NPUW_KOKORO": "YES",
    #     }
    # }

    # # NPUW_CACHE_DIR can be used to avoid compilation on every run
    # config["NPU"]["NPUW_CACHE_DIR"] = "./npu_cache_kokoro"