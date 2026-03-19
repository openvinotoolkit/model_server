#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Inpainting / outpainting demo using the OpenAI image edits API against OVMS.

Usage:
    python inpainting_client.py \
        --image  C:/git/genai_checks/outputs/v2_01_cat_bench_gpu.png \
        --mask   C:/git/genai_checks/outputs/v2_01_cat_bench_gpu_mask.png \
        --prompt "a dalmatian dog sitting on a bench, bright pink spots on white fur" \
        --output result_inpainting.png \
        --host   http://localhost:9992

Requirements:
    pip install openai pillow
"""

import argparse
import base64
import io
import sys

from openai import OpenAI
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OVMS inpainting demo via OpenAI API")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the source image (PNG/JPG).",
    )
    parser.add_argument(
        "--mask",
        required=True,
        help="Path to the mask image. White pixels = area to repaint, black = keep.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "a dalmatian dog sitting on a bench, "
            "bright neon pink spots on white fur, "
            "photorealistic, high detail"
        ),
        help="Text prompt describing the desired result in the masked region.",
    )
    parser.add_argument(
        "--output",
        default="result_inpainting.png",
        help="Where to save the output image.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:9992",
        help="OVMS REST base URL (default: http://localhost:9992).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Denoising strength [0.0, 1.0]. Higher = more creative, lower = closer to source.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (overrides server default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = OpenAI(
        base_url=f"{args.host}/v3",
        api_key="unused",  # OVMS does not require an API key
    )

    print(f"Source image : {args.image}")
    print(f"Mask image   : {args.mask}")
    print(f"Prompt       : {args.prompt}")
    print(f"Server       : {args.host}")

    open_kwargs: dict = {}
    if args.strength is not None:
        # strength is passed as extra_body because the standard openai client
        # does not expose it; OVMS accepts it as a multipart field.
        open_kwargs["extra_body"] = {}
        if args.strength is not None:
            open_kwargs["extra_body"]["strength"] = str(args.strength)
        if args.steps is not None:
            open_kwargs["extra_body"]["num_inference_steps"] = str(args.steps)

    with open(args.image, "rb") as img_file, open(args.mask, "rb") as mask_file:
        print("Sending inpainting request …")
        response = client.images.edit(
            model="image_generation",  # matches the servable name in OVMS
            image=img_file,
            mask=mask_file,
            prompt=args.prompt,
            response_format="b64_json",
            **open_kwargs,
        )

    # Decode the base64 PNG returned by OVMS
    b64_data = response.data[0].b64_json
    if b64_data is None:
        print("ERROR: No image data in response.", file=sys.stderr)
        sys.exit(1)

    image_bytes = base64.b64decode(b64_data)
    result_image = Image.open(io.BytesIO(image_bytes))
    result_image.save(args.output)
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
