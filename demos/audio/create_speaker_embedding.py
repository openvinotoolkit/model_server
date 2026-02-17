#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import sys

file = sys.argv[1]
signal, fs = torchaudio.load(file)
if signal.shape[0] > 1:
    signal = torch.mean(signal, dim=0, keepdim=True)
expected_sample_rate = 16000
if(fs != expected_sample_rate):
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=expected_sample_rate)
    signal = resampler(signal)

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
embedding = classifier.encode_batch(signal)
embedding = torch.nn.functional.normalize(embedding, dim=2)
embedding = embedding.squeeze().cpu().numpy().astype("float32")

output_file = sys.argv[2]
embedding.tofile(output_file)