#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Variables setup
KALDI_PATH=/opt/kaldi
ASPIRE_PATH=$KALDI_PATH/egs/aspire/s5
DATA_PATH=/tmp

# WAV preparation
ffmpeg -y -i $1  -acodec pcm_s16le -ac 1 -ar 8000 $DATA_PATH/sample.wav

# Prepare required files
mkdir $ASPIRE_PATH/data/conversion
cd $ASPIRE_PATH/data/conversion
echo "$1 $DATA_PATH/sample.wav" > wav.scp
echo "$1 $1" > utt2spk
cd $ASPIRE_PATH
utils/utt2spk_to_spk2utt.pl data/conversion/utt2spk > data/conversion/spk2utt
utils/copy_data_dir.sh data/conversion data/conversion_hires

train_cmd="run.pl"
decode_cmd="run.pl --mem 2G"

# Make MFCC features
steps/make_mfcc.sh --nj 1 --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" data/conversion_hires
$KALDI_PATH/src/featbin/compute-mfcc-feats --config=$ASPIRE_PATH/conf/mfcc_hires.conf scp:$ASPIRE_PATH/data/conversion/wav.scp ark,scp:$DATA_PATH/feats.ark,$DATA_PATH/feats.scp

# Extract ivectors
nspk=$(wc -l <data/conversion_hires/spk2utt)
steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" --ivector_period 5000 data/conversion_hires exp/nnet3/extractor exp/nnet3/ivectors_conversion_hires

# Replicate ivectors <number of frames> times to fulfil OpenVINO requirement
cd $ASPIRE_PATH/exp/nnet3/ivectors_conversion_hires
$KALDI_PATH/src/featbin/copy-feats --binary=False ark:ivector_online.1.ark ark,t:ivector_online.1.ark.txt

cp /opt/model_server/example_client/stateful/asr_demo/expand_ivectors.py .
python3 expand_ivectors.py

$KALDI_PATH/src/featbin/copy-feats --binary=True ark,t:ivector_online_ie.ark.txt ark:ivector_online_ie.ark

cp ivector_online_ie.ark $DATA_PATH/ivectors.ark
