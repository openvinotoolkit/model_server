#!/bin/bash

# Variables setup
KALDI_PATH=/opt/kaldi
ASPIRE_PATH=$KALDI_PATH/egs/aspire/s5
DATA_PATH=/opt/data

# WAV download and preparation
cd $DATA_PATH
#wget https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0039_8k.wav
#sox OSR_us_000_0039_8k.wav harvard_sample_01.wav trim 11.8 3.9
#mv harvard_sample_01.wav recording.wav
ffmpeg -i recording.wav  -acodec pcm_s16le -ac 1 -ar 8000 sample.wav

# Prepare required files
mkdir $ASPIRE_PATH/data/conversion
cd $ASPIRE_PATH/data/conversion
echo "sample /opt/data/sample.wav" > wav.scp
echo "sample sample" > utt2spk
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

cp /ovms/example_client/stateful/expand_ivectors.py .
python3 expand_ivectors.py

$KALDI_PATH/src/featbin/copy-feats --binary=True ark,t:ivector_online_ie.ark.txt ark:ivector_online_ie.ark

cp ivector_online_ie.ark $DATA_PATH/ivectors.ark
