#!/bin/bash

# Variables setup
export KALDI_PATH=/opt/kaldi
export ASPIRE_PATH=$KALDI_PATH/egs/aspire/s5
export DATA_PATH=/opt/data

# Model download and unpack 
cd $ASPIRE_PATH
wget https://kaldi-asr.org/models/1/0001_aspire_chain_model.tar.gz
tar -xvf 0001_aspire_chain_model.tar.gz

# Make language graph for decoding
cd $ASPIRE_PATH
steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf data/lang_chain exp/nnet3/extractor exp/chain/tdnn_7b exp/tdnn_7b_chain_online
utils/mkgraph.sh --self-loop-scale 1.0 data/lang_pp_test exp/tdnn_7b_chain_online exp/tdnn_7b_chain_online/graph_pp