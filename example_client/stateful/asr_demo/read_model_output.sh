#!/bin/bash

# Variables setup
export KALDI_PATH=/opt/kaldi
export ASPIRE_PATH=$KALDI_PATH/egs/aspire/s5
export DATA_PATH=/opt/data

cd $ASPIRE_PATH/exp/tdnn_7b_chain_online/graph_pp

$KALDI_PATH/src/bin/latgen-faster-mapped --max-active=7000 --max-mem=50000000 --beam=13 --lattice-beam=6.0 --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=words.txt ../final.mdl HCLG.fst ark:/opt/data/scores.ark ark:-| $KALDI_PATH/src/latbin/lattice-scale --inv-acoustic-scale=13 ark:- ark:- | $KALDI_PATH/src/latbin/lattice-best-path --word-symbol-table=words.txt ark:- ark,t:-  > /opt/data/out.txt
cat /opt/data/out.txt | $ASPIRE_PATH/utils/int2sym.pl -f 2- words.txt | tee /opt/data/out.txt
