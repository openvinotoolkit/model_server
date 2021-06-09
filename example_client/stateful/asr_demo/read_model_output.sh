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
export KALDI_PATH=/opt/kaldi
export ASPIRE_PATH=$KALDI_PATH/egs/aspire/s5
export DATA_PATH=/tmp

cd $ASPIRE_PATH/exp/tdnn_7b_chain_online/graph_pp

$KALDI_PATH/src/bin/latgen-faster-mapped --max-active=7000 --max-mem=50000000 --beam=13 --lattice-beam=6.0 --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=words.txt $ASPIRE_PATH/exp/chain/tdnn_7b/final.mdl HCLG.fst ark:$DATA_PATH/scores.ark ark:-| $KALDI_PATH/src/latbin/lattice-scale --inv-acoustic-scale=13 ark:- ark:- | $KALDI_PATH/src/latbin/lattice-best-path --word-symbol-table=words.txt ark:- ark,t:-  > $DATA_PATH/out.txt
cat $DATA_PATH/out.txt | $ASPIRE_PATH/utils/int2sym.pl -f 2- words.txt | tee $1.txt
