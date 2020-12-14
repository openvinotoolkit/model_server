#
# Copyright (c) 2018-2020 Intel Corporation
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

import grpc
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray, expand_dims
import classes
import datetime
import argparse
import math
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import print_statistics, prepare_certs
from kaldi_python_io import ArchiveReader

def printDebug(msg):
    global debug_mode
    if debug_mode:
        print(msg)

def CalculateUtteranceError(referenceArray, resultArray):
        
    printDebug("OUTPUT: {} \n".format(resultArray))
    printDebug("OUTPUT SHAPE: {} \n".format(resultArray.shape))
    printDebug("REF SHAPE: {} \n".format(referenceArray.shape))
    printDebug("REF: {} \n".format(referenceArray))
    errorSum = 0.0

    rootMeanErr = math.sqrt((np.square(resultArray - referenceArray)).mean(axis=None))
    maxRef = np.amax(referenceArray)
    maxOut = np.amax(resultArray)
    printDebug("OUTPUT MAX: {} \n".format(maxOut)) 
    printDebug("REF MAX: {} \n".format(maxRef)) 

    return rootMeanErr

delimiter = ","

# Example commands:
# RM_LSTM4F
# grpc_stateful_client.py --model_input_path rm_lstm4f/test_feat_1_10.ark --model_score_path rm_lstm4f/test_score_1_10.ark 
#     --grpc_address localhost --grpc_port 9000 --input_name Parameter --output_name affinetransform/Fused_Add_
#     --model_name rm_lstm4f --utterances 0 --samples 0 --debug

# ASPIRE_TDNN
# grpc_stateful_client.py --model_input_path aspire_tdnn/mini_feat_1_10.ark,aspire_tdnn/mini_feat_1_10_ivector.ark
#     --model_score_path aspire_tdnn/aspire_tdnn_mini_feat_1_10_kaldi_score.ark 
#     --grpc_address localhost --grpc_port 9000 --input_name input,ivector --output_name Final_affine
#     --model_name aspire_tdnn --utterances 0 --samples 0 --cw_l 17 --cw_r 12 --debug

parser = argparse.ArgumentParser(description='Sends requests via TFS gRPC API using data in stateful model ark input file. '
                                             'It displays performance statistics and optionally')
parser.add_argument('--model_input_path', required=False, default='rm_lstm4f/test_feat_1_10.ark', help='Path to input ark file')
parser.add_argument('--model_score_path', required=False, default='rm_lstm4f/test_score_1_10.ark', help='Path to input ark file')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='Parameter', help='Specify input tensor name. default: Parameter')
parser.add_argument('--output_name',required=False, default='affinetransform/Fused_Add_',
                    help='Specify output name. default: affinetransform/Fused_Add_')
parser.add_argument('--model_name', default='rm_lstm4f', help='Define model name, must be same as is in service. default: rm_lstm4f',
                    dest='model_name')

parser.add_argument('--utterances', required=False, default=0, help='How many utterances to process from ark file. default 0 meaning no limit')
parser.add_argument('--samples', required=False, default=0, help='How many samples to process from each utterance file. default 0 meaning no limit')
parser.add_argument('--debug', required=False, default=0, help='Enabling debug prints. default 0')
parser.add_argument('--cw_l', required=False, default=0, help='Left model input context window padding. default 0')
parser.add_argument('--cw_r', required=False, default=0, help='Right model input context window padding. default 0')

print('### Starting grpc_stateful_client.py client processing ###')

args = vars(parser.parse_args())

global debug_mode
debug_mode = args.get('debug')
channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

processing_times = np.zeros((0),int)

cw_l = args.get('cw_l')
cw_r = args.get('cw_l')

ark_readers = []
ark_scores = []
input_names = []
output_names = []

model_input_paths = args.get('model_input_path').split(delimiter)
for model_input_path in model_input_paths:
    print('Reading input ark file {}'.format(model_input_path))
    ark_reader = ArchiveReader(model_input_path)
    ark_readers.append(ark_reader)

model_score_paths = args.get('model_score_path').split(delimiter)
for model_score_path in model_score_paths:
    print('Reading scores ark file {}'.format(model_score_path))
    ark_score = ArchiveReader(model_score_path)
    ark_scores.append(ark_score)

inputs = args['input_name'].split(delimiter)
for input in inputs:
    print('Adding input name {}'.format(input))
    input_names.append(input)

outputs = args['output_name'].split(delimiter)
for output in outputs:
    print('Adding output name {}'.format(output))
    output_names.append(output)

print('Start processing:')
print('\tModel name: {}'.format(args.get('model_name')))

SEQUENCE_START = 1
SEQUENCE_END = 2
sequence_id = 2
utterances_limit = int(args.get('utterances'))
samples_limit = int(args.get('samples'))
print('\tUtterances limit: {}'.format(utterances_limit))
print('\tSamples limit: {}'.format(samples_limit))

numberOfKeys = 0

for key, obj in ark_reader:
    printDebug("Input ark file data range {0}: {1}".format(key, obj.shape))
    numberOfKeys += 1

for key, obj in ark_score:
    printDebug("Scores ark file data range {0}: {1}".format(key, obj.shape))

utterance = 0
meanErrGlobal = 0.0

# First ark file is the one we will iterate through for all input ark files
ark_reader = ark_readers[0]

score_index = (cw_l + cw_r) * -1
for key, obj in ark_reader:
    utterance += 1
    if utterances_limit > 0 and utterance > utterances_limit:
        break

    scoreObjects = { k:m for k,m in ark_score }

    batch_size = obj.shape[0]
    print('\n\tInput name: {}'.format(key))
    printDebug('\tInput in shape: {}'.format(obj.shape))
    printDebug('\tInput batch size: {}'.format(batch_size))
    printDebug('\tSequence id: {}'.format(sequence_id))

    meanErrSum = 0.0

    for x in range(0, batch_size + cw_l + cw_r):
        if samples_limit > 0 and x >= samples_limit:
            break

        printDebug('\tExecution: {}\n'.format(x))
        request = predict_pb2.PredictRequest()
        request.model_spec.name = args.get('model_name')

        #Add input data
        input_index = x
        # Input for context window padding
        if x < cw_l:
            # Left padding
            input_index = 0
        elif x >= cw_l and x < batch_size + cw_l:
            # Middle
            input_index = x-cw_l
        else:
            # Right padding
            input_index = batch_size - 1

        # Standard input
        name_index = 0
        for ark_reader in ark_readers:
            input_name = input_names[name_index]
            for nameKey, nameObj in ark_reader:
                if nameKey == key:
                    printDebug('\tTensor before input in shape: {}\n'.format(nameObj[input_index].shape))
                    inputArray = np.expand_dims(nameObj[input_index], axis=0)
                    printDebug('\tTensor input in shape: {}\n'.format(inputArray.shape))
                    request.inputs[input_name].CopyFrom(make_tensor_proto(inputArray, shape=inputArray.shape))
                    break
            name_index += 1

        if x == 0:
            request.inputs['sequence_control_input'].CopyFrom(make_tensor_proto(SEQUENCE_START, dtype="uint32"))
        
        request.inputs['sequence_id'].CopyFrom(make_tensor_proto(sequence_id, dtype="uint64"))

        if x == batch_size + cw_l + cw_r - 1 or x == samples_limit - 1:
            request.inputs['sequence_control_input'].CopyFrom(make_tensor_proto(SEQUENCE_END, dtype="uint32"))

        start_time = datetime.datetime.now()
        result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
        end_time = datetime.datetime.now()

        for output_name in output_names:
            if output_name not in result.outputs:
                print("ERROR: Invalid output name", output_name)
                print("Available outputs:")
                for Y in result.outputs:
                    print(Y)
                exit(1)

        if 'sequence_id' not in result.outputs:
            print("ERROR: Missing sequence_id in model output")
            print("Available outputs:")
            for Y in result.outputs:
                print(Y)
            exit(1)

        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))

        # Reset sequence end for testing purpose
        #request.inputs['sequence_control_input'].CopyFrom(make_tensor_proto(SEQUENCE_END, dtype="uint32"))
        #result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs

        # compare results after we are pass initial context window results
        if score_index >= 0:
            # Only one reference results array - loop can be added here if needed
            referenceArray = scoreObjects[score_index][x]

            # Parse output
            resultsArrays = dict()
            for output_name in output_names:
                resultsArrays[output_name] = make_ndarray(result.outputs[output_name])

            meanErr = CalculateUtteranceError(referenceArray, resultsArrays[output_name][0])

            meanErrSum += meanErr
            # Statistics
            printDebug('Iteration {}; Mean error: {:.10f} Processing time: {:.2f} ms; speed {:.2f} fps\n'.format(x,meanErr,round(np.average(duration), 2), round(1000 * batch_size / np.average(duration), 2)))

        score_index += 1
        #END utterance loop

    if samples_limit > 0:
        meanErrAvg = meanErrSum/min(samples_limit,batch_size)
    else:
        meanErrAvg = meanErrSum/batch_size
    print("\tSequence {} mean error: {:.10f}\n".format(sequence_id, meanErrAvg))
    sequence_id += 1
    meanErrGlobal += meanErrAvg
    #END input name loop

if utterances_limit > 0:
    meanGlobalErrAvg = meanErrGlobal/min(numberOfKeys, utterances_limit)
else:
    meanGlobalErrAvg = meanErrGlobal/numberOfKeys

print("Global mean error: {:.10f}\n".format(meanGlobalErrAvg))
print_statistics(processing_times, batch_size)
print('### Finished grpc_stateful_client.py client processing ###')
