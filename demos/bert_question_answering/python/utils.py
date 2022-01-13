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
import numpy as np

from tokens_bert import text_to_tokens, load_vocab_file
from html_reader import get_paragraphs
import grpc
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# return entire sentence as start-end positions for a given answer (within the sentence).
def find_sentence_range(context, s, e):
    # find start of sentence
    for c_s in range(s, max(-1, s - 200), -1):
        if context[c_s] in "\n\.":
            c_s += 1
            break

    # find end of sentence
    for c_e in range(max(0, e - 1), min(len(context), e + 200), +1):
        if context[c_e] in "\n\.":
            break

    return c_s, c_e

def predict_bert(question):
    # create grpc connection
    channel = grpc.insecure_channel("{}:{}".format("localhost", 9000))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # load vocabulary file for model
    vocab = load_vocab_file("vocab.txt")

    # get context as a string (as we might need it's length for the sequence reshape)
    paragraphs = get_paragraphs(["https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"])
    context = '\n'.join(paragraphs)
    # encode context into token ids list
    c_tokens_id, c_tokens_se = text_to_tokens(context.lower(), vocab)

    # check input and output names
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["output_s", "output_e"]

    q_tokens_id, _ = text_to_tokens(question.lower(), vocab)

    # maximum number of tokens that can be processed by network at once
    max_length = 384

    # calculate number of tokens for context in each inference request.
    # reserve 3 positions for special tokens
    # [CLS] q_tokens [SEP] c_tokens [SEP]
    c_wnd_len = max_length - (len(q_tokens_id) + 3)

    # token num between two neighbour context windows
    # 1/2 means that context windows are overlapped by half
    c_stride = c_wnd_len // 2

    # array of answers from each window
    answers = []

    # init a window to iterate over context
    c_s, c_e = 0, min(c_wnd_len, len(c_tokens_id))

    # iterate while context window is not empty
    while c_e > c_s:
        print("c_e",c_e,"c_s",c_s)
        # form the request
        tok_cls = vocab['[CLS]']
        tok_sep = vocab['[SEP]']
        input_ids = [tok_cls] + q_tokens_id + [tok_sep] + c_tokens_id[c_s:c_e] + [tok_sep]
        token_type_ids = [0] + [0] * len(q_tokens_id) + [0] + [1] * (c_e - c_s) + [0]
        attention_mask = [1] * len(input_ids)

        # pad the rest of the request
        pad_len = max_length - len(input_ids)
        input_ids += [0] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

        # create numpy inputs for IE
        inputs = {
            input_names[0]: np.array([input_ids], dtype=np.int32),
            input_names[1]: np.array([attention_mask], dtype=np.int32),
            input_names[2]: np.array([token_type_ids], dtype=np.int32),
        }
        if len(input_names)>3:
            inputs[input_names[3]] = np.arange(len(input_ids), dtype=np.int32)[None,:]

        #print("inputs:",inputs)

        # create grpc prediction request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = "bert"
        for inp_name in inputs:
            request.inputs[inp_name].CopyFrom(make_tensor_proto(inputs[inp_name], shape=(inputs[inp_name].shape)))

        result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
        #print("\nresult:", result)
        res = {}
        for out_name in output_names:
            #   print("out_name:",out_name)
            res[out_name] = make_ndarray(result.outputs[out_name])

        # get start-end scores for context
        def get_score(name):
            out = np.exp(res[name].reshape((max_length,)))
            return out / out.sum(axis=-1)

        score_s = get_score(output_names[0])
        score_e = get_score(output_names[1])
        score_na = score_s[0] * score_e[0]

        # find product of all start-end combinations to find the best one
        c_s_idx = len(q_tokens_id) + 2  # index of first context token in tensor
        c_e_idx = max_length - (1 + pad_len)  # index of last+1 context token in tensor
        score_mat = np.matmul(
            score_s[c_s_idx:c_e_idx].reshape((c_e - c_s, 1)),
            score_e[c_s_idx:c_e_idx].reshape((1, c_e - c_s))
        )
        # reset candidates with end before start
        score_mat = np.triu(score_mat)
        # reset long candidates (>max_answer_token_num)
        score_mat = np.tril(score_mat, 14)
        # find the best start-end pair
        max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
        max_score = score_mat[max_s, max_e] * (1 - score_na)

        # convert to context text start-end index
        max_s = c_tokens_se[c_s + max_s][0]
        max_e = c_tokens_se[c_s + max_e][1]

        # check that answers list does not have duplicates (because of context windows overlapping)
        same = [i for i, a in enumerate(answers) if a[1] == max_s and a[2] == max_e]
        if same:
            assert len(same) == 1
            # update existing answer record
            a = answers[same[0]]
            answers[same[0]] = (max(max_score, a[0]), max_s, max_e)
        else:
            # add new record
            answers.append((max_score, max_s, max_e))

        # check that context window reached the end
        if c_e == len(c_tokens_id):
            break

        # move to next window position
        c_s = min(c_s + c_stride, len(c_tokens_id))
        c_e = min(c_s + c_wnd_len, len(c_tokens_id))

    # print top 3 results
    answers = sorted(answers, key=lambda x: -x[0])
    for _, s, e in answers[:3]:
        c_s, c_e = find_sentence_range(context, s, e)
        return context[c_s:s] + context[s:e] + context[e:c_e]

