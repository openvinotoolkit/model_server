#
# Copyright (c) 2021-2023 Intel Corporation
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

import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS

import numpy as np

from tokens_bert import text_to_tokens, load_vocab_file, Token
from html_reader import get_paragraphs
import numpy as np
import ovmsclient

class ConcatenatedParagraph():
    def __init__(self, text="", tokens=[]):
        self.text = text
        self.tokens = tokens
        self.tokens_length = len(tokens)

    def append_paragraph(self, text, tokens):
        self.append_tokens(tokens)
        self.append_text(text)

    def append_text(self, text):
        self.text += text

    def append_tokens(self, tokens):
        self.tokens += self.append_offset(tokens, len(self.text), len(self.text))
        self.tokens_length = len(self.tokens)

    def append_offset(self, tokens, s_offset, e_offset):
        return list(map(lambda token: Token(token.id, token.start + s_offset, token.end + e_offset), tokens))
    
    def get_tokens_id(self):
        return [token.id for token in self.tokens]

    def get_token(self, token_index):
        return self.tokens[token_index]


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-v", "--vocab", help="Required. path to the vocabulary file with tokens",
                      required=True, type=str)
    args.add_argument("-i", "--input_url", help="Required. URL to a page with context",
                      action='append',
                      required=True, type=str)
    args.add_argument("-q", "--question", help="Question about the input text",
                      action='append',
                      required=True, type=str)
    args.add_argument("--input_names",
                      help="Optional. Inputs names for the network. "
                           "Default values are \"input_ids,attention_mask,token_type_ids\" ",
                      required=False, type=str, default="input_ids,attention_mask,token_type_ids")
    args.add_argument("--output_names",
                      help="Optional. Outputs names for the network. "
                           "Default values are \"output_s,output_e\" ",
                      required=False, type=str, default="output_s,output_e")
    args.add_argument("--model_squad_ver", help="Optional. SQUAD version used for model fine tuning",
                      default="1.2", required=False, type=str)
    args.add_argument("--max_question_token_num", help="Optional. Maximum number of tokens in question",
                      default=8, required=False, type=int)
    args.add_argument("--max_answer_token_num", help="Optional. Maximum number of tokens in answer",
                      default=15, required=False, type=int)
    args.add_argument("--min_request_token_num", help="Optional. Minimum number of tokens in paragraph",
                      default=300, required=False, type=int)
    args.add_argument('-c', '--colors', action='store_true',
                      help="Optional. Nice coloring of the questions/answers. "
                           "Might not work on some terminals (like Windows* cmd console)")
    args.add_argument('--grpc_address',required=False, default='localhost',
                      help='Specify url to grpc service. default:localhost')
    args.add_argument('--grpc_port',required=False, default=9000,
                      help='Specify port to grpc service. default: 9000')
    args.add_argument('--model_name', default='bert', help='Define model name, must be same as is in ovms service. default: bert',
                    dest='model_name'),
    args.add_argument('--loop', action='store_true', help='Set true to loop the questions')
    return parser

def main():

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # create grpc connection
    client = ovmsclient.make_grpc_client("{}:{}".format(args.grpc_address,args.grpc_port))

    if args.colors:
        COLOR_RED = "\033[91m"
        COLOR_RESET = "\033[0m"
    else:
        COLOR_RED = ""
        COLOR_RESET = ""

    # check input and output names
    input_names = list(i.strip() for i in args.input_names.split(','))
    output_names = list(o.strip() for o in args.output_names.split(','))

    # load vocabulary file for model
    log.info("Loading vocab file:\t{}".format(args.vocab))
    vocab = load_vocab_file(args.vocab)
    log.info("{} tokens loaded".format(len(vocab)))

    # get context as a string for answers visualization
    paragraphs = get_paragraphs(args.input_url)
    context = '\n'.join(paragraphs)
    log.info("Size: {} chars".format(len(context)))
    log.info("Context: " + COLOR_RED + context + COLOR_RESET)

    loop = True
    print("arg", args.loop)
    # loop on user's questions
    while loop:
        question = " "
        question = question.join(args.question)
        print("question:", question)
        if not question:
            break

        q_tokens = text_to_tokens(question.lower(), vocab)
        q_tokens_length = len(q_tokens)

        t0 = time.perf_counter()
        t_count = 0

        # array of ConcatenatedParagraphs(size >= args.min_request_token_num)
        concatenated_paragraphs = []
        # number of current concatenated paragraph
        cp_number = 0
        # array of answers from each concatenated paragraph
        answers = []

        # iterate through paragraphs
        for i in range(len(paragraphs)):
            p_tokens = text_to_tokens(paragraphs[i].lower(), vocab)
            if len(p_tokens) == 0: continue
            # concatenate paragraphs
            if len(concatenated_paragraphs) == cp_number:
                concatenated_paragraphs.append(ConcatenatedParagraph(paragraphs[i], p_tokens))
            else:
                concatenated_paragraphs[cp_number].append_paragraph(paragraphs[i], p_tokens)
            
            cur_cp = concatenated_paragraphs[cp_number]

            if cur_cp.tokens_length < args.min_request_token_num:
                if i != len(paragraphs) - 1:
                    continue

            # form the request
            tok_cls = vocab['[CLS]']
            tok_sep = vocab['[SEP]']
            input_ids = [tok_cls] + [token.id for token in q_tokens] + [tok_sep] + cur_cp.get_tokens_id() + [tok_sep]
            input_ids_length = len(input_ids)
            token_type_ids = [0] + [0] * q_tokens_length + [0] + [1] * cur_cp.tokens_length + [0]
            attention_mask = [1] * input_ids_length

            # create numpy inputs for IE
            inputs = {
                input_names[0]: np.array([input_ids], dtype=np.int64),
                input_names[1]: np.array([attention_mask], dtype=np.int64),
                input_names[2]: np.array([token_type_ids], dtype=np.int64),
            }
            if len(input_names)>3:
                inputs[input_names[3]] = np.arange(input_ids_length, dtype=np.int64)[None,:]

            t_start = time.perf_counter()
            res = client.predict(inputs, args.model_name, timeout=10.0)
            t_end = time.perf_counter()

            t_count += 1
            log.info("Sequence of length {} is processed with {:0.2f} requests/sec ({:0.2} sec per request)".format(
                cur_cp.tokens_length,
                1 / (t_end - t_start),
                t_end - t_start
            ))

            # get start-end scores for context
            def get_score(name):
                out = np.exp(res[name].reshape((input_ids_length,)))
                return out / out.sum(axis=-1)

            score_s = get_score(output_names[0])
            score_e = get_score(output_names[1])

            # get 'no-answer' score (not valid if model has been fine-tuned on squad1.x)
            if args.model_squad_ver.split('.')[0] == '1':
                score_na = 0
            else:
                score_na = score_s[0] * score_e[0]

            # find product of all start-end combinations to find the best one
            c_s_idx = q_tokens_length + 2  # index of first context token in tensor
            c_e_idx = input_ids_length - 1  # index of last+1 context token in tensor
            score_mat = np.matmul(
                score_s[c_s_idx:c_e_idx].reshape((cur_cp.tokens_length, 1)),
                score_e[c_s_idx:c_e_idx].reshape((1, cur_cp.tokens_length))
            )
            # reset candidates with end before start
            score_mat = np.triu(score_mat)
            # reset long candidates (>max_answer_token_num)
            score_mat = np.tril(score_mat, args.max_answer_token_num - 1)
            # find the best start-end pair
            max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
            max_score = score_mat[max_s, max_e] * (1 - score_na)

            max_s = cur_cp.get_token(max_s).get_start_position()
            max_e = cur_cp.get_token(max_e).get_end_position()

            answers.append((max_score, max_s, max_e, cp_number))
            cp_number += 1

        t1 = time.perf_counter()
        log.info("The performance below is reported only for reference purposes, "
                 "please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.")
        log.info("{} requests were processed in {:0.2f}sec ({:0.2}sec per request)".format(
            t_count,
            t1 - t0,
            (t1 - t0) / t_count
        ))

        # print top 3 results
        answers = sorted(answers, key=lambda x: -x[0])
        for score, s, e, par_number in answers[:3]:
            paragraph = concatenated_paragraphs[par_number]
            log.info("---answer: {:0.2f} {}".format(score, paragraph.text[s:e]))
            log.info("   " + paragraph.text[:s] + COLOR_RED + paragraph.text[s:e] + COLOR_RESET + paragraph.text[e:])
        if args.loop == False:
            loop = False

if __name__ == '__main__':
    sys.exit(main() or 0)
