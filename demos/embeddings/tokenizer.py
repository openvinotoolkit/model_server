import numpy as np
from transformers import AutoTokenizer, AutoModel
hf_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
query = "what is it?"
passages = [{"text":"one text"},{"text":"two text"},{"text":"three text which is longer than the ones before"}]
query_passage_pairs = [[query, passage["text"]] for passage in passages]
print(query_passage_pairs)
input_tensors = hf_tokenizer(query_passage_pairs, padding=True, truncation=True, return_tensors="pt")
print(input_tensors)
text = hf_tokenizer.decode([    0,  2367,    83,   442,    32,     2,     2,  6626,  7986,     2])
print(text)
print(hf_tokenizer)


from openvino import compile_model, convert_model
from openvino_tokenizers import convert_tokenizer, connect_models
import openvino as ov

core = ov.Core()
# embedding model
tokenizer_model = core.read_model(model="bge-reranker-large_tokenizer_ovms/1/openvino_tokenizer.xml")
reranker_model = core.read_model(model="bge-reranker-large_reranker_ovms/1/openvino_model.xml")

# Get input information
print("## Inputs:")
for input in tokenizer_model.inputs:
    print(f"Name: {input.get_any_name()}")
    print(f"Shape: {input.partial_shape}")
    print(f"Precision: {input.element_type}")
    print()

# Get output information 
print("## Outputs:")
for output in tokenizer_model.outputs:
    print(f"Name: {output.get_any_name()}")
    print(f"Shape: {output.partial_shape}")
    print(f"Precision: {output.element_type}")
    print()

tokenizer_ov_model = compile_model(tokenizer_model)
rerank_ov_model = compile_model(reranker_model)

print(hf_tokenizer('what is it?'))
print(hf_tokenizer('one text'))
print(hf_tokenizer(query_passage_pairs))

"""
DIFFERENT BATCHES HAVE DIFFERENT LENGTHS, TOKENIZER RESULT IS NOT A TENSOR
{'input_ids': [0, 2367, 83, 442, 32, 2], 'attention_mask': [1, 1, 1, 1, 1, 1]}
{'input_ids': [0, 1632, 7986, 2], 'attention_mask': [1, 1, 1, 1]}
{'input_ids': [[0, 2367, 83, 442, 32, 2, 2, 1632, 7986, 2], [0, 2367, 83, 442, 32, 2, 2, 6626, 7986, 2], [0, 2367, 83, 442, 32, 2, 2, 17262, 7986, 3129, 83, 51713, 3501, 70, 64333, 8108, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}


XLMRobertaTokenizerFast(name_or_path='BAAI/bge-reranker-large', vocab_size=250002, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
        0: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        250001: AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True, special=True),
"""

def use_ov_model_for_query_pairs(query_pairs, ov_tokenizer):

    # 1. Prepare strings
    # <s> what is it?</s></s> two text</s>
    # <s> = hf_tokenizer.bos_token # TODO: How to take from ov_tokenizer?
    # </s> = hf_tokenizer.eos_token # TODO: How to take from ov_tokenizer?
    input_strings = []
    for pair in query_pairs:
        input_string = pair[0] + hf_tokenizer.eos_token + hf_tokenizer.eos_token + pair[1]
        input_strings.append(input_string)

    print(input_strings)
    print(ov_tokenizer(input_strings))

    return ov_tokenizer(input_strings)

print(hf_tokenizer.bos_token)  # TODO: How to retrieve it from OV model?
print(hf_tokenizer.eos_token)  # TODO: How to retrieve it from OV model?

print(hf_tokenizer([
            ["hello world", "one text abc longer longer longer longer longer text"],
            ["hello world", "one text"]
        ], padding=True, truncation=True, return_tensors="pt"))

padded_in = hf_tokenizer([
            ["hello world", "one text abc longer longer longer longer longer text"],
            ["hello world", "one text"]
        ], padding=True, truncation=True, return_tensors="pt")

"""
---------------------- when padding not used -----------
    return Tensor(value)
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
---------------------- when padding is used --------
OK

            ["hello world", "one text abc longer longer longer longer longer text"],
            ["hello world", "one text"]

{'input_ids': tensor([
    [ 0, 33600,    31,  8999,     2,     2,  1632,  7986,  1563,   238, 51713, 51713, 51713, 51713, 51713,  7986,     2],
    [ 0, 33600,    31,  8999,     2,     2,  1632,  7986,     2,     1, 1,     1,     1,     1,     1,     1,     1]]), 
 'attention_mask': tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])}

"""

print('------ ov output when used HF tokenizer -----')
ov_output = rerank_ov_model(
        {'input_ids':       padded_in.input_ids,
         'attention_mask':  padded_in.attention_mask}
    )

print(ov_output)
print(ov_output['last_hidden_state'].shape)

print('===================================')
ov_generated_tokens = use_ov_model_for_query_pairs([
            ["hello world", "one text abc longer longer longer longer longer text"],
            ["hello world", "one text"]
        ], tokenizer_ov_model)
print(ov_generated_tokens)

print('------ ov output when used OV tokenizer + eos -----')
ov_output = rerank_ov_model(
        ov_generated_tokens
    )

print(ov_output)
print(ov_output['last_hidden_state'].shape)


"""
['<s>hello world</s></s>one text abc longer longer longer longer longer text</s>',
 '<s>hello world</s></s>one text</s>']
{<ConstOutput: names[input_ids] shape[?,..512] type: i64>: array([
[    0,     6,     0,   127, 13817,  8999,     2,     2,  3630,
         7986,  1563,   238, 51713, 51713, 51713, 51713, 51713,  7986,
            2,     2],
[    0,     6,     0,   127, 13817,  8999,     2,     2,  3630,
         7986,     2,     2,     1,     1,     1,     1,     1,     1,
            1,     1]]), <ConstOutput: names[attention_mask] shape[?,..512] type: i64>: array([
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])}
"""

