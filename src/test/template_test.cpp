#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main() {
    using chat_t = std::vector<std::unordered_map<std::string, std::string>>;

    chat_t chat1{{{"role", "user"}, {"content", "hello"}}};
    chat_t chat2{{{"role", "system"}, {"content", "You are assistant."}, {"role", "user"}, {"content", "hello"}}};
    chat_t chat3{{{"role", "system"}, {"content", "You are assistant."}, {"role", "user"}, {"content", "hello"},
        {"role", "assistant"}, {"content", "how can I help you?"}, {"role", "user"}, {"content", "how much is 2+2?"}}};
    /*
curl http://ov-spr-19.sclab.intel.com:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"meta-llama/Llama-2-7b-chat-hf","messages":[{"role":"user","content":"hello"}], "max_tokens":30}'
curl http://ov-spr-19.sclab.intel.com:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"meta-llama/Llama-2-7b-chat-hf","messages":[{"role":"system","content":"You are assistant."},{"role":"user","content":"hello"}], "max_tokens":30}'
curl http://ov-spr-19.sclab.intel.com:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"meta-llama/Llama-2-7b-chat-hf","messages":[{"role":"system","content":"You are assistant."},{"role":"user","content":"hello"},{"role":"assistant","content":"how can I help you?"}, {"role":"user","content":"how much is 2+2?"}], "max_tokens":30}'
*/

    using expected_prompt = std::string;
    using expected_tokens = std::vector<int>;
    std::unordered_map<std::string, std::vector<expected_prompt>> test_prompts;
    std::unordered_map<std::string, std::vector<expected_tokens>> test_tokens;

    ////////
    std::string model_name{"meta-llama/Llama-2-7b-chat-hf"};
    std::vector<expected_prompt> prompts{
        "<s>[INST] hello [/INST]",
        "<s>[INST] <<SYS>>\nYou are assistant.\n<</SYS>>\n\nhello [/INST]",
        "<s>[INST] <<SYS>>\nYou are assistant.\n<</SYS>>\n\nhello [/INST] how can I help you? </s><s>[INST] how much is 2+2? [/INST]"};
    std::vector<expected_tokens> tokens{
        {1, 518, 25580, 29962, 22172, 518, 29914, 25580, 29962},
        {1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 20255, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 12199, 518, 29914, 25580, 29962},
        {1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 20255, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 12199, 518, 29914, 25580, 29962, 920, 508, 306, 1371, 366, 29973, 29871, 2, 1, 518, 25580, 29962, 920, 1568, 338, 29871, 29906, 29974, 29906, 29973, 518, 29914, 25580, 29962}};
    test_prompts.insert({model_name, prompts});
    test_tokens.insert({model_name, tokens});

    ////////
    std::string model_name{"meta-llama/Meta-Llama-3-8B-Instruct"};
    std::vector<expected_prompt> prompts{
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nhow can I help you?<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhow much is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"};
    std::vector<expected_tokens> tokens{
        {128000, 128006, 882, 128007, 271, 15339, 128009, 128006, 78191, 128007, 271},
        {128000, 128006, 9125, 128007, 271, 2675, 527, 18328, 13, 128009, 128006, 882, 128007, 271, 15339, 128009, 128006, 78191, 128007, 271},
        {128000, 128006, 9125, 128007, 271, 2675, 527, 18328, 13, 128009, 128006, 882, 128007, 271, 15339, 128009, 128006, 78191, 128007, 271, 5269, 649, 358, 1520, 499, 30, 128009, 128006, 882, 128007, 271, 5269, 1790, 374, 220, 17, 10, 17, 30, 128009, 128006, 78191, 128007, 271}};
    test_prompts.insert({model_name, prompts});
    test_tokens.insert({model_name, tokens});

    ////////
    std::string model_name{"TinyLlama/TinyLlama-1.1B-Chat-v0.6"};
    std::vector<expected_prompt> prompts{
        "<|user|>\nhello</s>\n<|assistant|>\n",
        "<|system|>\nYou are assistant.</s>\n<|user|>\nhello</s>\n<|assistant|>\n",
        "<|system|>\nYou are assistant.</s>\n<|user|>\nhello</s>\n<|assistant|>\nhow can I help you?</s>\n<|user|>\nhow much is 2+2?</s>\n<|assistant|>\n"};
    std::vector<expected_tokens> tokens{
        {529, 29989, 1792, 29989, 29958, 13, 12199, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13},
        {529, 29989, 5205, 29989, 29958, 13, 3492, 526, 20255, 29889, 2, 29871, 13, 29966, 29989, 1792, 29989, 29958, 13, 12199, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13},
        {529, 29989, 5205, 29989, 29958, 13, 3492, 526, 20255, 29889, 2, 29871, 13, 29966, 29989, 1792, 29989, 29958, 13, 12199, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13, 3525, 508, 306, 1371, 366, 29973, 2, 29871, 13, 29966, 29989, 1792, 29989, 29958, 13, 3525, 1568, 338, 29871, 29906, 29974, 29906, 29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13}};
    test_prompts.insert({model_name, prompts});
    test_tokens.insert({model_name, tokens});

    ////////
    std::string model_name{"Qwen/Qwen-7B-Chat"};
    std::vector<expected_prompt> prompts{
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhow can I help you?<|im_end|>\n<|im_start|>user\nhow much is 2+2?<|im_end|>\n<|im_start|>assistant\n"};
    std::vector<expected_tokens> tokens{
        {151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198},
        {151644, 8948, 198, 2610, 525, 17847, 13, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198},
        {151644, 8948, 198, 2610, 525, 17847, 13, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198, 5158, 646, 358, 1492, 498, 30, 151645, 198, 151644, 872, 198, 5158, 1753, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198}};
    test_prompts.insert({model_name, prompts});
    test_tokens.insert({model_name, tokens});

    ////////
    std::string model_name{"mistralai/Mistral-7B-Instruct-v0.2"};
    std::vector<expected_prompt> prompts{
        "<s>[INST] hello [/INST]",
        "error",
        "error"};
    std::vector<expected_tokens> tokens{
        {1, 733, 16289, 28793, 6312, 28709, 733, 28748, 16289, 28793},
        {},
        {}};
    test_prompts.insert({model_name, prompts});
    test_tokens.insert({model_name, tokens});

    ////////
    std::string model_name{"THUDM/glm-4-9b-chat"};
    std::vector<expected_prompt> prompts{
        "[gMASK]<sop><|user|>\nhello<|assistant|>",
        "[gMASK]<sop><|system|>\nYou are assistant.<|user|>\nhello<|assistant|>",
        "[gMASK]<sop><|system|>\nYou are assistant.<|user|>\nhello<|assistant|>\nhow can I help you?<|user|>\nhow much is 2+2?<|assistant|>"};
    std::vector<expected_tokens> tokens{
        {151331, 151333, 151336, 198, 14978, 151337},
        {151331, 151333, 151335, 198, 2610, 525, 17821, 13, 151336, 198, 14978, 151337},
        {151331, 151333, 151335, 198, 2610, 525, 17821, 13, 151336, 198, 14978, 151337, 198, 5158, 646, 358, 1492, 498, 30, 151336, 198, 5158, 1753, 374, 220, 17, 10, 17, 30, 151337}};
    test_prompts.insert({model_name, prompts});
    test_tokens.insert({model_name, tokens});

    std::cout << chat1[0].at("role") << ": " << chat1[0].at("content") << std::endl;
    std::cout << prompts[0] << std::endl;
};
