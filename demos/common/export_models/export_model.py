#
# Copyright (c) 2024 Intel Corporation
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

import argparse
import os
import jinja2
import json
import shutil
import tempfile

def add_common_arguments(parser):
    parser.add_argument('--model_repository_path', required=False, default='models', help='Where the model should be exported to', dest='model_repository_path')
    parser.add_argument('--source_model', required=True, help='HF model name or path to the local folder with PyTorch or OpenVINO model', dest='source_model')
    parser.add_argument('--model_name', required=False, default=None, help='Model name that should be used in the deployment. Equal to source_model if HF model name is used', dest='model_name')
    parser.add_argument('--weight-format', default='int8', help='precision of the exported model', dest='precision')
    parser.add_argument('--config_file_path', default='config.json', help='path to the config file', dest='config_file_path')
    parser.add_argument('--overwrite_models', default=False, action='store_true', help='Overwrite the model if it already exists in the models repository', dest='overwrite_models')
    parser.add_argument('--target_device', default="CPU", help='CPU, GPU, NPU or HETERO, default is CPU', dest='target_device')
    parser.add_argument('--ov_cache_dir', default=None, help='Folder path for compilation cache to speedup initialization time', dest='ov_cache_dir')
    parser.add_argument('--extra_quantization_params', required=False, help='Add advanced quantization parameters. Check optimum-intel documentation. Example: "--sym --group-size -1 --ratio 1.0 --awq --scale-estimation --dataset wikitext2"', dest='extra_quantization_params')

parser = argparse.ArgumentParser(description='Export Hugging face models to OVMS models repository including all configuration for deployments')

subparsers = parser.add_subparsers(help='subcommand help', required=True, dest='task')
parser_text = subparsers.add_parser('text_generation', help='export model for chat and completion endpoints')
add_common_arguments(parser_text)
parser_text.add_argument('--pipeline_type', default=None, choices=["LM", "LM_CB", "VLM", "VLM_CB", "AUTO"], help='Type of the pipeline to be used. AUTO is used by default', dest='pipeline_type')
parser_text.add_argument('--kv_cache_precision', default=None, choices=["u8"], help='u8 or empty (model default). Reduced kv cache precision to u8 lowers the cache size consumption.', dest='kv_cache_precision')
parser_text.add_argument('--enable_prefix_caching', action='store_true', help='This algorithm is used to cache the prompt tokens.', dest='enable_prefix_caching')
parser_text.add_argument('--disable_dynamic_split_fuse', action='store_false', help='The maximum number of tokens that can be batched together.', dest='dynamic_split_fuse')
parser_text.add_argument('--max_num_batched_tokens', default=None, help='empty or integer. The maximum number of tokens that can be batched together.', dest='max_num_batched_tokens')
parser_text.add_argument('--max_num_seqs', default=None, help='256 by default. The maximum number of sequences that can be processed together.', dest='max_num_seqs')
parser_text.add_argument('--cache_size', default=10, type=int, help='KV cache size in GB', dest='cache_size')
parser_text.add_argument('--draft_source_model', required=False, default=None, help='HF model name or path to the local folder with PyTorch or OpenVINO draft model. '
                         'Using this option will create configuration for speculative decoding', dest='draft_source_model')
parser_text.add_argument('--draft_model_name', required=False, default=None, help='Draft model name that should be used in the deployment. '
                         'Equal to draft_source_model if HF model name is used. Available only in draft_source_model has been specified.', dest='draft_model_name')
parser_text.add_argument('--max_prompt_len', required=False, type=int, default=None, help='Sets NPU specific property for maximum number of tokens in the prompt. '
                         'Not effective if target device is not NPU', dest='max_prompt_len')
parser_text.add_argument('--prompt_lookup_decoding', action='store_true', help='Set pipeline to use prompt lookup decoding', dest='prompt_lookup_decoding')
parser_text.add_argument('--reasoning_parser', choices=["qwen3"], help='Set the type of the reasoning parser for reasoning content extraction', dest='reasoning_parser')
parser_text.add_argument('--tool_parser', choices=["llama3","phi4","hermes3", "qwen3","mistral"], help='Set the type of the tool parser for tool calls extraction', dest='tool_parser')
parser_text.add_argument('--enable_tool_guided_generation', action='store_true', help='Enables enforcing tool schema during generation. Requires setting tool_parser', dest='enable_tool_guided_generation')

parser_embeddings = subparsers.add_parser('embeddings', help='[deprecated] export model for embeddings endpoint with models split into separate, versioned directories')
add_common_arguments(parser_embeddings)
parser_embeddings.add_argument('--skip_normalize', default=True, action='store_false', help='Skip normalize the embeddings.', dest='normalize')
parser_embeddings.add_argument('--truncate', default=False, action='store_true', help='Truncate the prompts to fit to the embeddings model', dest='truncate')
parser_embeddings.add_argument('--num_streams', default=1,type=int, help='The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.', dest='num_streams')
parser_embeddings.add_argument('--version', default=1, type=int, help='version of the model', dest='version')

parser_embeddings_ov = subparsers.add_parser('embeddings_ov', help='export model for embeddings endpoint with directory structure aligned with OpenVINO tools')
add_common_arguments(parser_embeddings_ov)
parser_embeddings_ov.add_argument('--skip_normalize', default=True, action='store_false', help='Skip normalize the embeddings.', dest='normalize')
parser_embeddings_ov.add_argument('--pooling', default="CLS", choices=["CLS", "LAST"], help='Embeddings pooling mode', dest='pooling')
parser_embeddings_ov.add_argument('--truncate', default=False, action='store_true', help='Truncate the prompts to fit to the embeddings model', dest='truncate')
parser_embeddings_ov.add_argument('--num_streams', default=1,type=int, help='The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.', dest='num_streams')

parser_rerank = subparsers.add_parser('rerank', help='[deprecated] export model for rerank endpoint with models split into separate, versioned directories')
add_common_arguments(parser_rerank)
parser_rerank.add_argument('--num_streams', default="1", help='The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.', dest='num_streams')
parser_rerank.add_argument('--max_doc_length', default=16000, type=int, help='Maximum length of input documents in tokens', dest='max_doc_length')
parser_rerank.add_argument('--version', default="1", help='version of the model', dest='version')

parser_rerank_ov = subparsers.add_parser('rerank_ov', help='export model for rerank endpoint with directory structure aligned with OpenVINO tools')
add_common_arguments(parser_rerank_ov)
parser_rerank_ov.add_argument('--num_streams', default="1", help='The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.', dest='num_streams')
parser_rerank_ov.add_argument('--max_doc_length', default=16000, type=int, help='Maximum length of input documents in tokens', dest='max_doc_length')

parser_image_generation = subparsers.add_parser('image_generation', help='export model for image generation endpoint')
add_common_arguments(parser_image_generation)
parser_image_generation.add_argument('--num_streams', default=0, type=int, help='The number of parallel execution streams to use for the models in the pipeline.', dest='num_streams')
parser_image_generation.add_argument('--resolution', default="", help='Selection of allowed resolutions in a format of WxH; W=width H=height, space separated. If only one is selected, the pipeline will be reshaped to static.', dest='resolution')
parser_image_generation.add_argument('--guidance_scale', default="", help='Static guidance scale for the image generation requests. If not specified, default 7.5f is used.', dest='guidance_scale')
parser_image_generation.add_argument('--num_images_per_prompt', default="", help='Static number of images to be generated per the image generation request. If not specified, default 1 is used.', dest='num_images_per_prompt')
parser_image_generation.add_argument('--max_resolution', default="", help='Max allowed resolution in a format of WxH; W=width H=height', dest='max_resolution')
parser_image_generation.add_argument('--default_resolution', default="", help='Default resolution when not specified by client', dest='default_resolution')
parser_image_generation.add_argument('--max_num_images_per_prompt', type=int, default=0, help='Max allowed number of images client is allowed to request for a given prompt', dest='max_num_images_per_prompt')
parser_image_generation.add_argument('--default_num_inference_steps', type=int, default=0, help='Default number of inference steps when not specified by client', dest='default_num_inference_steps')
parser_image_generation.add_argument('--max_num_inference_steps', type=int, default=0, help='Max allowed number of inference steps client is allowed to request for a given prompt', dest='max_num_inference_steps')
args = vars(parser.parse_args())

embedding_graph_template = """input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:tokenizer"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "{{model_name}}_tokenizer_model"
    }
  }
}
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:embeddings"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "{{model_name}}_embeddings_model"
    }
  }
}
node {
  input_side_packet: "TOKENIZER_SESSION:tokenizer"
  input_side_packet: "EMBEDDINGS_SESSION:embeddings"
  calculator: "EmbeddingsCalculator"
  input_stream: "REQUEST_PAYLOAD:input"
  output_stream: "RESPONSE_PAYLOAD:output"
  node_options: {
    [type.googleapis.com / mediapipe.EmbeddingsCalculatorOptions]: {
      normalize_embeddings: {% if not normalize %}false{% else %}true{% endif%},
    }
  }
}
"""

embedding_graph_ov_template = """
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
  name: "EmbeddingsExecutor"
  input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
  calculator: "EmbeddingsCalculatorOV"
  input_stream: "REQUEST_PAYLOAD:input"
  output_stream: "RESPONSE_PAYLOAD:output"
  node_options: {
    [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {
      models_path: "{{model_path}}",
      normalize_embeddings: {% if not normalize %}false{% else %}true{% endif%},
      {%- if pooling %}
      pooling: {{pooling}},{% endif %}
      target_device: "{{target_device|default("CPU", true)}}"
    }
  }
}
"""

rerank_graph_ov_template = """
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
  name: "RerankExecutor"
  input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
  calculator: "RerankCalculatorOV"
  input_stream: "REQUEST_PAYLOAD:input"
  output_stream: "RESPONSE_PAYLOAD:output"
  node_options: {
    [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {
      models_path: "{{model_path}}",
      target_device: "{{target_device|default("CPU", true)}}"
    }
  }
}
"""

rerank_graph_template = """input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:tokenizer"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "{{model_name}}_tokenizer_model"
    }
  }
}
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:rerank"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "{{model_name}}_rerank_model"
    }
  }
}
node {
    input_side_packet: "TOKENIZER_SESSION:tokenizer"
    input_side_packet: "RERANK_SESSION:rerank"
    calculator: "RerankCalculator"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
}
"""

text_generation_graph_template = """input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          {%- if pipeline_type %}
          pipeline_type: {{pipeline_type}},{% endif %}
          models_path: "{{model_path}}",
          plugin_config: '{{plugin_config}}',
          enable_prefix_caching: {% if not enable_prefix_caching %}false{% else %} true{% endif%},
          cache_size: {{cache_size|default("10", true)}},
          {%- if max_num_batched_tokens %}
          max_num_batched_tokens: {{max_num_batched_tokens}},{% endif %}
          {%- if not dynamic_split_fuse %}
          dynamic_split_fuse: false, {% endif %}
          max_num_seqs: {{max_num_seqs|default("256", true)}},
          device: "{{target_device|default("CPU", true)}}",
          {%- if draft_model_dir_name %}
          # Speculative decoding configuration
          draft_models_path: "./{{draft_model_dir_name}}",{% endif %}
          {%- if reasoning_parser %}
          reasoning_parser: "{{reasoning_parser}}",{% endif %}
          {%- if tool_parser %}
          tool_parser: "{{tool_parser}}",{% endif %}
          {%- if enable_tool_guided_generation %}
          enable_tool_guided_generation: {% if not enable_tool_guided_generation %}false{% else %} true{% endif%},{% endif %}
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}"""

embeddings_subconfig_template = """{
    "model_config_list": [
    { "config": 
	    {
                "name": "{{model_name}}_tokenizer_model",
                "base_path": "tokenizer"
            }
	},
    { "config": 
	    {
                "name": "{{model_name}}_embeddings_model",
                "base_path": "embeddings",
                "target_device": "{{target_device|default("CPU", true)}}",
                "plugin_config": { "NUM_STREAMS": "{{num_streams|default(1, true)}}" }
            }
	}
   ]
}"""

rerank_subconfig_template = """{
    "model_config_list": [
    { "config": 
	    {
                "name": "{{model_name}}_tokenizer_model",
                "base_path": "tokenizer"
            }
	},
    { "config": 
	    {
                "name": "{{model_name}}_rerank_model",
                "base_path": "rerank",
                "target_device": "{{target_device|default("CPU", true)}}",
                "plugin_config": { "NUM_STREAMS": "{{num_streams|default(1, true)}}" }
            }
	}
   ]
}"""

image_generation_graph_template = """input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
    [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
      models_path: "{{model_path}}",
      {%- if plugin_config_str %}
      plugin_config: '{{plugin_config_str}}',{% endif %}
      device: "{{target_device|default("CPU", true)}}",
      {%- if resolution %}
      resolution: "{{resolution}}",{% endif %}
      {%- if num_images_per_prompt %}
      num_images_per_prompt: {{num_images_per_prompt}},{% endif %}
      {%- if guidance_scale %}
      guidance_scale: {{guidance_scale}},{% endif %}
      {%- if max_resolution %}
      max_resolution: '{{max_resolution}}',{% endif %}
      {%- if default_resolution %}
      default_resolution: '{{default_resolution}}',{% endif %}
      {%- if max_num_images_per_prompt > 0 %}
      max_num_images_per_prompt: {{max_num_images_per_prompt}},{% endif %}
      {%- if default_num_inference_steps > 0 %}
      default_num_inference_steps: {{default_num_inference_steps}},{% endif %}
      {%- if max_num_inference_steps > 0 %}
      max_num_inference_steps: {{max_num_inference_steps}},{% endif %}
    }
  }
}"""

def export_rerank_tokenizer(source_model, destination_path, max_length):
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(source_model)
    hf_tokenizer.model_max_length = max_length
    hf_tokenizer.save_pretrained(destination_path)
    ov_tokenizer = convert_tokenizer(hf_tokenizer, add_special_tokens=False)
    ov.save_model(ov_tokenizer, os.path.join(destination_path, "openvino_tokenizer.xml"))

def set_rt_info(model_folder_path, model_filename, config_filename):
    import openvino as ov
    model = ov.Core().read_model(os.path.join(model_folder_path, model_filename))
    with open(os.path.join(model_folder_path, config_filename), 'r') as config_file:
        config_data = json.load(config_file)
        for key, value in config_data.items():
          try:
              model.set_rt_info(value, ['model_info', key])
          except Exception as e:
              model.set_rt_info(str(value), ['model_info', key])
    temp_model_name = model_filename.replace('.xml', '_temp.xml')
    ov.save_model(model, os.path.join(model_folder_path, temp_model_name))
    del model
    shutil.move(os.path.join(model_folder_path, temp_model_name), os.path.join(model_folder_path, model_filename))
    shutil.move(os.path.join(model_folder_path, temp_model_name.replace('.xml','.bin')), os.path.join(model_folder_path, model_filename.replace('.xml','.bin')))

def get_models_max_context(tmpdirname, config_filename):
    with open(os.path.join(tmpdirname, config_filename), 'r') as config_file:
        config_data = json.load(config_file)
        if config_data['max_position_embeddings'] is not None:
            return config_data['max_position_embeddings']
        if config_data['n_positions'] is not None:
            return config_data['n_positions']
        return None

def add_servable_to_config(config_path, mediapipe_name, base_path):
    print(config_path, mediapipe_name, base_path)
    if not os.path.isfile(config_path):
        print("Creating new config file")
        with open(config_path, 'w') as config_file:
            json.dump({'mediapipe_config_list': [], "model_config_list": []}, config_file, indent=4)
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)
        if 'mediapipe_config_list' not in config_data:
            config_data['mediapipe_config_list'] = []
        mp_list = config_data['mediapipe_config_list']
        updated = False
        for mp_config in mp_list:
            if mp_config['name'] == mediapipe_name:
                mp_config['base_path'] = base_path
                updated = True
        if not updated:
            mp_list.append({'name': mediapipe_name, 'base_path': base_path})
    with open(config_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)
    print("Added servable to config file", config_path)

def export_text_generation_model(model_repository_path, source_model, model_name, precision, task_parameters, config_file_path):
    model_path = "./"
    ### Export model
    if os.path.isfile(os.path.join(source_model, 'openvino_model.xml')) or os.path.isfile(os.path.join(source_model, 'openvino_language_model.xml')):
        print("OV model is source folder. Skipping conversion.")
        model_path = source_model
    elif source_model.startswith("OpenVINO/"):
        if precision:
            print("Precision change is not supported for OpenVINO models. Parameter --weight-format {} will be ignored.".format(precision))
        hugging_face_cmd = "huggingface-cli download {} --local-dir {} ".format(source_model, os.path.join(model_repository_path, model_name))
        if os.system(hugging_face_cmd):
            raise ValueError("Failed to download llm model", source_model)
    else: # assume HF model name or local pytorch model folder
        llm_model_path = os.path.join(model_repository_path, model_name)
        print("Exporting LLM model to ", llm_model_path)
        if not os.path.isdir(llm_model_path) or args['overwrite_models']:
            if task_parameters['target_device'] == 'NPU':
                if precision != 'int4':
                    print("NPU target device requires int4 precision. Changing to int4")
                    precision = 'int4'
                if task_parameters['extra_quantization_params'] == "":
                    print("Using default quantization parameters for NPU: --sym --ratio 1.0 --group-size -1")
                    task_parameters['extra_quantization_params'] = "--sym --ratio 1.0 --group-size -1"
            optimum_command = "optimum-cli export openvino --model {} --weight-format {} {} --trust-remote-code {}".format(source_model, precision, task_parameters['extra_quantization_params'], llm_model_path)
            if os.system(optimum_command):
                raise ValueError("Failed to export llm model", source_model)
            if not (os.path.isfile(os.path.join(llm_model_path, 'openvino_detokenizer.xml'))):
                print("Tokenizer and detokenizer not found in the exported model. Exporting tokenizer and detokenizer from HF model")
                convert_tokenizer_command = "convert_tokenizer --with-detokenizer -o {} {}".format(llm_model_path, source_model)
                if os.system(convert_tokenizer_command):
                    raise ValueError("Failed to export tokenizer and detokenizer", source_model)
    ### Export draft model for speculative decoding 
    draft_source_model = task_parameters.get("draft_source_model", None)
    draft_model_dir_name = None   
    if draft_source_model:
        draft_model_dir_name = draft_source_model.replace("/", "-") # flatten the name so we don't create nested directory structure
        draft_llm_model_path = os.path.join(model_repository_path, model_name, draft_model_dir_name)
        if os.path.isfile(os.path.join(draft_llm_model_path, 'openvino_model.xml')):
            print("OV model is source folder. Skipping conversion.")
        elif source_model.startswith("OpenVINO/"):
            if precision:
                print("Precision change is not supported for OpenVINO models. Parameter --weight-format {} will be ignored.".format(precision))
            hugging_face_cmd = "huggingface-cli download {} --local-dir {} ".format(source_model, os.path.join(model_repository_path, model_name))
            if os.system(hugging_face_cmd):
                raise ValueError("Failed to download llm model", source_model)    
        else: # assume HF model name or local pytorch model folder
            print("Exporting draft LLM model to ", draft_llm_model_path)
            if not os.path.isdir(draft_llm_model_path) or args['overwrite_models']:
                optimum_command = "optimum-cli export openvino --model {} --weight-format {} --trust-remote-code {}".format(draft_source_model, precision, draft_llm_model_path)
                if os.system(optimum_command):
                    raise ValueError("Failed to export llm model", source_model)

    ### Prepare plugin config string for jinja rendering
    plugin_config = {}
    if task_parameters['kv_cache_precision'] is not None:
        plugin_config['KV_CACHE_PRECISION'] = task_parameters['kv_cache_precision']
    if task_parameters['max_prompt_len'] is not None:
        if task_parameters['target_device'] != 'NPU':
            raise ValueError("max_prompt_len is only supported for NPU target device")
        if task_parameters['max_prompt_len'] <= 0:
            raise ValueError("max_prompt_len should be a positive integer")
        plugin_config['MAX_PROMPT_LEN'] = task_parameters['max_prompt_len']
    if task_parameters['ov_cache_dir'] is not None:
        plugin_config['CACHE_DIR'] = task_parameters['ov_cache_dir']

    if task_parameters['prompt_lookup_decoding']:
        plugin_config['prompt_lookup'] = True
    
    # Additional plugin properties for HETERO
    if "HETERO" in task_parameters['target_device']:
        if task_parameters['pipeline_type'] is None:
            raise ValueError("pipeline_type should be specified for HETERO target device. It should be set to either LM or VLM")
        if task_parameters['pipeline_type'] not in ["LM", "VLM"]:
            raise ValueError("pipeline_type should be either LM or VLM for HETERO target device")
        plugin_config['MODEL_DISTRIBUTION_POLICY'] = 'PIPELINE_PARALLEL'

    plugin_config_str = json.dumps(plugin_config)
    task_parameters['plugin_config'] = plugin_config_str
    
    os.makedirs(os.path.join(model_repository_path, model_name), exist_ok=True)
    gtemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(text_generation_graph_template)
    print("task_parameters", task_parameters)
    graph_content = gtemplate.render(model_path=model_path, draft_model_dir_name=draft_model_dir_name, **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'graph.pbtxt'), 'w') as f:
        f.write(graph_content)
    print("Created graph {}".format(os.path.join(model_repository_path, model_name, 'graph.pbtxt')))

    if template_parameters.get("tool_parser") is not None:
        print("Adding tuned chat template")
        template_mapping = {
            "phi4": "tool_chat_template_phi4_mini.jinja",
            "llama3": "tool_chat_template_llama3.1_json.jinja",
            "hermes3": "tool_chat_template_hermes.jinja",
            "mistral": "tool_chat_template_mistral_parallel.jinja",
            "qwen3": None
            }
        template_name = template_mapping[task_parameters.get("tool_parser")]
        if template_name is not None:
            template_path = os.path.join(model_repository_path, model_name, "template.jinja")
            import requests
            response = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/" + template_name)
            print(response.raise_for_status())
            with open(template_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded tuned chat template to {template_path}")

    add_servable_to_config(config_file_path, model_name, os.path.relpath( os.path.join(model_repository_path, model_name), os.path.dirname(config_file_path)))

def export_embeddings_model(model_repository_path, source_model, model_name, precision, task_parameters, version, config_file_path, truncate=True):
    if os.path.isfile(os.path.join(model_name, 'openvino_model.xml')):
        print("OV model is source folder. Skipping conversion.")
        os.makedirs(os.path.join(model_repository_path, model_name, 'embeddings', version), exist_ok=True)
        os.makedirs(os.path.join(model_repository_path, model_name, 'tokenizer', version), exist_ok=True)
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_tokenizer.xml'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.xml'))
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_tokenizer.bin'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.bin'))
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_model.xml'), os.path.join(model_repository_path, model_name, 'embeddings', version, 'model.xml'))
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_model.bin'), os.path.join(model_repository_path, model_name, 'embeddings', version, 'model.bin'))
    else: # assume HF model 
        set_max_context_length = ""
        with tempfile.TemporaryDirectory() as tmpdirname:
            embeddings_path = os.path.join(model_repository_path, model_name,'embeddings', version)
            print("Exporting embeddings model to ",embeddings_path)
            if not os.path.isdir(embeddings_path) or args['overwrite_models']:
                optimum_command = "optimum-cli export openvino --disable-convert-tokenizer --model {} --task feature-extraction --weight-format {} {} --trust-remote-code --library sentence_transformers {}".format(source_model, precision, task_parameters['extra_quantization_params'], tmpdirname)
                if os.system(optimum_command):
                    raise ValueError("Failed to export embeddings model", source_model)
                set_rt_info(tmpdirname, 'openvino_model.xml', 'config.json')
                if truncate:
                    max_context_length = get_models_max_context(tmpdirname, 'config.json')
                    if max_context_length is not None:
                        set_max_context_length = "--max_length " + str(get_models_max_context(tmpdirname, 'config.json'))
                os.makedirs(embeddings_path, exist_ok=True)
                shutil.move(os.path.join(tmpdirname, 'openvino_model.xml'), os.path.join(embeddings_path, 'model.xml'))
                shutil.move(os.path.join(tmpdirname, 'openvino_model.bin'), os.path.join(embeddings_path, 'model.bin'))
            tokenizer_path = os.path.join(model_repository_path, model_name,'tokenizer', version)
            print("Exporting tokenizer to ", tokenizer_path)
            if not os.path.isdir(tokenizer_path) or args['overwrite_models']:
                from openvino_tokenizers import convert_tokenizer
                convert_tokenizer_command = "convert_tokenizer -o {} {} {}".format(tmpdirname, source_model, set_max_context_length) 
                if (os.system(convert_tokenizer_command)):
                    raise ValueError("Failed to export tokenizer model", source_model)
                set_rt_info(tmpdirname, 'openvino_tokenizer.xml', 'tokenizer_config.json')
                os.makedirs(tokenizer_path, exist_ok=True)
                shutil.move(os.path.join(tmpdirname, 'openvino_tokenizer.xml'), os.path.join(tokenizer_path, 'model.xml'))
                shutil.move(os.path.join(tmpdirname, 'openvino_tokenizer.bin'), os.path.join(tokenizer_path, 'model.bin'))
    gtemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(embedding_graph_template)
    graph_content = gtemplate.render(model_name=model_name, **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'graph.pbtxt'), 'w') as f:
        f.write(graph_content)
    print("Created graph {}".format(os.path.join(model_repository_path, model_name, 'graph.pbtxt')))
    stemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(embeddings_subconfig_template)
    subconfig_content = stemplate.render(model_name=model_name, **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'subconfig.json'), 'w') as f:
        f.write(subconfig_content)
    print("Created subconfig {}".format(os.path.join(model_repository_path, model_name, 'subconfig.json')))
    add_servable_to_config(config_file_path, model_name, os.path.relpath(os.path.join(model_repository_path, model_name), os.path.dirname(config_file_path)))

def export_embeddings_model_ov(model_repository_path, source_model, model_name, precision, task_parameters, config_file_path, truncate=True):
    set_max_context_length = ""
    destination_path = os.path.join(model_repository_path, model_name)
    print("Exporting embeddings model to ",destination_path)
    if not os.path.isdir(destination_path) or args['overwrite_models']:
        optimum_command = "optimum-cli export openvino --model {} --disable-convert-tokenizer --task feature-extraction --weight-format {} {} --trust-remote-code --library sentence_transformers {}".format(source_model, precision, task_parameters['extra_quantization_params'], destination_path)
        if os.system(optimum_command):
            raise ValueError("Failed to export embeddings model", source_model)
        if truncate:
            max_context_length = get_models_max_context(destination_path, 'config.json')
            if max_context_length is not None:
                set_max_context_length = "--max_length " + str(get_models_max_context(destination_path, 'config.json'))
        print("Exporting tokenizer to ", destination_path)
        convert_tokenizer_command = "convert_tokenizer -o {} {} {}".format(destination_path, source_model, set_max_context_length) 
        if (os.system(convert_tokenizer_command)):
            raise ValueError("Failed to export tokenizer model", source_model)
    gtemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(embedding_graph_ov_template)
    graph_content = gtemplate.render(model_path="./", **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'graph.pbtxt'), 'w') as f:
        f.write(graph_content)
    print("Created graph {}".format(os.path.join(model_repository_path, model_name, 'graph.pbtxt')))
    add_servable_to_config(config_file_path, model_name, os.path.relpath(os.path.join(model_repository_path, model_name), os.path.dirname(config_file_path)))

def export_rerank_model_ov(model_repository_path, source_model, model_name, precision, task_parameters, config_file_path, max_doc_length):
    destination_path = os.path.join(model_repository_path, model_name)
    print("Exporting rerank model to ",destination_path)
    if not os.path.isdir(destination_path) or args['overwrite_models']:
        optimum_command = "optimum-cli export openvino --model {} --disable-convert-tokenizer --task text-classification --weight-format {} {} --trust-remote-code {}".format(source_model, precision, task_parameters['extra_quantization_params'], destination_path)
        if os.system(optimum_command):
            raise ValueError("Failed to export rerank model", source_model)
        print("Exporting tokenizer to ", destination_path)
        export_rerank_tokenizer(source_model, destination_path, max_doc_length)
    gtemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(rerank_graph_ov_template)
    graph_content = gtemplate.render(model_path="./", **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'graph.pbtxt'), 'w') as f:
        f.write(graph_content)
    print("Created graph {}".format(os.path.join(model_repository_path, model_name, 'graph.pbtxt')))
    add_servable_to_config(config_file_path, model_name, os.path.relpath(os.path.join(model_repository_path, model_name), os.path.dirname(config_file_path)))

def export_rerank_model(model_repository_path, source_model, model_name, precision, task_parameters, version, config_file_path, max_doc_length):
    if os.path.isfile(os.path.join(model_name, 'openvino_model.xml')):
        print("OV model is source folder. Skipping conversion.")
        os.makedirs(os.path.join(model_repository_path, model_name, 'rerank', version), exist_ok=True)
        os.makedirs(os.path.join(model_repository_path, model_name, 'tokenizer', version), exist_ok=True)
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_tokenizer.xml'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.xml'))
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_tokenizer.bin'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.bin'))
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_model.xml'), os.path.join(model_repository_path, model_name, 'rerank', version, 'model.xml'))
        shutil.move(os.path.join(model_repository_path, model_name, 'openvino_model.bin'), os.path.join(model_repository_path, model_name, 'rerank', version, 'model.bin'))
    else: # assume HF model name
        with tempfile.TemporaryDirectory() as tmpdirname:
            embeddings_path = os.path.join(model_repository_path, model_name, 'rerank', version)
            print("Exporting rerank model to ",embeddings_path)
            if not os.path.isdir(embeddings_path) or args['overwrite_models']:
                optimum_command = "optimum-cli export openvino --disable-convert-tokenizer --model {} --task text-classification --weight-format {} {} --trust-remote-code {}".format(source_model, precision, task_parameters['extra_quantization_params'], tmpdirname)
                if os.system(optimum_command):
                    raise ValueError("Failed to export rerank model", source_model)
                set_rt_info(tmpdirname, 'openvino_model.xml', 'config.json')
                os.makedirs(embeddings_path, exist_ok=True)
                shutil.move(os.path.join(tmpdirname, 'openvino_model.xml'), os.path.join(embeddings_path, 'model.xml'))
                shutil.move(os.path.join(tmpdirname, 'openvino_model.bin'), os.path.join(embeddings_path, 'model.bin'))
            tokenizer_path = os.path.join(model_repository_path, model_name,'tokenizer', version)
            print("Exporting tokenizer to ",tokenizer_path)
            if not os.path.isdir(tokenizer_path) or args['overwrite_models']:
                export_rerank_tokenizer(source_model, tmpdirname, max_doc_length)
                set_rt_info(tmpdirname, 'openvino_tokenizer.xml', 'tokenizer_config.json')
                os.makedirs(tokenizer_path, exist_ok=True)
                shutil.move(os.path.join(tmpdirname, 'openvino_tokenizer.xml'), os.path.join(tokenizer_path, 'model.xml'))
                shutil.move(os.path.join(tmpdirname, 'openvino_tokenizer.bin'), os.path.join(tokenizer_path, 'model.bin'))
    gtemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(rerank_graph_template)
    graph_content = gtemplate.render(model_name=model_name, **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'graph.pbtxt'), 'w') as f:
        f.write(graph_content)
    print("Created graph {}".format(os.path.join(model_repository_path, model_name, 'graph.pbtxt')))
    stemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(rerank_subconfig_template)
    subconfig_content = stemplate.render(model_name=model_name, **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'subconfig.json'), 'w') as f:
        f.write(subconfig_content)
    print("Created subconfig {}".format(os.path.join(model_repository_path, model_name, 'subconfig.json')))
    add_servable_to_config(config_file_path, model_name, os.path.relpath( os.path.join(model_repository_path, model_name), os.path.dirname(config_file_path)))


def export_image_generation_model(model_repository_path, source_model, model_name, precision, task_parameters, config_file_path, num_streams):
    model_path = "./"
    target_path = os.path.join(model_repository_path, model_name)
    model_index_path = os.path.join(target_path, 'model_index.json')

    if os.path.isfile(model_index_path):
        print("Model index file already exists. Skipping conversion, re-generating graph only.")
    else:
        optimum_command = "optimum-cli export openvino --model {} --weight-format {} {} {}".format(source_model, precision, task_parameters['extra_quantization_params'], target_path)
        print(f'optimum cli command: {optimum_command}')
        if os.system(optimum_command):
            raise ValueError("Failed to export image generation model", source_model)

    plugin_config = {}
    assert num_streams >= 0, "num_streams should be a non-negative integer"
    if num_streams > 0:
        plugin_config['NUM_STREAMS'] = num_streams
    if 'ov_cache_dir' in task_parameters and task_parameters['ov_cache_dir'] is not None:
        plugin_config['CACHE_DIR'] = task_parameters['ov_cache_dir']
    
    if len(plugin_config) > 0:
        task_parameters['plugin_config_str'] = json.dumps(plugin_config)

    # assert that max_resolution if exists, is in WxH format
    for param in ['max_resolution', 'default_resolution']:
        if task_parameters[param]:
            if 'x' not in task_parameters[param]:
                raise ValueError(param + " should be in WxH format, e.g. 1024x768")
            width, height = task_parameters[param].split('x')
            if not (width.isdigit() and height.isdigit()):
                raise ValueError(param + " should be in WxH format with positive integers, e.g. 1024x768")
            task_parameters[param] = '{}x{}'.format(int(width), int(height))

    gtemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(image_generation_graph_template)
    graph_content = gtemplate.render(model_path=model_path, **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'graph.pbtxt'), 'w') as f:
         f.write(graph_content)
    print("Created graph {}".format(os.path.join(model_repository_path, model_name, 'graph.pbtxt')))
    add_servable_to_config(config_file_path, model_name, os.path.relpath( os.path.join(model_repository_path, model_name), os.path.dirname(config_file_path)))


if not os.path.isdir(args['model_repository_path']):
    raise ValueError(f"The model repository path '{args['model_repository_path']}' is not a valid directory.")
if args['source_model'] is None:
    args['source_model'] = args['model_name']
if args['model_name'] is None:
    args['model_name'] = args['source_model']
if args['model_name'] is None and args['source_model'] is None:
    raise ValueError("Either model_name or source_model should be provided")

### Speculative decoding specific
if args['task'] == 'text_generation':
    if args['draft_source_model'] is None:
        args['draft_source_model'] = args['draft_model_name']
    if args['draft_model_name'] is None:
        args['draft_model_name'] = args['draft_source_model']
###

if args['extra_quantization_params'] is None:
    args['extra_quantization_params'] = ""

template_parameters = {k: v for k, v in args.items() if k not in ['model_repository_path', 'source_model', 'model_name', 'precision', 'version', 'config_file_path', 'overwrite_models']}
print("template params:", template_parameters)

if args['task'] == 'text_generation':
    export_text_generation_model(args['model_repository_path'], args['source_model'], args['model_name'], args['precision'], template_parameters, args['config_file_path'])

elif args['task'] == 'embeddings':
    export_embeddings_model(args['model_repository_path'], args['source_model'], args['model_name'],  args['precision'], template_parameters, str(args['version']), args['config_file_path'], args['truncate'])

elif args['task'] == 'embeddings_ov':
    export_embeddings_model_ov(args['model_repository_path'], args['source_model'], args['model_name'],  args['precision'], template_parameters, args['config_file_path'], args['truncate'])

elif args['task'] == 'rerank':
    export_rerank_model(args['model_repository_path'], args['source_model'], args['model_name'] ,args['precision'], template_parameters, str(args['version']), args['config_file_path'], args['max_doc_length'])

elif args['task'] == 'rerank_ov':
    export_rerank_model_ov(args['model_repository_path'], args['source_model'], args['model_name'] ,args['precision'], template_parameters, args['config_file_path'], args['max_doc_length'])

elif args['task'] == 'image_generation':
    template_parameters = {k: v for k, v in args.items() if k in [
        'ov_cache_dir',
        'target_device',
        'resolution',
        'num_images_per_prompt',
        'guidance_scale',
        'max_resolution',
        'default_resolution',
        'max_num_images_per_prompt',
        'default_num_inference_steps',
        'max_num_inference_steps',
        'extra_quantization_params'
    ]}
    export_image_generation_model(args['model_repository_path'], args['source_model'], args['model_name'], args['precision'], template_parameters, args['config_file_path'], args['num_streams'])
