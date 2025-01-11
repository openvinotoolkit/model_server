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
from openvino_tokenizers import convert_tokenizer, connect_models
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from peft import PeftModel
#from peft.utils import PeftConfig
import torch
import jinja2
import json
import shutil
import tempfile
import openvino as ov

def add_common_arguments(parser):
    parser.add_argument('--model_repository_path', required=False, default='models', help='Where the model should be exported to', dest='model_repository_path')
    parser.add_argument('--source_model', required=True, help='HF model name or path to the local folder with PyTorch or OpenVINO model', dest='source_model')
    parser.add_argument('--model_name', required=False, default=None, help='Model name that should be used in the deployment. Equal to source_name if HF model name is used', dest='model_name')
    parser.add_argument('--weight-format', default='int8', help='precision of the exported model', dest='precision')
    parser.add_argument('--config_file_path', default='config.json', help='path to the config file', dest='config_file_path')
    parser.add_argument('--overwrite_models', default=False, action='store_true', help='Overwrite the model if it already exists in the models repository', dest='overwrite_models')
    parser.add_argument('--target_device', default="CPU", help='CPU or GPU, default is CPU', dest='target_device')

parser = argparse.ArgumentParser(description='Export Hugging face models to OVMS models repository including all configuration for deployments')

subparsers = parser.add_subparsers(help='subcommand help', required=True, dest='task')
parser_text = subparsers.add_parser('text_generation', help='export model for chat and completion endpoints')
add_common_arguments(parser_text)
parser_text.add_argument('--kv_cache_precision', default=None, choices=["u8"], help='u8 or empty (model default). Reduced kv cache precision to u8 lowers the cache size consumption.', dest='kv_cache_precision')
parser_text.add_argument('--enable_prefix_caching', action='store_true', help='This algorithm is used to cache the prompt tokens.', dest='enable_prefix_caching')
parser_text.add_argument('--disable_dynamic_split_fuse', action='store_false', help='The maximum number of tokens that can be batched together.', dest='dynamic_split_fuse')
parser_text.add_argument('--max_num_batched_tokens', default=None, help='empty or integer. The maximum number of tokens that can be batched together.', dest='max_num_batched_tokens')
parser_text.add_argument('--max_num_seqs', default=None, help='256 by default. The maximum number of sequences that can be processed together.', dest='max_num_seqs')
parser_text.add_argument('--cache_size', default=10, type=int, help='cache size in GB', dest='cache_size')
parser_text.add_argument('--adapter',action='append', help='lora adapter in HF or a local folder with the adapter', dest='adapter')
parser_embeddings = subparsers.add_parser('embeddings', help='export model for embeddings endpoint')
add_common_arguments(parser_embeddings)
parser_embeddings.add_argument('--skip_normalize', default=True, action='store_false', help='Skip normalize the embeddings.', dest='normalize')
parser_embeddings.add_argument('--num_streams', default=1,type=int, help='The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.', dest='num_streams')
parser_embeddings.add_argument('--version', default=1, type=int, help='version of the model', dest='version')

parser_rerank = subparsers.add_parser('rerank', help='export model for rerank endpoint')
add_common_arguments(parser_rerank)
parser_rerank.add_argument('--num_streams', default="1", help='The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.', dest='num_streams')
parser_rerank.add_argument('--max_doc_length', default=16000, type=int, help='Maximum length of input documents in tokens', dest='max_doc_length')
parser_rerank.add_argument('--version', default="1", help='version of the model', dest='version')
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
          models_path: "{{model_path}}",
          plugin_config: '{ {% if kv_cache_precision %}"KV_CACHE_PRECISION": "{{kv_cache_precision}}"{% endif %}}',
          enable_prefix_caching: {% if not enable_prefix_caching %}false{% else %} true{% endif%},
          cache_size: {{cache_size|default("10", true)}},
          {%- if max_num_batched_tokens %}
          max_num_batched_tokens: {{max_num_batched_tokens}},{% endif %}
          {%- if not dynamic_split_fuse %}
          dynamic_split_fuse: false, {% endif %}
          max_num_seqs: {{max_num_seqs|default("256", true)}},
          device: "{{target_device|default("CPU", true)}}",

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
                "plugin_config": { "NUM_STREAMS": {{num_streams|default("1", true)}} }
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
                "plugin_config": { "NUM_STREAMS": {{num_streams|default("1", true)}} }
            }
	}
   ]
}"""

def export_rerank_tokenizer(source_model, destination_path, max_length):
    hf_tokenizer = AutoTokenizer.from_pretrained(source_model)
    hf_tokenizer.model_max_length = max_length
    hf_tokenizer.save_pretrained(destination_path)
    ov_tokenizer = convert_tokenizer(hf_tokenizer, add_special_tokens=False)
    ov.save_model(ov_tokenizer, os.path.join(destination_path, "openvino_tokenizer.xml"))

def set_rt_info(model_folder_path, model_filename, config_filename):
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

def export_text_generation_model(model_repository_path, source_model, model_name, precision, task_parameters, config_file_path, adapter):
    model_path = "./"
    if os.path.isfile(os.path.join(source_model, 'openvino_model.xml')):
            print("OV model is source folder. Skipping conversion.")
            model_path = source_model
    else: # assume HF model name or local pytorch model folder
        llm_model_path = os.path.join(model_repository_path, model_name)
        tmp_folder = None
        if not os.path.isdir(llm_model_path) or args['overwrite_models']:
            if adapter is not None:
                tmp_folder = tempfile.mkdtemp()
                print("Loading model with adapter")
                HFmodel = LlamaForCausalLM.from_pretrained(source_model, trust_remote_code=True)  
                for adapteri in adapter:
                    print("Loading adapter", adapteri)
                    HFmodel = PeftModel.from_pretrained(HFmodel, adapteri)
                print("Merging model with adapters")
                HFmodel = HFmodel.merge_and_unload()
                HFmodel.save_pretrained(tmp_folder)
                tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)
                tokenizer.save_pretrained(tmp_folder)
                source_model = tmp_folder
            print("Exporting LLM model to ", llm_model_path)
            optimum_command = "optimum-cli export openvino --task text-generation --disable-convert-tokenizer --model {} --weight-format {} --trust-remote-code {}".format(source_model, precision, llm_model_path)
            if os.system(optimum_command):
                raise ValueError("Failed to export llm model", source_model)
            print("Exporting tokenizer to ", llm_model_path)
            convert_tokenizer_command = "convert_tokenizer --utf8_replace_mode replace --with-detokenizer --skip-special-tokens --streaming-detokenizer -o {} {}".format(llm_model_path, source_model) 
            if (os.system(convert_tokenizer_command)):
                raise ValueError("Failed to export tokenizer model", source_model)
            if adapter is not None:
                shutil.rmtree(tmp_folder)
    os.makedirs(os.path.join(model_repository_path, model_name), exist_ok=True)
    gtemplate = jinja2.Environment(loader=jinja2.BaseLoader).from_string(text_generation_graph_template)
    graph_content = gtemplate.render(tokenizer_model="{}_tokenizer_model".format(model_name), embeddings_model="{}_embeddings_model".format(model_name), model_path=model_path, **task_parameters)
    with open(os.path.join(model_repository_path, model_name, 'graph.pbtxt'), 'w') as f:
        f.write(graph_content)
    print("Created graph {}".format(os.path.join(model_repository_path, model_name, 'graph.pbtxt')))
    add_servable_to_config(config_file_path, model_name, os.path.relpath( os.path.join(model_repository_path, model_name), os.path.dirname(config_file_path)))
    
    
def export_embeddings_model(model_repository_path, source_model, model_name, precision, task_parameters, version, config_file_path):
    if os.path.isfile(os.path.join(source_model, 'openvino_model.xml')):
        print("OV model is source folder. Skipping conversion.")
        os.makedirs(os.path.join(model_repository_path, model_name, 'embeddings', version), exist_ok=True)
        os.makedirs(os.path.join(model_repository_path, model_name, 'tokenizer', version), exist_ok=True)
        shutil.move(os.path.join(source_model, 'openvino_tokenizer.xml'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.xml'))
        shutil.move(os.path.join(source_model, 'openvino_tokenizer.bin'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.bin'))
        shutil.move(os.path.join(source_model, 'openvino_model.xml'), os.path.join(model_repository_path, model_name, 'embeddings', version, 'model.xml'))
        shutil.move(os.path.join(source_model, 'openvino_model.bin'), os.path.join(model_repository_path, model_name, 'embeddings', version, 'model.bin'))
    else: # assume HF model name
        with tempfile.TemporaryDirectory() as tmpdirname:
            embeddings_path = os.path.join(model_repository_path, model_name,'embeddings', version)
            print("Exporting embeddings model to ",embeddings_path)
            if not os.path.isdir(embeddings_path) or args['overwrite_models']:
                optimum_command = "optimum-cli export openvino --disable-convert-tokenizer --model {} --task feature-extraction --weight-format {} --trust-remote-code --library sentence_transformers {}".format(source_model, precision, tmpdirname)
                if os.system(optimum_command):
                    raise ValueError("Failed to export embeddings model", source_model)
                set_rt_info(tmpdirname, 'openvino_model.xml', 'config.json')
                os.makedirs(embeddings_path, exist_ok=True)
                shutil.move(os.path.join(tmpdirname, 'openvino_model.xml'), os.path.join(embeddings_path, 'model.xml'))
                shutil.move(os.path.join(tmpdirname, 'openvino_model.bin'), os.path.join(embeddings_path, 'model.bin'))
            tokenizer_path = os.path.join(model_repository_path, model_name,'tokenizer', version)
            print("Exporting tokenizer to ", tokenizer_path)
            if not os.path.isdir(tokenizer_path) or args['overwrite_models']:
                convert_tokenizer_command = "convert_tokenizer -o {} {}".format(tmpdirname, source_model) 
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


def export_rerank_model(model_repository_path, source_model, model_name, precision, task_parameters, version, config_file_path, max_doc_length):
    if os.path.isfile(os.path.join(source_model, 'openvino_model.xml')):
        print("OV model is source folder. Skipping conversion.")
        os.makedirs(os.path.join(model_repository_path, model_name, 'rerank', version), exist_ok=True)
        os.makedirs(os.path.join(model_repository_path, model_name, 'tokenizer', version), exist_ok=True)
        shutil.move(os.path.join(source_model, 'openvino_tokenizer.xml'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.xml'))
        shutil.move(os.path.join(source_model, 'openvino_tokenizer.bin'), os.path.join(model_repository_path, model_name, 'tokenizer', version, 'model.bin'))
        shutil.move(os.path.join(source_model, 'openvino_model.xml'), os.path.join(model_repository_path, model_name, 'rerank', version, 'model.xml'))
        shutil.move(os.path.join(source_model, 'openvino_model.bin'), os.path.join(model_repository_path, model_name, 'rerank', version, 'model.bin'))
    else: # assume HF model name
        with tempfile.TemporaryDirectory() as tmpdirname:
            embeddings_path = os.path.join(model_repository_path, model_name,'rerank', version)
            print("Exporting rerank model to ",embeddings_path)
            if not os.path.isdir(embeddings_path) or args['overwrite_models']:
                optimum_command = "optimum-cli export openvino --disable-convert-tokenizer --model {} --task text-classification --weight-format {} --trust-remote-code {}".format(source_model, precision, tmpdirname)
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


if not os.path.isdir(args['model_repository_path']):
    raise ValueError(f"The model repository path '{args['model_repository_path']}' is not a valid directory.")
if args['source_model'] is None:
    args['source_model'] = args['model_name']
if args['model_name'] is None:
    args['model_name'] = args['source_model']
if args['model_name'] is None and args['source_model'] is None:
    raise ValueError("Either model_name or source_model should be provided")

template_parameters = {k: v for k, v in args.items() if k not in ['model_repository_path', 'source_model', 'model_name', 'precision', 'version', 'config_file_path', 'overwrite_models']}
print("template params:",template_parameters)

if args['task'] == 'text_generation':
    export_text_generation_model(args['model_repository_path'], args['source_model'], args['model_name'], args['precision'], template_parameters, args['config_file_path'], args['adapter'])

elif args['task'] == 'embeddings':
    export_embeddings_model(args['model_repository_path'], args['source_model'], args['model_name'],  args['precision'], template_parameters, str(args['version']), args['config_file_path'])

elif args['task'] == 'rerank':
    export_rerank_model(args['model_repository_path'], args['source_model'], args['model_name'] ,args['precision'], template_parameters, str(args['version']), args['config_file_path'], args['max_doc_length'])


