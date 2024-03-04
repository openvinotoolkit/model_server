import gradio as gr
import argparse

from streamer import OvmsStreamer


parser = argparse.ArgumentParser(description='Script to download LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot')

parser.add_argument('--web_url',
                    required=True,
                    help='Web server URL')
parser.add_argument('--ovms_url',
                    required=True,
                    help='OVMS server URL')
args = parser.parse_args()


def callback(message, history):
    streamer = OvmsStreamer(args.ovms_url.split(':')[0], int(args.ovms_url.split(':')[1]))
    streamer.request_async(message)
    result = ""
    for completion in streamer:
        result += completion
        yield result

gr.ChatInterface(callback, concurrency_limit=None).launch(server_name=args.web_url.split(':')[0], server_port=int(args.web_url.split(':')[1]))
