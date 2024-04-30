#*****************************************************************************
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************

import gradio as gr
import argparse
from streamer import OvmsStreamer
import os
from servable_stream.utils import config_reader as reader

parser = argparse.ArgumentParser(description='Gradio frontend launcher')

parser.add_argument('--web_url',
                    required=True,
                    help='Web server URL')
parser.add_argument('--ovms_url',
                    required=True,
                    help='OVMS server URL')
args = parser.parse_args()

config = reader.read_config('servable_stream/config.yaml')

list_of_videos = os.listdir('servable_stream/' + config['videos'])

#video_to_play = 'documents/videos_all/op_1_0320241821.mp4'
video_to_play = None

def callback(message, history):
    global video_to_play
    streamer = OvmsStreamer(args.ovms_url.split(':')[0], int(args.ovms_url.split(':')[1]))
    streamer.request_async(message)
    result = ""
    videofile = ""
    compflag = False
    last_word = None
    for completion in streamer:
        result += completion
        last_word = completion
        yield result
        
    videofile = last_word
    video_to_play = 'servable_stream/' + config['videos'] + videofile
    print (f'video path {video_to_play}')
    print(result, flush=True)
    print(videofile, flush=True)

def vcallback(video):
    print(video_to_play)
    return video_to_play

def update_video(input_video):
    print("In update_video", input_video)
    return gr.Video(value=video_to_play)

css = """
.container {
    height: 100vh;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Row():        
        with gr.Column(scale=1, min_width=200):
            ChatBlock = gr.ChatInterface(callback, retry_btn=None, undo_btn=None, \
                examples = [
                    'People reading product description',
                    'Picking an item from shelf',
                    'Man holding red shopping cart',
                    'Man in blue shirt shopping',
                ]) 
        with gr.Column(scale=1, min_width=200):
            VidBlock = gr.Video(vcallback(video_to_play), height="50vh", interactive=False, scale=4, autoplay=True, show_download_button=False, show_share_button=False)
            update = gr.Button('Play')
        
            update.click(update_video, inputs=None, outputs=[VidBlock])
        #VidBlock = gr.Interface(fn=vcallback, allow_flagging="never", inputs=None, outputs=gr.Video(None, height="50vh", interactive=False, scale=4, autoplay=True, show_download_button=False, show_share_button=False))
 

demo.launch(server_name=args.web_url.split(':')[0], server_port=int(args.web_url.split(':')[1]))