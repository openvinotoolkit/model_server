#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License. #
from __future__ import unicode_literals
from __future__ import print_function

import pyaudio
import sys
import time
import msvcrt
import wave
import threading
import paramiko
import os
import getpass

CHANNELS, RATE, FORMAT = 1, 8000, pyaudio.paInt16
BATCH_SIZE = 4560
PORT = 22

class LiveDemo(object):

    def __init__(self, host, host_path, user, password, dec_args):
        self.batch_size = BATCH_SIZE
        self.args = dec_args
        self.pin, self.stream = None, None
        self.frames = []
        self.utt_frames, self.new_frames = 0, 0
        self.utt_end, self.dialog_end, self.recording = False, False, False
        self.host_path = host_path

        try:
            self.transport = paramiko.Transport((host, PORT))
            start = time.time() 
            self.transport.connect(username = user, password = password)
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)
            print("Connection success.")
        except Exception as e:
            print("Connection issue: " + e)


    def setup(self):
        self.pin = pyaudio.PyAudio()
        self.stream = self.pin.open(format=FORMAT, channels=CHANNELS,
                                    rate=RATE, input=True, frames_per_buffer=self.batch_size,
                                    stream_callback=self.get_audio_callback())
        self.utt_frames, self.new_frames = 0, 0
        self.utt_end, self.dialog_end, self.recording = False, False, False
        self.frames = []

    def tear_down(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.pin is not None:
            self.pin.terminate()
        p, stream = None, None
        self.frames = []

    def get_audio_callback(self):
        def frame_in(in_data, frame_count, time_info, status):
            self.frames.append(in_data)
            return in_data, pyaudio.paContinue
        return frame_in

    def _user_control(self):
        '''Simply stupid sollution how to control state of recogniser.'''
        self.utt_end, self.dialog_end, self.recording = False, False, False
        print('Press r key to toggle recording')
        print('Press c key to exit')
        try:
            while True:
                if msvcrt.kbhit():
                    c = msvcrt.getch().decode("utf-8").lower()
                    if c == 'r':
                        self.recording = not self.recording
                        if self.recording:
                            print('Recording started...')
                            self.frames = []
                        else:
                            print('Recording stopped')
                            self.utt_end = True
                    elif c == 'c':
                        self.dialog_end = True
                        print('\nMarked end of dialogue\n')
                        break
        finally:
            print("""Exit""")

    def run(self):
        x = threading.Thread(target=self._user_control, args=())
        x.start()
        while True:
            time.sleep(0.1)
            if self.utt_end:
                start = time.time()
                self.save_utt_wav(str(start))
                self.utt_end = False
                self.utt_frames = 0
                # zero out frames
                self.frames = []
            if self.dialog_end:
                x.join()
                break

        self.sftp.close()
        self.transport.close()

    def save_utt_wav(self, timestamp):
        filename = timestamp+'-utt.wav'
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setframerate(RATE)
        wf.setsampwidth(self.pin.get_sample_size(FORMAT))
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.send_file(filename)

    def send_file(self,local_file):
        print("Sending file " + local_file)
        start = time.time()
        remote_path = self.host_path + local_file
        this_dir_path = os.path.dirname(os.path.realpath(__file__))
        local_path = os.path.join(this_dir_path, local_file)
        print("Sending from " + local_path)
        print("Sending to " + remote_path)
        self.sftp.put(local_path, remote_path)
        print("File sent in %s secs"% str(time.time() - start) )
        start = time.time() 
        print("Waiting for " + remote_path + ".txt")
        while True and not self.dialog_end:
            time.sleep(0.1)
            try:
                response_file = self.sftp.open(remote_path + ".txt")
                print("Got model response in %s secs"% str(time.time() - start) )
                for line in response_file.readlines():
                    if line == "" or len(line.split(".wav")) < 2:
                        print("Nothing detected")
                    else:
                        detection = line.split(".wav")[1]
                        print("DETECTED TEXT: " + detection)
                        f=open(local_path+".txt", "a")
                        f.write(detection +"\n")
                        f.close()
                break
            except:
                continue

if __name__ == '__main__':
    print('Python args: %s' % str(sys.argv), file=sys.stderr)
    host = str(sys.argv[1])
    host_path = str(sys.argv[2])
    user = str(sys.argv[3])
    
    try:
        password = getpass.getpass()
    except Exception as error:
        print('ERROR', error)

    argv = sys.argv[4:]
    demo = LiveDemo(host, host_path, user, password, argv)
    demo.setup()
    demo.run()
