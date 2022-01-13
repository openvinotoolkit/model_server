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

import subprocess
subprocess.run(["/opt/kaldi/src/featbin/feat-to-len", "scp:/tmp/feats.scp", "ark,t:feats_length.txt"])
 
f = open("ivector_online.1.ark.txt", "r")
g = open("ivector_online_ie.ark.txt", "w")
length_file = open("feats_length.txt", "r")
for line in f:
    if "[" not in line:
        for i in range(frame_count):
            line = line.replace("]", " ")
            g.write(line)
    else:
        g.write(line)
        frame_count = int(length_file.read().split(" ")[1])
g.write("]")
f.close()
g.close()
length_file.close()