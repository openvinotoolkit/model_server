#
# Copyright (c) 2019 Intel Corporation
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
import os
import pickle
import time

from config import artifacts_dir
from xdist.dsession import LoadScheduling
from xdist.scheduler import EachScheduling


class OvmsCLoadScheduling(LoadScheduling):
    #
    # def mark_test_complete(self, node, item_index, duration=0):
    #     super().mark_test_complete(node, item_index, duration)
    #     foo = 0
    #
    # def check_schedule(self, node, duration=0):
    #     foo = 0


    # def add_node_collection(self, node, collection):
    #     path_to_test_list = os.path.join(artifacts_dir, f"assigned_tests_{node.workerinput['workerid']}.xdist")
    #     with open(path_to_test_list, "rb") as file:
    #         test_list = pickle.load(file)
    #     self.node2pending[node] = test_list
    #     #self.node2collection[node] = test_list
    #     self.node2collection[node] = list(collection)

    def check_schedule(self, node, duration=0):
        xxx = 0

    def schedule(self):
        # Collections are identical, create the index of pending items.
        self.collection = list(self.node2collection.values())[0]
        self.pending[:] = range(len(self.collection))

        for node in self.nodes:
           # logger.error(f"artifacts_dir")
            path_to_test_list = os.path.join(artifacts_dir, f"assigned_tests_{node.workerinput['workerid']}.xdist")
            with open(path_to_test_list, "rb") as file:
                node.assigned_test_list = pickle.load(file)
                self._assign_tests_to_node(node, node.assigned_test_list)

        for node in self.nodes:
            node.shutdown()

    def _assign_tests_to_node(self, node, tests):
        test_indexes = list(map(lambda x: self.collection.index(x), tests))
        self.node2pending[node].extend(test_indexes)
        node.send_runtest_some(test_indexes)
        return
