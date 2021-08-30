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
from collections import defaultdict

from config import artifacts_dir

from utils.helpers import get_xdist_worker_nr, get_xdist_worker_count
from xdist.dsession import LoadScheduling


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


    @staticmethod
    def _get_server_fixtures(item):
        server_fixtures = list(filter(lambda x: "start_server_" in x, item.fixturenames))
        assert server_fixtures
        return server_fixtures

    @staticmethod
    def reorder_items_by_fixtures_used(session):
        server_fixtures_to_item = defaultdict(lambda: [])
        for item in session.items:
            item._server_fixtures = OvmsCLoadScheduling._get_server_fixtures(item)
            if not item._server_fixtures:
                server_fixtures_to_item[''].append(item)
            else:
                for fixture in item._server_fixtures:
                    server_fixtures_to_item[fixture].append(item)
        session._server_fixtures_to_item = server_fixtures_to_item.copy()

        # Try to order test execution minimal 'start_server_*' fixtures working
        ordered_items = []

        # Choose fixture with max tests assigned to be executed first.
        most_cases_lambda = lambda x: len(x[1])
        fixture_with_most_cases = max(server_fixtures_to_item.items(), key=most_cases_lambda)[0]

        fixtures_working = [fixture_with_most_cases] # FIFO queue
        tasks_for_nodes = []
        current_node = []
        while server_fixtures_to_item:
            current_fixture = fixtures_working[0]
            for item in server_fixtures_to_item[current_fixture]:
                if item not in ordered_items:
                    ordered_items.append(item)
                    current_node.append(item)
                    item_fixtures = OvmsCLoadScheduling._get_server_fixtures(item)
                    for it in item_fixtures:
                        # Test execute multiple fixtures  with servers, add fixture to be processed next (out of order).
                        if it not in fixtures_working:
                            fixtures_working.append(it)
                        if item in server_fixtures_to_item:
                            del server_fixtures_to_item[item]
            fixtures_working.remove(current_fixture)
            del server_fixtures_to_item[current_fixture]
            if server_fixtures_to_item and not fixtures_working:
                tasks_for_nodes.append(current_node)
                current_node = []
                fixtures_working.append(max(server_fixtures_to_item.items(), key=most_cases_lambda)[0])

        if current_node:
            tasks_for_nodes.append(current_node)

        session.items = ordered_items

        node_to_test = [[] for i in range(get_xdist_worker_count())]
        for tasks in tasks_for_nodes:
            idx_min_tasks = node_to_test.index(min(node_to_test, key=lambda x: len(x)))
            node_to_test[idx_min_tasks].extend(tasks)

        worker = os.environ.get("PYTEST_XDIST_WORKER", None)
        if worker:
            assigned_tests_path = os.path.join(artifacts_dir, f"assigned_tests_{worker}.xdist")
            with open(assigned_tests_path, "wb") as file:
                test_ids = list(map(lambda x: x.nodeid, node_to_test[get_xdist_worker_nr()]))
                pickle.dump(test_ids, file)
        return ordered_items