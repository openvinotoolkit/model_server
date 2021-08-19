from collections import defaultdict
from itertools import cycle

import pytest
import _pytest

from xdist.dsession import LoadScheduling


class OvmsCLoadScheduling(LoadScheduling):

    def schedule(self):
        assert self.collection_is_completed

        # Initial distribution already happened, reschedule on all nodes
        if self.collection is not None:
            for node in self.nodes:
                self.check_schedule(node)
            return

        # XXX allow nodes to have different collections
        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return

        # Collections are identical, create the index of pending items.
        self.collection = list(self.node2collection.values())[0]
        self.pending[:] = range(len(self.collection))

        all_tests = sorted(list(self.node2collection.values())[0])
        self.per_test_class_dict = defaultdict(lambda: [])
        for test in all_tests:
            file, test_class, test_name = test.split("::")
            self.per_test_class_dict[test_class].append(test)

        while self.per_test_class_dict:
            for node in self.nodes:
                self._assign_tasks_to_node(node)
                if not self.per_test_class_dict:
                    break

        for node in self.nodes:
            node.shutdown()

    def check_schedule(self, node, duration=0):
        """Maybe schedule new items on the node

        If there are any globally pending nodes left then this will
        check if the given node should be given any more tests.  The
        ``duration`` of the last test is optionally used as a
        heuristic to influence how many tests the node is assigned.
        """
        if node.shutting_down:
            return

    def _assign_tasks_to_node(self, node):
        if not self.per_test_class_dict:
            return
        test_class, test_list = sorted(list(self.per_test_class_dict.items()), key=lambda x: len(x[1]))[-1]
        del self.per_test_class_dict[test_class]
        test_indexes = list(map(lambda x: self.collection.index(x), test_list))
        self.node2pending[node].extend(test_indexes)
        node.send_runtest_some(test_indexes)
        return
