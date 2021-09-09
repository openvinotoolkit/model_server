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


from collections import defaultdict


def get_server_fixtures_from_pytest_item(item):
    server_fixtures = list(filter(lambda x: "start_server_" in x, item.fixturenames))
    return server_fixtures

def reorder_items_by_fixtures_used(session):
    """
    Reorder test items, group them by fixtures used
    """

    # Keep track how many tests use different container fixtures ('start_server_*')
    server_fixtures_to_tests = defaultdict(lambda: [])

    # For each item (test case) collect used 'start_server_*' fixtures.
    for test in session.items:
        test._server_fixtures = get_server_fixtures_from_pytest_item(test)
        if not test._server_fixtures:
            server_fixtures_to_tests[''].append(test)
        else:
            for fixture in test._server_fixtures:
                server_fixtures_to_tests[fixture].append(test)
    session._server_fixtures_to_tests = server_fixtures_to_tests.copy()

    # Try to order test execution by minimal 'start_server_*' fixtures usage
    ordered_tests = []

    # Choose fixture with min tests assigned to be executed first.
    number_of_tests_lambda = lambda x: len(x[1])
    fixture_with_min_number_of_cases = min(server_fixtures_to_tests.items(), key=number_of_tests_lambda)[0]

    # FIFO queue with processed fixtures
    fixtures_working = [fixture_with_min_number_of_cases]

    while server_fixtures_to_tests:
        current_fixture = fixtures_working[0]
        for test in server_fixtures_to_tests[current_fixture]:
            if test not in ordered_tests:
                ordered_tests.append(test)
                fixtures_used_by_test = get_server_fixtures_from_pytest_item(test)

                # Check all fixtures used by given test.
                for fixture in fixtures_used_by_test:
                    if fixture not in fixtures_working:
                        # Test execute multiple fixtures, add it to queue, it to be processed next.
                        fixtures_working.append(fixture)
        fixtures_working.remove(current_fixture)
        del server_fixtures_to_tests[current_fixture]

        if server_fixtures_to_tests and not fixtures_working:
            # If queue is empty add fixture with least tests (left).
            fixtures_working.append(min(server_fixtures_to_tests.items(), key=number_of_tests_lambda)[0])

    session.items = ordered_tests
    return ordered_tests
