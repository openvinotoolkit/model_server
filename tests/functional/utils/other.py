from collections import defaultdict


def get_server_fixtures_from_pytest_item(item):
    server_fixtures = list(filter(lambda x: "start_server_" in x, item.fixturenames))
    return server_fixtures

def reorder_items_by_fixtures_used(session):
    """
    Reorder test items, group them by fixtures used
    """

    # Keep track how many tests use different container fixtures ('start_server_*')
    server_fixtures_to_item = defaultdict(lambda: [])

    # For each item (test case) collect used 'start_server_*' fixtures.
    for item in session.items:
        item._server_fixtures = get_server_fixtures_from_pytest_item(item)
        if not item._server_fixtures:
            server_fixtures_to_item[''].append(item)
        else:
            for fixture in item._server_fixtures:
                server_fixtures_to_item[fixture].append(item)
    session._server_fixtures_to_item = server_fixtures_to_item.copy()

    # Try to order test execution by minimal 'start_server_*' fixtures usage
    ordered_items = []

    # Choose fixture with min tests assigned to be executed first.
    most_cases_lambda = lambda x: len(x[1])
    fixture_with_most_cases = min(server_fixtures_to_item.items(), key=most_cases_lambda)[0]

    # FIFO queue with processed fixtures
    fixtures_working = [fixture_with_most_cases]

    while server_fixtures_to_item:
        current_fixture = fixtures_working[0]
        for item in server_fixtures_to_item[current_fixture]:
            if item not in ordered_items:
                ordered_items.append(item)
                item_fixtures = get_server_fixtures_from_pytest_item(item)

                # Check all fixtures used by given test.
                for it in item_fixtures:
                    if it not in fixtures_working:
                        # Test execute multiple fixtures, add it to queue, it to be processed next.
                        fixtures_working.append(it)
                    # Remove test reference
                    if item in server_fixtures_to_item:
                        del server_fixtures_to_item[item]
        fixtures_working.remove(current_fixture)
        del server_fixtures_to_item[current_fixture]

        if server_fixtures_to_item and not fixtures_working:
            # If queue is empty add fixture with most tests (left).
            fixtures_working.append(min(server_fixtures_to_item.items(), key=most_cases_lambda)[0])

    session.items = ordered_items
    return ordered_items
