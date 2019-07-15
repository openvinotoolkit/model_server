def test_get_model_status_successful(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/test',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 200


def test_get_model_status_successful_with_specific_version(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/test/versions/2',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 200


def test_get_model_status_wrong_model(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/fake_model',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 404


def test_get_model_status_wrong_version(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/test/versions/5',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 404
