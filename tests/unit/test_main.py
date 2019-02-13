#
# Copyright (c) 2018-2019 Intel Corporation
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

from ie_serving import main
from unittest import mock
import collections
import pytest
import json


def test_open_config(mocker):
    test_dict = {'config': 'test'}
    test_json = json.dumps(test_dict)
    fake_file_path = 'file/path/mock'
    open_mocker = mocker.patch("ie_serving.main.open",
                               new=mock.mock_open(read_data=test_json))
    actual = main.open_config(fake_file_path)
    open_mocker.assert_called_once_with(fake_file_path, 'r')
    assert actual == test_dict


@pytest.mark.parametrize("should_exit, test_config", [
    (False, {'model_config_list': [{'config': {'base_path': '',
                                               'name': ''}}]}),
    (True, {'model_config_list': [{'config': {'base_path': '', 'test': ''}}]}),
    (True, {'model_config_list': [{'config': ''}]}),
    (True, {'model_config_list': 'test'}),
    (True, {'model_config_list': 5})])
def test_check_config_structure(should_exit, test_config):
    if should_exit:
        with pytest.raises(SystemExit):
            main.check_config_structure(test_config)
    else:
        main.check_config_structure(test_config)


def test_open_config_wrong_json(mocker):
    test_dict = {'config': 'test'}
    fake_file_path = 'file/path/mock'
    open_mocker = mocker.patch("ie_serving.main.open",
                               new=mock.mock_open(read_data=str(test_dict)))
    with pytest.raises(SystemExit):
        main.open_config(fake_file_path)
    open_mocker.assert_called_once_with(fake_file_path, 'r')


@pytest.mark.parametrize("should_fail, model_version_policy, "
                         "exceptions, unexpected_exception",
                         [(False, '{"specific": { "versions":[1,2] }}',
                           None, False),
                          (True, '{"specific": { "test": }}',
                           (SystemExit, json.decoder.JSONDecodeError), False),
                          (True, '{"specific": { "ver":[1,2] }}',
                           (SystemExit, main.ValidationError), False),
                          (False, '{"specific": { "versions":[1,2] }}',
                           (SystemExit, Exception), True)])
def test_parse_one_model(mocker, should_fail, model_version_policy,
                         exceptions, unexpected_exception):
    args = collections.namedtuple('args',
                                  'model_name model_path batch_size'
                                  ' model_version_policy port')
    arguments = args('test', 'test', None, model_version_policy, 9000)
    if should_fail:
        if unexpected_exception:
            builder_mocker = mocker.patch('ie_serving.main.'
                                          'ModelBuilder.build')
            builder_mocker.side_effect = Exception
            with pytest.raises(exceptions):
                main.parse_one_model(arguments)
            assert builder_mocker.called
        else:
            with pytest.raises(exceptions):
                main.parse_one_model(arguments)
    else:
        start_server_mocker = mocker.patch('ie_serving.main.start_server')
        builder_mocker = mocker.patch('ie_serving.main.ModelBuilder.build')
        main.parse_one_model(arguments)
        assert start_server_mocker.called
        assert builder_mocker.called


@pytest.mark.parametrize("args, should_fail", [
    (['python', 'test.py'], True),
    (['python', 'model', '--model_path', 'test_path'], True),
    (['python', 'model', '--model_name', 'test_path', '--model_path',
      'test_path'], False),
    (['python', 'config'], True),
    (['python', 'config', '--config_path', 'test_path'], False),
])
def test_main(mocker, args, should_fail):
    mocker.patch('argparse._sys.argv', args)
    if should_fail:
        with pytest.raises(SystemExit):
            main.main()
    elif 'model' in args:
        arg_parse_mocker = mocker.patch('ie_serving.main.parse_one_model')
        main.main()
        assert arg_parse_mocker.called
    elif 'config' in args:
        arg_parse_mocker = mocker.patch('ie_serving.main.parse_config')
        main.main()
        assert arg_parse_mocker.called
