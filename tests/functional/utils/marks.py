#
# Copyright (c) 2026 Intel Corporation
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

from enum import Enum
from itertools import chain
from re import Pattern
from typing import Union

from _pytest.nodes import Item

from tests.functional.utils.logger import get_logger
from tests.functional.config import repository_name

logger = get_logger(__name__)


class MarkMeta(str, Enum):
    def __new__(cls, mark: str, description: str = None, *args):
        obj = str.__new__(cls, mark)  # noqa
        obj._value_ = mark
        obj.description = description
        return obj

    def __init__(self, *args):
        super(MarkMeta, self).__init__()

    def __hash__(self) -> int:
        return hash(self.mark)

    def __format__(self, format_spec):
        return self.mark

    def __repr__(self):
        return self.mark

    def __str__(self):
        return self.mark

    @classmethod
    def get_by_name(cls, name):
        return name

    @property
    def mark(self):
        return self._value_

    @property
    def marker_with_description(self):
        return f"{self.mark}{f': {self.description}' if self.description is not None else ''}"

    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            return self.mark.__eq__(o)
        return super().__eq__(o)


class ConditionalMark(MarkMeta):
    @classmethod
    def get_conditional_marks_from_item(cls, name, item):
        marks = list(filter(lambda x: x.name == name and x.args is not None, item.keywords.node.own_markers))
        return marks

    @classmethod
    def _params_phrase_match_test_params(cls, params, item):
        """
            Verify if current 'item' parameter match pytest Mark from test case
        """
        if params is None:  # no filtering -> any param will match
            return True
        if hasattr(item.keywords.node, "callspec"):
            test_params = item.keywords.node.callspec.id
            if isinstance(params, Pattern):
                return bool(params.match(test_params))
            elif isinstance(params, str):
                return params == test_params
            else:
                raise AttributeError(f"Unexpected conditional marker params {params}")
        return True

    @classmethod
    def _process_single_entry(cls, entry, item):
        """
            Check if mark 'condition' is meet and item parameters match re/str phrase.
            Then return mark value
        """
        value, condition, params = None, True, None
        if isinstance(entry, str):
            # Simple string do not have condition nor parameters.
            value = entry
        elif isinstance(entry, dict):
            value = entry.get('value')  # required
            condition = entry.get('condition', True)
            params = entry.get('params', None)
        elif isinstance(entry, tuple):
            value, *_optional = entry
            if isinstance(value, list):
                return cls._process_single_entry(value, item)

            if len(_optional) > 0:
                condition = _optional[0]
            if len(_optional) > 1:
                params = _optional[1]
        elif isinstance(entry, list):
            for _element in entry:
                value = cls._process_single_entry(_element, item)
                if value:   # Return first match
                    return value
            return None
        else:
            raise AttributeError(f"Unexpected conditional marker entry {entry}")

        if not condition:
            return None
        return value if cls._params_phrase_match_test_params(params, item) else None

    @classmethod
    def get_all_marks_values_from_item(cls, item, marks):
        mark_values = []
        for mark in marks:
            values = cls.get_all_marker_values_from_item(item, mark)
            if values:
                mark_values.extend(values)
        return mark_values

    @classmethod
    def get_all_marker_values_from_item(cls, item, mark, _args=None):
        """
            Marker can be set as 'str', 'list', 'tuple', 'dict'.
            Process it accordingly and list of values.
        """
        marker_values = []
        args = _args if _args else mark.args
        if isinstance(args, list):
            for entry in args:
                value = cls._process_single_entry(entry, item)
                if not value:
                    continue
                marker_values.append(value)
        elif isinstance(args, tuple):
            value = cls._process_single_entry(args, item)
            if value:
                marker_values.append(value)
        elif isinstance(args, str):
            marker_values.append(args)
        elif isinstance(args, dict):
            for params, value in args.items():
                if not cls._params_phrase_match_test_params(params, item):
                    continue
                if isinstance(value, list):
                    marker_values.extend(value)
                else:
                    marker_values.append(value)
        else:
            raise AttributeError(f"Unrecognized conditional marker {mark}")
        return marker_values

    @classmethod
    def get_markers_values_from_item(cls, item, marks):
        result = []
        for mark in marks:
            result.extend(cls.get_all_marker_values_from_item(item, mark))
        return result

    @classmethod
    def get_markers_values_via_conditional_marker(cls, item, name):
        conditional_marks = cls.get_conditional_marks_from_item(name, item)
        markers_values = cls.get_markers_values_from_item(item, conditional_marks)
        return markers_values

    @classmethod
    def get_mark_from_item(cls, item: Item, conditional_marker_name=None):
        marks = cls.get_markers_values_via_conditional_marker(item, conditional_marker_name)
        if not marks:
            return cls.get_closest_mark(item)
        marks = marks[0]
        return marks

    @classmethod
    def get_closest_mark(cls, item: Item):
        for mark in cls:  # type: 'MarkRunType'
            if item.get_closest_marker(mark.mark):
                return mark
        return None

    @classmethod
    def get_by_name(cls, name):
        mark = list(filter(lambda x: x.value == name, list(cls)))
        return mark[0]


class MarkBugs(ConditionalMark):
    @classmethod
    def get_all_bug_marks_values_from_item(cls, item: Item, conditional_marks=None):
        if not conditional_marks:
            conditional_marks = cls.get_conditional_marks_from_item("bugs", item)
        bugs = cls.get_all_marks_values_from_item(item, conditional_marks)
        return bugs


class MarkGeneral(MarkMeta):
    COMPONENTS = "components"
    REQIDS = "reqids", "Mark requirements tested"


class MarkPriority(MarkMeta):
    HIGH = "priority_high", "Mark as a priority high"
    MEDIUM = "priority_medium", "Mark as a priority medium"
    LOW = "priority_low", "Mark as a priority low"


class MarkSupportedDevices(MarkMeta):
    DEVICES_SUPPORTED = "devices_supported_for_test", "Mark devices that are supported for test"
    DEVICES_NOT_SUPPORTED = "devices_not_supported_for_test", "Mark devices that are not supported for test"


class MarkSupportedOvmsTypes(MarkMeta):
    OVMS_TYPES_SUPPORTED = "ovms_types_supported_for_test", "Mark ovms types that are supported for test"
    OVMS_TYPES_NOT_SUPPORTED = "ovms_types_not_supported_for_test", "Mark ovms types that are not supported for test"


class MarkSupportedOsTypes(MarkMeta):
    OS_TYPES_SUPPORTED = "os_types_supported_for_test", "Mark os types that are supported for test"
    OS_TYPES_NOT_SUPPORTED = "os_types_not_supported_for_test", "Mark os types that are not supported for test"


class MarkTestParameters(MarkMeta):
    MODEL_TYPE = "model_type"
    MODEL_AUX_TYPE = "model_aux_type"
    ALL_MODELS = "all_models"
    MANY_MODELS = "many_models"
    ITERATION_INFO = "iteration_info"
    INPUT_SHAPE = "input_shape"
    INPUT_SHAPE_NO_AUTO = "input_shape_no_auto"
    PLUGIN_CONFIG = "plugin_config"


class MarkConditionalRunType(MarkMeta):
    CONDITIONAL_RUN_TYPE = "conditional_run_type", \
        "Conditionally assign single/non-single run type mark based on device and OS"
    CONDITIONAL_RUN_TYPE_BY_MODEL = "conditional_run_type_by_model", \
        "Conditionally assign run type mark based on model_type membership in model collections"


class MarkRunType(ConditionalMark):
    TEST_MARK_COMPONENT = "component", "run component tests", "component"
    TEST_MARK_SMOKE = "api_smoke", "run api-smoke tests", "api_smoke"
    TEST_MARK_ON_COMMIT = "api_on_commit", "run api-on-commit tests", "api_on-commit"
    TEST_MARK_REGRESSION = "api_regression", "run api-regression tests", "api_regression"
    TEST_MARK_REGRESSION_SINGLE = "api_regression_single", "run api-regression-single tests", "api_regression-single"
    TEST_MARK_REGRESSION_WEEKLY = "api_regression_weekly", "run api-regression-weekly tests", "api_regression-weekly"
    TEST_MARK_REGRESSION_WEEKLY_SINGLE = "api_regression_weekly_single", "run api-regression-weekly-single tests", \
        "api_regression-weekly-single"
    TEST_MARK_ENABLING = "api_enabling", "run api-enabling tests", "api_enabling"
    TEST_MARK_MANUAL = "manual", "run api-manual tests", "api_manual"
    TEST_MARK_STRESS_AND_LOAD = "api_stress_and_load", "run api-stress-and-load tests", "api_stress-and-load"
    TEST_MARK_STRESS_AND_LOAD_SINGLE = "api_stress_and_load_single", "run api-stress-and-load-single tests",\
        "api_stress-and-load-single"
    TEST_MARK_LONG = "api_long", "run api-long tests", "api_long"
    TEST_MARK_PERFORMANCE = "api_performance", "run api-performance tests", "api_performance"
    TEST_MARK_UNSTABLE = "api_unstable", "run api_api_unstable tests", "api_unstable"

    def __init__(self, mark: str, description: str = None, run_type: str = None) -> None:
        super().__init__(self, mark, description)
        self.run_type = f"{repository_name}_{run_type}" if repository_name is not None else run_type

    @classmethod
    def test_mark_to_test_run_type(cls, test_type_mark: Union['MarkRunType', str]):
        if isinstance(test_type_mark, str):
            return MarkRunType(test_type_mark).run_type
        return test_type_mark.run_type

    @classmethod
    def get_test_type_mark(cls, item: Item):
        return cls.get_mark_from_item(item, "test_group")

    @classmethod
    def test_type_mark_to_int(cls, item):
        mark = cls.get_test_type_mark(item)
        assert mark, "Cannot find test_type mark from {item}"
        return list(cls).index(mark)


API_COMPONENT = MarkRunType.TEST_MARK_COMPONENT
API_ON_COMMIT = MarkRunType.TEST_MARK_ON_COMMIT
API_REGRESSION = MarkRunType.TEST_MARK_REGRESSION
API_REGRESSION_SINGLE = MarkRunType.TEST_MARK_REGRESSION_SINGLE
API_REGRESSION_WEEKLY = MarkRunType.TEST_MARK_REGRESSION_WEEKLY
API_REGRESSION_WEEKLY_SINGLE = MarkRunType.TEST_MARK_REGRESSION_WEEKLY_SINGLE
API_ENABLING = MarkRunType.TEST_MARK_ENABLING
API_MANUAL = MarkRunType.TEST_MARK_MANUAL
API_STRESS_AND_LOAD = MarkRunType.TEST_MARK_STRESS_AND_LOAD
API_STRESS_AND_LOAD_SINGLE = MarkRunType.TEST_MARK_STRESS_AND_LOAD_SINGLE
API_LONG = MarkRunType.TEST_MARK_LONG
API_PERFORMANCE = MarkRunType.TEST_MARK_PERFORMANCE
API_UNSTABLE = MarkRunType.TEST_MARK_UNSTABLE


class MarksRegistry(tuple):
    MARKERS = "markers"
    MARK_ENUMS = [MarkGeneral, MarkRunType, MarkPriority, MarkBugs, MarkSupportedDevices, MarkTestParameters,
                  MarkSupportedOvmsTypes, MarkSupportedOsTypes, MarkConditionalRunType]

    def __new__(cls) -> 'MarksRegistry':
        # noinspection PyTypeChecker
        return tuple.__new__(cls, [mark for mark in chain(*cls.MARK_ENUMS)])

    @staticmethod
    def register(pytest_config):
        for mark in MarksRegistry():
            pytest_config.addinivalue_line(MarksRegistry.MARKERS, mark.marker_with_description)
