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

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional.config import custom_nodes_path, ovms_c_repo_path
from tests.functional.models import ModelInfo
from tests.functional.constants.ovms import CurrentOvmsType
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths

logger = get_logger(__name__)


@dataclass
class CustomNode(ModelInfo):
    path: str = None
    filename: str = None

    def __post_init__(self):
        super().__post_init__()
        self.ovms_type = CurrentOvmsType.ovms_type

        if self.filename is None:
            self.filename = f"libcustom_node_{self.name}.so"

        if self.path is None:
            if self.ovms_type == OvmsType.KUBERNETES:
                self.path = os.path.join("/config", self.filename)
            else:
                self.path = os.path.join(Paths.CUSTOM_NODE_LIBRARIES_PATH_INTERNAL, self.name, self.filename)

    def get_config(self):
        config = {"name": self.name, "base_path": self.path}
        return config

    def get_parameters(self):
        return None

    def prepare_resources(self, resource_location):
        raise NotImplementedError()

    @classmethod
    def get_volume_mount(cls):
        return {custom_nodes_path: {"bind": Paths.CUSTOM_NODE_LIBRARIES_PATH_INTERNAL, "mode": "ro"}}

    def get_parameters(self):
        return {}

    @staticmethod
    def copy_model_server_src_directory(destination_dir):
        src_dst_path = os.path.join(destination_dir, "src")
        if not os.path.exists(src_dst_path):
            shutil.copytree(os.path.join(ovms_c_repo_path, "src"), src_dst_path)
        third_party_dst_path = os.path.join(destination_dir, "third_party", "opencv")
        if not os.path.exists(third_party_dst_path):
            shutil.copytree(os.path.join(ovms_c_repo_path, "third_party", "opencv"), third_party_dst_path)
        return src_dst_path

    @staticmethod
    def get_custom_nodes_path(image):
        cmd = f"docker cp $(docker create --rm {image}):/{Paths.CUSTOM_NODE_PATH_NAME} ."
        proc = Process()
        cwd = os.path.join(ovms_c_repo_path, "tests", "functional", "utils", "ovms_testing_image")
        proc.run_and_check(cmd, cwd=cwd)
        dst_file_path = os.path.join(cwd, Paths.CUSTOM_NODE_PATH_NAME)
        return dst_file_path


@dataclass
class DevCustomNode(CustomNode):
    src_type: str = "cpp"
    src_dir: str = None
    src_file_path: str = None
    build_successfully: bool = None

    def __post_init__(self):
        super().__post_init__()
        if self.filename is None:
            self.filename = f"libcustom_{self.name}.so"

        if self.src_file_path is None:
            self.src_file_path = os.path.join(self.src_dir, self.name, f"{self.name}.{self.src_type}")

    def prepare_resources(self, resource_location):
        pass
        # src_dst_path = self.copy_model_server_src_directory(resource_location)
        # lib_path = self.get_output_lib_path(src_dst_path)
        # assert lib_path.exists()
        # dst_lib_path = Path(resource_location, Paths.CUSTOM_NODE_PATH_NAME, lib_path.name)
        # os.makedirs(dst_lib_path.parent, exist_ok=True)
        # shutil.copyfile(lib_path, dst_lib_path)
        # return [dst_lib_path.parent]

    def get_output_lib_path(self, tmp_ovms_source_files_path):
        make_cwd = Path(tmp_ovms_source_files_path, Paths.CUSTOM_NODE_PATH_NAME)
        output_lib_path = Path(make_cwd, "lib", self.base_os, self.filename)
        return output_lib_path


@dataclass
class OvmsCCustomNode(DevCustomNode):
    def __post_init__(self):
        self.src_dir = os.path.join(ovms_c_repo_path, "src", Paths.CUSTOM_NODE_PATH_NAME)
        self.src_file_path = os.path.join(self.src_dir, self.name, f"{self.name}.{self.src_type}")
        super().__post_init__()


@dataclass
class OvmsCUnitTestCustomNode(DevCustomNode):
    def __post_init__(self):
        self.src_dir = os.path.join(ovms_c_repo_path, "src", "test", Paths.CUSTOM_NODE_PATH_NAME)
        self.src_file_path = os.path.join(self.src_dir, f"{self.name}.{self.src_type}")
        self.path = os.path.join(Paths.CUSTOM_NODE_LIBRARIES_PATH_INTERNAL, f"libcustom_node_{self.name}.so")
        super().__post_init__()


# Custom nodes located in ovms-tests repo data/custom_nodes/
@dataclass
class OvmsTestDevCustomNode(DevCustomNode):
    def __post_init__(self):
        self.src_dir = os.path.join(ovms_test_repo_path, "data", "ovms_testing_image", Paths.CUSTOM_NODE_PATH_NAME)
        self.src_file_path = os.path.join(self.src_dir, self.name, f"{self.name}.{self.src_type}")
        super().__post_init__()


@dataclass
class CustomNodeEastOcr(OvmsCCustomNode):

    def __init__(self, **kwargs):
        super().__init__(
            name="east_ocr",
            inputs={
                "image": {"shape": [1, 3, 1024, 100], "dtype": np.float32},
                "scores": {"shape": [1, 256, 480, 1], "dtype": np.float32},
                "geometry": {"shape": [1, 256, 480, 5], "dtype": np.float32},
            },
            outputs={
                "text_images": {"shape": [0, 1, 3, 32, 100], "dtype": np.float32},
                "text_coordinates": {"shape": [0, 1, 4], "dtype": np.int32},
                "confidence_levels": {"shape": [0, 1, 1], "dtype": np.float32},
            },
            **kwargs,
        )

    def get_parameters(self):
        return {
            "original_image_width": "1920",
            "original_image_height": "1024",
            "original_image_layout": "NHWC",
            "target_image_layout": "NHWC",
            "target_image_width": "100",
            "target_image_height": "32",
            "confidence_threshold": "0.9",
            "debug": "true",
        }


@dataclass
class CustomNodeVehicles(OvmsCCustomNode):

    def __init__(self, **kwargs):
        super().__init__(
            name="model_zoo_intel_object_detection",
            inputs={
                "image": {"shape": [1, 3, 512, 512], "dtype": np.float32},
                "detection": {"shape": [1, 1, 200, 7], "dtype": np.float32},
            },
            outputs={
                "images": {"shape": [0, 1, 3, 512, 512], "dtype": np.float32},
                "coordinates": {"shape": [0, 1, 4], "dtype": np.int32},
                "confidences": {"shape": [0, 1, 1], "dtype": np.float32},
            },
            **kwargs,
        )

    def get_parameters(self):
        return {
            "original_image_width": "512",
            "original_image_height": "512",
            "target_image_width": "72",
            "target_image_height": "72",
            "original_image_layout": "NHWC",
            "target_image_layout": "NHWC",
            "convert_to_gray_scale": "false",
            "max_output_batch": "100",
            "confidence_threshold": "0.7",
            "debug": "false",
        }


@dataclass
class CustomNodeFaces(OvmsCCustomNode):

    def __init__(self, **kwargs):
        super().__init__(
            name="model_zoo_intel_object_detection",
            inputs={
                "image": {"shape": [1, 3, 600, 400], "dtype": np.float32},
                "detection": {"shape": [1, 1, 200, 7], "dtype": np.float32},
            },
            outputs={
                "images": {"shape": [0, 1, 3, 600, 400], "dtype": np.float32},
                "coordinates": {"shape": [0, 1, 4], "dtype": np.int32},
                "confidences": {"shape": [0, 1, 1], "dtype": np.float32},
            },
            **kwargs,
        )

    def get_parameters(self):
        return {
            "original_image_width": "600",
            "original_image_height": "400",
            "target_image_width": "64",
            "target_image_height": "64",
            "original_image_layout": "NHWC",
            "target_image_layout": "NHWC",
            "convert_to_gray_scale": "false",
            "max_output_batch": "100",
            "confidence_threshold": "0.7",
            "debug": "true",
        }


@dataclass
class CustomNodeImageTransformation(OvmsCCustomNode):

    def __init__(self, original_image_layout="NCHW", target_image_layout="NCHW", **kwargs):
        super().__init__(
            name="image_transformation",
            inputs={"image": {"shape": [1, 3, 224, 224], "dtype": np.float32}},
            outputs={"image": {"shape": [1, 3, 224, 224], "dtype": np.float32}},
            **kwargs,
        )

        self.original_image_layout = original_image_layout
        self.target_image_layout = target_image_layout

    def get_parameters(self):
        return {
            "target_image_width": "224",
            "target_image_height": "224",
            "original_image_color_order": "RGB",
            "target_image_color_order": "RGB",
            "original_image_layout": self.original_image_layout,
            "target_image_layout": self.target_image_layout,
            "scale_values": "[0.003921568627451,0.003921568627451,0.003921568627451]",
            "mean_values": "[-2,-2,-2]",
            "debug": "true",
        }


@dataclass
class CustomNodeDemultiply(OvmsTestDevCustomNode):
    ORIGINAL_DEMULTIPLY_COUNT = 3

    def __init__(self, demultiply_size=None, **kwargs):
        super().__init__(
            name="demultiply",
            inputs={"tensor": {"shape": [1, 3, 224, 224], "dtype": np.float32}},
            outputs={"tensor_out": {"shape": [demultiply_size, 1, 3, 224, 224], "dtype": np.float32}},
            **kwargs,
        )

        self.demultiply_size = demultiply_size

    def get_parameters(self):
        return {"demultiply_size": str(self.demultiply_size)}


@dataclass
class CustomNodeElastic1T(OvmsTestDevCustomNode):

    def __init__(self, input_shape=None, output_shape=None, **kwargs):
        super().__init__(
            name="elastic_in_1t_out_1t",
            inputs={"tensor_in": {"shape": input_shape, "dtype": np.float32}},
            outputs={"tensor_out": {"shape": output_shape, "dtype": np.float32}},
            **kwargs,
        )
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_parameters(self):
        return {"input_shape": str(self.input_shape), "output_shape": str(self.output_shape)}


@dataclass
class CustomNodeDemultiplyGather(OvmsTestDevCustomNode):

    def __init__(self, **kwargs):
        super().__init__(
            name="demultiply_gather",
            inputs={"tensor": {"shape": [4, 1, 10], "dtype": np.float32}},
            outputs={"tensor_out": {"shape": [4, 4, 1, 10], "dtype": np.float32}},
            **kwargs,
        )

    def get_parameters(self):
        return {
            "demultiply_count": "4",
        }


@dataclass
class CustomNodeDifferentOperations(OvmsCUnitTestCustomNode):

    def __init__(self, **kwargs):
        super().__init__(
            name="node_perform_different_operations",
            inputs={
                "input_numbers": {"shape": [1, 10], "dtype": np.float32},
                "op_factors": {"shape": [1, 4], "dtype": np.float32},
            },
            outputs={
                "different_ops_results": {"shape": [0, 1, 10], "dtype": np.float32},
                "factors_results": {"shape": [0, 1, 0], "dtype": np.float32},
            },
            **kwargs,
        )


@dataclass
class CustomNodeChooseMaximum(OvmsCUnitTestCustomNode):

    class Method(Enum):
        MAXIMUM_MINIMUM = "MAXIMUM_MINIMUM"
        MAXIMUM_AVERAGE = "MAXIMUM_AVERAGE"
        MAXIMUM_MAXIMUM = "MAXIMUM_MAXIMUM"

    selection_criteria: Method = None

    def __init__(self, **kwargs):
        super().__init__(
            name="node_choose_maximum",
            inputs={"input_tensors": {"shape": [4, 1, 10], "dtype": np.float32}},
            outputs={"maximum_tensor": {"shape": [1, 10], "dtype": np.float32}},
            **kwargs,
        )

    def __post_init__(self):
        self.selection_criteria = CustomNodeChooseMaximum.Method.MAXIMUM_MINIMUM
        super().__post_init__()

    def get_parameters(self):
        return {"selection_criteria": str(self.selection_criteria.value)}


@dataclass
class CustomNodeDynamicDemultiplex(OvmsCUnitTestCustomNode):

    def __init__(self, **kwargs):
        super().__init__(
            name="node_dynamic_demultiplex",
            inputs={"input_numbers": {"shape": [1, 10], "dtype": np.float32}},
            outputs={"dynamic_demultiplex_results": {"shape": [0, 1, 10], "dtype": np.float32}},
            **kwargs,
        )


@dataclass
class CustomNodeAddSub(OvmsCUnitTestCustomNode):
    def __init__(self, add_value=None, sub_value=None, **kwargs):
        super().__init__(
            name="node_add_sub",
            inputs={"input_numbers": {"shape": [1, 2], "dtype": np.float32}},
            outputs={"output_numbers": {"shape": [1, 2], "dtype": np.float32}},
            **kwargs,
        )
        self.add_value = np.float32(add_value)
        self.sub_value = np.float32(sub_value)

    def __post_init__(self):
        self.src_type = "c"
        super().__post_init__()

    def get_expected_output(self, input_data: dict, client_type: str = None):
        if input_data is None:
            return None

        input_value = list(input_data.values())[0]
        result = input_value + self.add_value - self.sub_value
        return {self.output_names[0]: result}

    def get_parameters(self):
        return {"add_value": str(self.add_value), "sub_value": str(self.sub_value)}
