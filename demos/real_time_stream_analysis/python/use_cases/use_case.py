#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from abc import ABC, abstractstaticmethod
import numpy as np

class UseCase(ABC):

    @abstractstaticmethod
    def support_visualization() -> bool:
        # return True, if you plan to implement visualize() method and use this use case with visualizer
        # otherwise return False
        pass

    @abstractstaticmethod
    def visualize(frame: np.ndarray, inference_result: np.ndarray) -> np.ndarray:
        # frame is the original frame taken from the stream and inference_result is the output from the model
        # Implement necessary operations on the original frame to visualize inference_results and return final frame that will be displayed
        pass

    @abstractstaticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        # This method is called between capturing frame from the stream and passing it to inference executor. 
        # Implement necessary operations on the original data to adjust it to model input. For example - resize, change layout, change color model etc.
        pass

    @abstractstaticmethod
    def postprocess(inference_result: np.ndarray):
        # This method is called when inference result on the frame is available.
        # Implement results processing and actions that need to be performed depending on the inference_results.
        pass