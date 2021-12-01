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

import setuptools
import subprocess

from setuptools import Command


class BuildApis(Command):

    description = """
    Prepare OVMS supported APIs modules
    """
    user_options = []

    def initialize_options(self):
        ...

    def finalize_options(self):
        ...

    def build_tfs_api(self):
        subprocess.run(["sh", "./scripts/build_tfs_api.sh"], check=True)
    
    def build_ovmsclient_api(self):
        subprocess.run(["sh", "./scripts/build_ovmsclient_api.sh"], check=True)

    def run(self):
        self.build_tfs_api()


from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
     name='ovmsclient',
     version='2021.4.2',
     license='Apache License 2.0',
     author="Intel Corporation",
     author_email="ovms.engineering@intel.com",
     description="OVMS Client",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/openvinotoolkit/model_server/tree/main/client/python/lib",
     cmdclass={
        "build_apis": BuildApis,
    },
     packages=setuptools.find_namespace_packages(include=["ovmsclient*", "tensorflow*", "tensorflow_serving*"]),
     install_requires=["grpcio>=1.21", "protobuf>=3.8", "numpy>=1.16.4", "validators>=0.18.2", "requests >= 2.26.0"],
 )
