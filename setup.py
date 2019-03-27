#
# Copyright (c) 2018 Intel Corporation
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

from setuptools import setup


setup(
    name='ie_serving',
    version=0.5,
    description="DLDT inference server",
    long_description="""DLDT inference server""",
    keywords='',
    author_email='',
    packages=['ie_serving', 'ie_serving.server', 'ie_serving.models',
              'ie_serving.tensorflow_serving_api'],
    include_package_data=True,
    zip_safe=False,
    install_requires=["grpcio", "numpy", "protobuf",
                      "tensorflow", "google-cloud-storage", "boto3"],
    entry_points={
        'console_scripts': [
            'ie_serving = ie_serving.main:main',
        ]
    },
)
