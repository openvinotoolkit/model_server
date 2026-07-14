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
from distutils.core import setup
from distutils.extension import Extension
from pathlib import Path

from Cython.Build import cythonize
from pyximport import pyximport


def build_ovms_capi_wrapper(ovms_capi_wrapper_path,
                            capi_package_content_path):
    includes_dir = str(Path(capi_package_content_path, "../include/"))
    lib_dir = str(Path(capi_package_content_path, "lib"))

    # Wrap parameters into distutils generic object
    ovms_capi_ext = Extension(
        name="lib.ovms_capi_wrapper",
        sources=["include/ovms_capi_wrapper.pyx"],
        libraries=["ovms_shared"],  # libovms_shared.so
        language="c",
        runtime_library_dirs=[lib_dir],
        library_dirs=[lib_dir],
        include_dirs=[includes_dir]
    )

    # During this stage pyx+pxd file syntax should be verified
    extensions = cythonize(
        [ovms_capi_ext],
        depfile=True,
        annotate=True,
        compiler_directives={'language_level': "3"}
    )
    return extensions


def prepare_dynamic_load(ovms_capi_wrapper_path, capi_package_content_path):
    extensions = build_ovms_capi_wrapper(ovms_capi_wrapper_path, capi_package_content_path)

    # Little trick to set up automatic ovms_capi Cython compilation on import.
    pyximport.install(
        language_level=3,
        build_in_temp=False,
        build_dir=capi_package_content_path,
        setup_args={
          "ext_modules": extensions
        }
    )
    return


if __name__ == "__main__":
    # Expect valid extracted capi package in `capi_package_content_path`
    capi_cython_extensions = build_ovms_capi_wrapper(ovms_capi_wrapper_path=f"{os.getcwd()}/include/ovms_capi_wrapper.pyx",
                                                     capi_package_content_path=os.getcwd())
    extension = cythonize(capi_cython_extensions)
    setup(ext_modules=extension)
