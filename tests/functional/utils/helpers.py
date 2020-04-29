#
# Copyright (c) 2020 Intel Corporation
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

from typing import Any


class SingletonMeta(type):
    """
    Metaclass for defining Singleton Classes

    src:
    https://www.datacamp.com/community/tutorials/python-metaclasses

    Singleton Design using a Metaclass

    This is a design pattern that restricts the instantiation of a class to only one object.
    This could prove useful for example when designing a class to connect to the database.
    One might want to have just one instance of the connection class.
    """
    _instances = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
