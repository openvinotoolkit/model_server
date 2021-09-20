#
# INTEL CONFIDENTIAL
# Copyright (c) 2021 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
#
import json
import os
import re
from logging import getLogger

from docker import DockerClient

from ovms.config import ovms_image

logger = getLogger(__name__)

product_version_suffix = os.environ.get("TT_PRODUCT_VERSION_SUFFIX", "oss")

DEFAULT_BUILD_NUMBER = 0
DEFAULT_SHORT_VERSION_NUMBER = "0.0.0"
DEFAULT_RELEASE_TYPE = "snapshot"
DEFAULT_FULL_VERSION_NUMBER = "{}-{}-{}".format(DEFAULT_SHORT_VERSION_NUMBER, product_version_suffix,
                                                DEFAULT_BUILD_NUMBER)

class OvmsInfo:
    """ Retrieves OVMS version and os distname from container. """

    _os_distname = None
    _version = None

    IMAGES = {}

    @property
    def os_distname(self):
        """ This method gets operating system distribution name from given container name. """
        try:
            self._get_info_from_container()
            return self._os_distname
        except Exception as exc:
            logger.error("Couldn't retrieve OVMS os distname "
                         "from.\nException: {}".format(exc))
            raise exc

    @property
    def version(self):
        """ This method gets OVMS version from given container name. """
        try:
            self._get_info_from_container()
            return self._version
        except Exception as exc:
            logger.error("Couldn't retrieve OVMS version "
                         "from.\nException: {}".format(exc))
            raise exc

    @classmethod
    def get(cls):
        """ This method gets OVMS version in dict format. """

        try:
            cls._get_info_from_container()
            return {"version": cls._version}
        except Exception as exc:
            logger.warning("Couldn't retrieve OVMS version "
                           "from.\nException: {}".format(exc))
            return {"version": DEFAULT_FULL_VERSION_NUMBER}

    @classmethod
    def _get_info_from_container(cls):
        """ Run container and get all desired information. """

        if cls._version and cls._os_distname:
            return

        client = DockerClient()
        ovms_container = client.create(
            image=ovms_image,
            entrypoint="sleep",
            command="inf",
        )

        try:
            ovms_container.start()

            # Get ovms container version from metadata
            exit_code, metadata = ovms_container.exec_run(
                cmd="cat /ovms/metadata.json")
            version = json.loads(metadata)["ovms-c"]
            cls._version = version

            # Get ovms container os distname
            exit_code, os_release = ovms_container.exec_run(
                cmd=["bash", "-c", "cat /etc/*-release"])

            os_release = os_release.decode("utf-8")

            os_distname_regex = re.compile("^PRETTY_NAME=")

            os_distname = None
            for line in os_release.splitlines():
                os_distname_match = os_distname_regex.match(line)
                if os_distname_match:
                    os_distname = line[os_distname_match.end():].strip('"\'')
                    break
            cls._os_distname = os_distname

        except Exception as exc:
            logger.error("Couldn't retrieve OVMS info by running container: "
                         "\nException: {}".format(exc))
            raise exc
        finally:
            ovms_container.remove(force=True)
