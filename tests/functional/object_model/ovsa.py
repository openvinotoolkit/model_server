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
import stat
from datetime import datetime
from pathlib import Path

from cryptography import x509

from tests.functional.utils.core import SelfDeletingFileLock
from tests.functional.utils.inference.communication.base import AbstractCommunicationInterface
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional.utils.ssl import SslCertificates
from tests.functional.constants.ovsa import OVSA

logger = get_logger(__name__)


class OvsaCerts(SslCertificates):
    default_certs = None

    def __init__(
        self,
        nginx_mtls_path: str = OVSA.NGINX_TMP_DIR_PATH,
        server_cert_name: str = OVSA.SERVER_CERT_NAME,
        server_key_name: str = OVSA.SERVER_KEY_NAME,
        client_cert_name: str = OVSA.CLIENT_CERT_NAME,
        client_key_name: str = OVSA.CLIENT_KEY_NAME,
        client_cert_ca_name: str = OVSA.CLIENT_CERT_CA_NAME,
        client_cert_ca_crl_name: str = OVSA.CLIENT_CERT_CA_CRL_NAME,
        dhparam_name: str = OVSA.DHPARAMS_NAME,
        mount_a_dir: bool = False,
    ):
        def get_absolute_path(target_value):
            if not os.path.isabs(target_value):
                target_value = os.path.join(nginx_mtls_path, target_value)
            return target_value

        super().__init__()

        self.certs_loaded = False
        self.nginx_mtls_path = nginx_mtls_path
        self.client_cert_ca_crl_path = get_absolute_path(client_cert_ca_crl_name)
        self.client_cert_ca_path = get_absolute_path(client_cert_ca_name)
        self.client_cert_path = get_absolute_path(client_cert_name)
        self.client_key_path = get_absolute_path(client_key_name)
        self.dhparam_path = get_absolute_path(dhparam_name)
        self.server_cert_path = get_absolute_path(server_cert_name)
        self.server_key_path = get_absolute_path(server_key_name)

        self.mount_a_dir = mount_a_dir

        self.initialize_certificates()

    def initialize_certificates(self):
        try:
            self.load_client_cert_ca_crl(self.client_cert_ca_crl_path)
            self.load_client_cert_ca(self.client_cert_ca_path)
            self.load_client_cert(self.client_cert_path)
            self.load_client_key(self.client_key_path)
            self.load_dhparam(self.dhparam_path)
            self.load_server_cert(self.server_cert_path)
            self.load_server_key(self.server_key_path)
        except (FileNotFoundError, IsADirectoryError) as e:
            logger.info(f"Some certificates missing, expecting to generate new: {e}")
            self.certs_loaded = False
            return
        self.certs_loaded = True

    def create_ovsa_volume_bindings(self) -> dict:
        if self.mount_a_dir:
            return {self.nginx_mtls_path: {"bind": OVSA.CERTS_CONTAINER_PATH, "mode": "ro"}}
        return {
            self.client_cert_ca_crl_path: {"bind": OVSA.CLIENT_CERT_CA_CRL_CONTAINER_PATH, "mode": "ro"},
            self.client_cert_ca_path: {"bind": OVSA.CLIENT_CERT_CA_CONTAINER_PATH, "mode": "ro"},
            self.dhparam_path: {"bind": OVSA.DHPARAMS_CONTAINER_PATH, "mode": "ro"},
            self.server_cert_path: {"bind": OVSA.SERVER_CERT_CONTAINER_PATH, "mode": "ro"},
            self.server_key_path: {"bind": OVSA.SERVER_KEY_CONTAINER_PATH, "mode": "ro"},
        }

    def are_valid(self) -> bool:
        if not self.certs_loaded:
            return False

        for file in [self.server_cert_path, self.client_cert_ca_path, self.client_cert_path, self.server_key_path]:
            file = Path(file)
            if not file.exists():
                logger.warning(f"Certificate file={str(file)} do not exist")
                return False
            if file.stat().st_mode & stat.S_IRGRP == 0:
                logger.warning(f"Certificate file={str(file)} got insufficient reading rights")
                return False

        _cert = x509.load_pem_x509_certificate(self.client_cert)
        _cert_ca = x509.load_pem_x509_certificate(self.client_cert_ca)

        timezone = _cert.not_valid_before_utc.tzinfo
        valid = _cert.not_valid_before_utc < datetime.now(timezone) < _cert.not_valid_after_utc
        valid = valid and (_cert_ca.not_valid_before_utc < datetime.now(timezone) < _cert_ca.not_valid_after_utc)
        return valid

    @staticmethod
    def generate_ovsa_certs(mount_a_dir: bool = False, destination_path=OVSA.NGINX_TMP_DIR_PATH, skip_if_valid=False):
        destination_path = Path(destination_path)
        if not destination_path.exists():
            destination_path.mkdir(parents=True)

        logger.info("Generate Certs")
        with SelfDeletingFileLock(f"{Path(destination_path, '.dir.lock')}") as fl:
            certs = OvsaCerts(mount_a_dir=mount_a_dir, nginx_mtls_path=destination_path)
            if certs.are_valid() and skip_if_valid:
                logger.info("Certificates are still valid and do not require generation")
                return certs

            gen_certs_process = Process()
            path_to_generate_script = os.path.join(destination_path, OVSA.GENERATE_CERTS_SCRIPT_NAME)
            shutil.copytree(OVSA.OVMS_C_NGINX_MTLS_PATH, destination_path, dirs_exist_ok=True)
            gen_certs_process.run_and_check(f"chmod +x {path_to_generate_script}")
            exit_code, stdout, stderr = gen_certs_process.run(
                cmd=path_to_generate_script, cwd=destination_path, timeout=900
            )
            assert exit_code == 0, f"Error generating OVSA certs:\nstdout: {stdout}\nstderr: {stderr}"
            certs = OvsaCerts(mount_a_dir=mount_a_dir, nginx_mtls_path=destination_path)
            assert certs.are_valid()
        return certs

    @staticmethod
    def init_ovsa_certs(_certs=None):
        certs = _certs if _certs else OvsaCerts()
        assert certs.are_valid()
        OvsaCerts.default_certs = certs
        AbstractCommunicationInterface._default_certs = certs
        return certs
