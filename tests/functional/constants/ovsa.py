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

from tests.functional.config import nginx_certs_dir, ovms_c_repo_path


class OVSA:
    OVMS_C_NGINX_MTLS_PATH = os.path.join(ovms_c_repo_path, "extras", "nginx-mtls-auth")
    NGINX_TMP_DIR_PATH = os.path.join(nginx_certs_dir, "nginx-mtls-auth")
    GENERATE_CERTS_CONFIG_NAME = "openssl_ca.conf"
    GENERATE_CERTS_CONFIG_PATH = os.path.join(NGINX_TMP_DIR_PATH, GENERATE_CERTS_CONFIG_NAME)
    GENERATE_CERTS_SCRIPT_NAME = "generate_certs.sh"
    GENERATE_CERTS_SCRIPT_PATH = os.path.join(NGINX_TMP_DIR_PATH, GENERATE_CERTS_SCRIPT_NAME)

    CLIENT_CERT_CA_CRL_NAME = "client_cert_ca.crl"
    CLIENT_CERT_CA_NAME = "client_cert_ca.pem"
    CLIENT_CERT_NAME = "client.pem"
    CLIENT_KEY_NAME = "client.key"
    DHPARAMS_NAME = "dhparam.pem"
    SERVER_CERT_NAME = "server.pem"
    SERVER_KEY_NAME = "server.key"

    CERTS_CONTAINER_PATH = "/certs"
    CLIENT_CERT_CA_CONTAINER_PATH = os.path.join(CERTS_CONTAINER_PATH, CLIENT_CERT_CA_NAME)
    CLIENT_CERT_CA_CRL_CONTAINER_PATH = os.path.join(CERTS_CONTAINER_PATH, CLIENT_CERT_CA_CRL_NAME)
    DHPARAMS_CONTAINER_PATH = os.path.join(CERTS_CONTAINER_PATH, DHPARAMS_NAME)
    SERVER_CERT_CONTAINER_PATH = os.path.join(CERTS_CONTAINER_PATH, SERVER_CERT_NAME)
    SERVER_KEY_CONTAINER_PATH = os.path.join(CERTS_CONTAINER_PATH, SERVER_KEY_NAME)
