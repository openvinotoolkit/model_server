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

from collections import namedtuple

from grpc import ssl_channel_credentials as grpc_ssl_channel_credentials

HttpsCerts = namedtuple("HttpsCerts", ["client_cert", "client_key", "server_cert"])


# pylint: disable=too-many-instance-attributes
class SslCertificates:
    """
    Load ssl certificates.
    """

    def __init__(self):
        self.server_cert_path = None
        self.server_cert = None

        self.server_key_path = None
        self.server_key = None

        self.client_cert_ca_path = None
        self.client_cert_ca = None

        self.client_cert_ca_crl_path = None
        self.client_cert_ca_crl = None

        self.dhparam_path = None
        self.dhparam = None

        self.client_cert_path = None
        self.client_cert = None

        self.client_key_path = None
        self.client_key = None

    def _load_file_into_attribute(self, attribute, filepath):
        setattr(self, f"{attribute}_path", filepath)
        with open(filepath, "rb") as f:
            setattr(self, attribute, f.read())

    def load_server_cert(self, filepath):
        self._load_file_into_attribute("server_cert", filepath)

    def load_server_key(self, filepath):
        self._load_file_into_attribute("server_key", filepath)

    def load_client_cert_ca(self, filepath):
        self._load_file_into_attribute("client_cert_ca", filepath)

    def load_client_cert_ca_crl(self, filepath):
        self._load_file_into_attribute("client_cert_ca_crl", filepath)

    def load_dhparam(self, filepath):
        self._load_file_into_attribute("dhparam", filepath)

    def load_client_cert(self, filepath):
        self._load_file_into_attribute("client_cert", filepath)

    def load_client_key(self, filepath):
        self._load_file_into_attribute("client_key", filepath)

    def get_server_cert_bytes(self):
        return self.server_cert

    def get_client_key_bytes(self):
        return self.client_key

    def get_client_cert_bytes(self):
        return self.client_cert

    def get_client_ca_bytes(self):
        return self.client_cert_ca

    def get_https_cert(self):
        return HttpsCerts(self.client_cert_path, self.client_key_path, self.server_cert_path)

    def get_grpc_ssl_channel_credentials(self):
        return grpc_ssl_channel_credentials(
            root_certificates=self.get_server_cert_bytes(),
            private_key=self.get_client_key_bytes(),
            certificate_chain=self.get_client_cert_bytes(),
        )
