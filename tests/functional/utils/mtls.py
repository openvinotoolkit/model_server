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

import config
import tempfile
import subprocess
import shutil

class MtlsKeyChain:
    def __init__(self):
        self.cert_dir = tempfile.mkdtemp(".ovms_test.tmp", "ovms_mtls_certs_")
        open(self.cert_dir + '/openssl_ca.conf', 'wb').write('''
[ ca ]
default_ca = myca

[ crl_ext ]
authorityKeyIdentifier=keyid:always

 [ myca ]
 dir = ./ca
 new_certs_dir = $dir
 unique_subject = no
 certificate = $dir/ca.crt
 database = $dir/certindex
 private_key = $dir/ca.key
 serial = $dir/certserial
 default_days = 730
 default_md = sha512
 policy = myca_policy
 x509_extensions = myca_extensions
 crlnumber = $dir/crlnumber
 default_crl_days = 730

 [ myca_policy ]
 commonName = supplied
 stateOrProvinceName = supplied
 countryName = optional
 emailAddress = optional
 organizationName = supplied
 organizationalUnitName = optional

 [ myca_extensions ]
 basicConstraints = CA:false
 subjectKeyIdentifier = hash
 authorityKeyIdentifier = keyid:always
 keyUsage = digitalSignature,keyEncipherment
 extendedKeyUsage = serverAuth
 crlDistributionPoints = URI:http://localhost/root.crl
 subjectAltName  = @alt_names

 [alt_names]
 DNS.1 = localhost
        '''.encode('utf-8'))
        subprocess.run('cd ' + self.cert_dir + ' && mkdir ca && cd ca && touch certindex && echo 01 > certserial && echo 01 > crlnumber ; ', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl req -x509 -nodes -days 1 -newkey rsa:4096 -keyout server.key -out server.pem -subj "/C=US/CN=localhost"', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl genrsa -out client_cert_ca.key 4096', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl req -x509 -new -nodes -key client_cert_ca.key -sha256 -days 1 -out client_cert_ca.pem -subj "/C=US/CN=localhost"', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl genrsa -out client.key 4096', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl req -new -key client.key -out client.csr -subj "/C=US/CN=client"', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl x509 -req -in client.csr -CA client_cert_ca.pem -CAkey client_cert_ca.key -CAcreateserial -out client.pem -days 1 -sha256', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl ca -config openssl_ca.conf -gencrl -keyfile client_cert_ca.key -cert client_cert_ca.pem -out client_cert_ca.crl ; ', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && openssl dhparam -out dhparam.pem 2048 ', shell=True, check=True)
        subprocess.run('cd ' + self.cert_dir + ' && chmod -R 777 * ; ', shell=True, check=True)
        subprocess.run('ls -lah ' + self.cert_dir, shell=True, check=True)

    def get_docker_args(self):
        return ' -v ' + self.cert_dir + '/server.pem:/certs/server.pem:ro ' + \
               ' -v ' + self.cert_dir + '/server.key:/certs/server.key:ro ' + \
               ' -v ' + self.cert_dir + '/dhparam.pem:/certs/dhparam.pem:ro ' + \
               ' -v ' + self.cert_dir + '/client_cert_ca.crl:/certs/client_cert_ca.crl:ro ' + \
               ' -v ' + self.cert_dir + '/client_cert_ca.pem:/certs/client_cert_ca.pem:ro '

    def docker_py_volumes_dict(self):
         return {
                 self.cert_dir + '/server.pem': {'bind': '/certs/server.pem', 'mode': 'ro'},
                 self.cert_dir + '/server.key': {'bind': '/certs/server.key', 'mode': 'ro'},
                 self.cert_dir + '/dhparam.pem': {'bind': '/certs/dhparam.pem', 'mode': 'ro'},
                 self.cert_dir + '/client_cert_ca.crl': {'bind': '/certs/client_cert_ca.crl', 'mode': 'ro'},
                 self.cert_dir + '/client_cert_ca.pem': {'bind': '/certs/client_cert_ca.pem', 'mode': 'ro'} }

    def server_cert_file_path(self):
        return self.cert_dir + '/server.pem'

    def server_cert_as_bytes(self):
        return open(self.server_cert_file_path(), 'rb').read()

    def client_cert_file_path(self):
        return self.cert_dir + '/client.pem'

    def client_cert_as_bytes(self):
        return open(self.client_cert_file_path(), 'rb').read()

    def client_key_file_path(self):
        return self.cert_dir + '/client.key'

    def client_key_as_bytes(self):
        return open(self.client_key_file_path(), 'rb').read()

    def get_requests_certs(self):
        return ( self.client_cert_file_path() , self.client_key_file_path() )

    def get_requsts_verify(self):
        return self.server_cert_file_path()

default_mtls_keychain = None
is_mtls_enabled = False

def is_enabled():
    global is_mtls_enabled
    return is_mtls_enabled

def set_mtls_enabled(on_off):
    global is_mtls_enabled
    is_mtls_enabled = on_off

def get_mtls_keychain():
    global default_mtls_keychain
    if default_mtls_keychain == None:
        default_mtls_keychain = MtlsKeyChain()
    return default_mtls_keychain


