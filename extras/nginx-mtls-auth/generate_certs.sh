#!/bin/bash -x
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
set -e

echo "Removing old certs..."
rm -rf ./ca/ client_cert_ca.pem server.pem server.key dhparam.pem client_cert_ca.crl

echo "Generating certificates..."
echo "===================================================================================================================================================="
echo "WARNING: For development and testing only. Please follow your organization security practices on handling and distribution of cryptography material."
echo "===================================================================================================================================================="
openssl req -x509 -nodes -days 1 -newkey rsa:4096 -keyout server.key -out server.pem -subj "/C=US/CN=localhost"

rm -rf ca && mkdir ca && cd ca && touch certindex && echo 01 > certserial && echo 01 > crlnumber ; cd -

openssl genrsa -out client_cert_ca.key 4096
openssl req -x509 -new -nodes -key client_cert_ca.key -sha512 -days 1 -out client_cert_ca.pem -subj "/C=US/CN=localhost"
openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr -subj "/C=US/CN=client"
openssl x509 -req -in client.csr -CA client_cert_ca.pem -CAkey client_cert_ca.key -CAcreateserial -out client.pem -days 1 -sha512
openssl ca -config openssl_ca.conf -gencrl -keyfile client_cert_ca.key -cert client_cert_ca.pem -out client_cert_ca.crl
openssl dhparam -out dhparam.pem 2048

chmod 666 client_cert_ca.pem server.pem server.key dhparam.pem client_cert_ca.crl

echo "Key material is ready."

