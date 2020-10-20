#!/bin/bash -x
curl -v -k https://localhost:9443/

echo "GRPC:"
curl -v --cert client.pem --key client.key -k https://localhost:9443/
echo "REST:"
curl -v --cert client.pem --key client.key -k https://localhost:8443/
