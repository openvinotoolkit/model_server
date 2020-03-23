#!/bin/bash
time printf %s\\n {1..28} | xargs -n 1 -P 28 ./grpc_start_one_stream.sh $* --id
