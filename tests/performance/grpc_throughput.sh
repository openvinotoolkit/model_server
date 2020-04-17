#!/bin/bash

CLIENT_COUNT=28
SECONDS=0

ITERATION=-1
for var in "$@"
do
    if [ $ITERATION = 0 ]; then
        ITERATION=$var
    fi

    if [ $var = "--iteration" ]; then
        ITERATION=0
    fi
done

if [ $ITERATION -lt 1 ]; then
    echo "Cannot find --iteration in command line"
    exit 1
fi

time printf %s\\n $(seq 1 $CLIENT_COUNT) | xargs -n 1 -P $CLIENT_COUNT ./grpc_start_one_stream.sh $* --id

FPS=$((CLIENT_COUNT*ITERATION/SECONDS))
echo "$FPS FPS"
