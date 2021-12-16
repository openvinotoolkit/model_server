#
# Copyright (c) 2021 Intel Corporation
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
#! /bin/bash
echo "OVMS benchmark client 1.12"

PROXY="no"
OPTIONS="$@"
for OPT in $OPTIONS
do
    if [ "$OPT" = "--proxy" ];
    then PROXY="yes"
    fi
done

SERVER_ADDRESS=""
SERVER_ADDRESS_FLAG=""

OPTIONS="$@"
for OPT in $OPTIONS
do
    if [ -n "$SERVER_ADDRESS_FLAG" ];
    then
	ARGS+=($SERVER_ADDRESS_FLAG)
	SERVER_ADDRESS_FLAG=""
	SERVER_ADDRESS=$OPT
	ARGS+=("localhost")
	continue
    fi
    case $OPT in
	-a | --server_address)
	    SERVER_ADDRESS_FLAG="$OPT"
	    continue;;
    esac
done

if [ -z "$NO_PROXY" ];
then NO_PROXY="$SERVER_ADDRESS"
else NO_PROXY="$NO_PROXY,$SERVER_ADDRESS"
fi
if [ -z "$no_proxy" ];
then no_proxy="$SERVER_ADDRESS"
else no_proxy="$no_proxy,$SERVER_ADDRESS"
fi

CLI_SCRIPT=/ovms_benchmark_client/main.py
if [ "$PROXY" = "no" ];
then
    echo "NO_PROXY=$NO_PROXY no_proxy=$no_proxy python3 $CLI_SCRIPT $@"
    NO_PROXY=$NO_PROXY no_proxy=$no_proxy python3 $CLI_SCRIPT $@
    exit 0
fi

GRPC_PORT=""
REST_PORT=""
SERVER_ADDRESS=""
GRPC_PORT_FLAG=""
REST_PORT_FLAG=""
SERVER_ADDRESS_FLAG=""
declare -a ARGS=($CLI_SCRIPT)

OPTIONS="$@"
for OPT in $OPTIONS
do
    if [ -n "$SERVER_ADDRESS_FLAG" ];
    then
	ARGS+=($SERVER_ADDRESS_FLAG)
	SERVER_ADDRESS_FLAG=""
	SERVER_ADDRESS=$OPT
	ARGS+=("localhost")
	continue
    fi
    if [ -n "$GRPC_PORT_FLAG" ];
    then
	ARGS+=($GRPC_PORT_FLAG)
	GRPC_PORT_FLAG=""
	GRPC_PORT=$OPT
	ARGS+=("11886")
	continue
    fi
    if [ -n "$REST_PORT_FLAG" ];
    then
	ARGS+=($REST_PORT_FLAG)
	REST_PORT_FLAG=""
	REST_PORT=$OPT
	ARGS+=("11887")
	continue
    fi

    case $OPT in
	--proxy) continue;;
	-a | --server_address)
	    SERVER_ADDRESS_FLAG="$OPT"
	    continue;;
	-p | --grpc_port)
	    GRPC_PORT_FLAG="$OPT"
	    continue;;
	-r | --rest_port)
	    REST_PORT_FLAG="$OPT"
	    continue;;
	*) ARGS+=($OPT)	    
    esac
done

IN_HAPROXY_CONFIG="/haproxy/haproxy.cfg"
OUT_HAPROXY_CONFIG="/usr/local/etc/haproxy/haproxy.cfg"
RULE_SERVER_ADDRESS="s/{{server_address}}/$SERVER_ADDRESS/g"
RULE_GRPC_PORT="s/{{grpc_port}}/$GRPC_PORT/g"
RULE_REST_PORT="s/{{rest_port}}/$REST_PORT/g"
RULES="$RULE_SERVER_ADDRESS; $RULE_GRPC_PORT; $RULE_REST_PORT"
sed -e "$RULES" $IN_HAPROXY_CONFIG > $OUT_HAPROXY_CONFIG

# to show haproxy configuration :
# cat $OUT_HAPROXY_CONFIG

if [ -z "$NO_PROXY" ];
then NO_PROXY="$SERVER_ADDRESS"
else NO_PROXY="$NO_PROXY,$SERVER_ADDRESS"
fi
if [ -z "$no_proxy" ];
then no_proxy="$SERVER_ADDRESS"
else no_proxy="$no_proxy,$SERVER_ADDRESS"
fi

# start haproxy:
NO_PROXY=$NO_PROXY no_proxy=$no_proxy haproxy -f $OUT_HAPROXY_CONFIG

echo "python3 ${ARGS[*]}"
python3 ${ARGS[*]} 2> /tmp/xcli-logger
if [ $? -ne 0 ]; then
    cat /tmp/xcli-logger
fi
