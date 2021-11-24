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
#! /bin/bash

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
