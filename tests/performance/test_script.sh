#!/bin/bash
NU_CLIENTS=$1
shift 1
sshpass -p user ssh user@10.237.114.207 docker restart ovms-c2 && sshpass -p user ssh user@10.237.114.207 docker exec -e GRPC_SERVERS=$1 -e NIREQ=$2 -e CPU_THROUGHPUT_STREAMS=$3 ovms-c2 /ovms/bazel-bin/ovms --config_path /models/config.json > /dev/null &
sleep 5
ITER=1000
A=$(./grpc_throughput.sh ${NU_CLIENTS} --grpc_address 192.168.0.4 --grpc_port 9179 --images_numpy_path imgs.npy --iteration ${ITER} --batchsize 1 --input_name "data" --report_every ${ITER}  2>&1 | grep -E "\[${NU_CLIENTS}\] Iterations:|real" | tr '\n' ' ')
echo "$A" ${1} ${2} ${3} >> results_${ITER}
