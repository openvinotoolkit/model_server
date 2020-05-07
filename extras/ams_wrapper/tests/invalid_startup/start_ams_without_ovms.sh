#!/bin/bash -e

export FILE_SYSTEM_POLL_WAIT_SECONDS=0
AMS_PORT=5000
OVMS_PORT=9000

for i in "$@"
do
case $i in
    --help=*)
        echo "$help_message"
        exit 0
    ;;
    --ams_port=*)
        AMS_PORT="${i#*=}"
        shift # past argument=value
    ;;
    --ovms_port=*)
        OVMS_PORT="${i#*=}"
        shift # past argument=value
    ;;
    *)
        echo "$help_message"
        exit 0
    ;;
esac
done

. /ie-serving-py/.venv/bin/activate 
python /ams_wrapper/src/wrapper.py --port $AMS_PORT
