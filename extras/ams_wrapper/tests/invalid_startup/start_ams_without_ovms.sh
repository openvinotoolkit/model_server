#!/bin/bash -e

AMS_PORT=5000

for i in "$@"
do
case $i in
    --ams_port=*)
        AMS_PORT="${i#*=}"
        shift # past argument=value
    ;;
    *)
        exit 0
    ;;
esac
done

. /ie-serving-py/.venv/bin/activate 
python /ams_wrapper/src/wrapper.py --port $AMS_PORT
