# Accuracy checker

The script `vehicle_detection_accuracy.py` is running accuracy validation and compares results returned from OVMS and AMS endpoints.
Minor difference might be due to preprocessing and output interpretation or rounding.

Make sure you have installed all its dependencies from [requirements.txt](requirements.txt).

Script should be started from repository root folder after starting AMS image and setting PYTHONPATH:
```bash
docker run -it --rm -p 5000:5000 -p 9000:9000 -e LOG_LEVEL=DEBUG ams:latest /ams_wrapper/start_ams.sh --ams_port=5000 --ovms_port=9000
export PYTHONPATH=${PWD}/:${PWD}/ie_serving/
python extras/ams_wrapper/tests/accuracy/vehicle_detection_accuracy.py
```

The script is comparing the output of preprocessing from AMS implementation via TensorFlow library with OpenCV implementation.
The difference should be negligible while using models from OpenVINO Model Zoo (<=1%).

Next the script is running the calls to OVMS and AMS endpoint. Both results are interpreted and compared in a form of
an array for all detections.

There will be also generated and image `results_combined.jpeg`, which visually presents bounding boxes from both
endpoints painted on the original picture.

  
