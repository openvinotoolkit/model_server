# Accuracy checker

The script `accuracy_checker.py` is running accuracy validation and compares results returned from OVMS and AMS endpoints.
Minor difference might be due to preprocessing and output interpretation or rounding.

Make sure you have installed all its dependencies from [requirements.txt](requirements.txt).

Script should be started from repository root folder after starting AMS image and setting PYTHONPATH:
```bash
docker run -it --rm -p 5000:5000 -p 9000:9000 -e LOG_LEVEL=DEBUG ams:latest /ams_wrapper/start_ams.py --ams_port=5000 --ovms_port=9000
export PYTHONPATH=${PWD}/:${PWD}/ie_serving/
python extras/ams_wrapper/tests/accuracy/accuracy_checker.py --model_type [detection/classifier] --ams_endpoint <endpoint name> --ovms_model_name <model name> --image_path <image pah> --model_json <JSON file with model description>
```

As an output there is comparison between OVMS and AMS results. For `detection` type models count of detections is compared.
For `classifier` models type script compares class and confidence.
Additionally for `detection` type models script produces two images `ams_results.jpg` and `ovms_results.jpg`, which visually presents bounding boxes from both endpoints painted on the original picture.

  

