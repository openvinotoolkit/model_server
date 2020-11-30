all:
	. /opt/intel/openvino/bin/setupvars.sh && /usr/bin/g++ -std=gnu++11 -fPIC -shared minimal_cpu_extension.cpp \
		-I /opt/intel/openvino/inference_engine/include/ \
		-I /opt/intel/openvino/deployment_tools/ngraph/include/ \
		-o libminimal_cpu_extension.so

test:
	curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/1/face-detection-retail-0004.xml -o model/1/face-detection-retail-0004.bin
	curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg
	chmod 666 -vR ./images/ ./model/
	/ovms/bazel-bin/src/ovms --cpu_extension /ovms/src/cpu_extension/libminimal_cpu_extension.so --model_path ./model/ --model_name face-detection
