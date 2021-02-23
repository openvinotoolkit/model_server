
yum install gtk+-devel gtk2-devel
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:bazel-out/k8-opt/bin/src/:/opt/intel/openvino/opencv/lib
/opt/rh/devtoolset-8/root/bin/g++ -std=c++17  src/example/CustomNodeResize/test_app.cpp -o test_app -lnode_resize_opencv -Lbazel-bin/src/ -L/opt/intel/openvino/opencv/lib/ -I/opt/intel/openvino/opencv/include/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs


