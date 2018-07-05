from grpc.beta import implementations
import numpy
import tensorflow as tf
import cv2
import classes

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


channel = implementations.insecure_channel('localhost', 9001)

stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


request = predict_pb2.PredictRequest()
request.model_spec.name = 'marek'
#request.model_spec.version.value = 1
nparr = numpy.fromfile('/home/marek/zebra.jpg', numpy.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
image = cv2.resize(img, (224, 224))
image = image.reshape((1, 224, 224, 3))
image = numpy.asarray(image, numpy.dtype('<f'))
image =  image.transpose((0, 3, 1, 2))
request.inputs['input'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image, shape=image.shape))
result = stub.Predict(request, 10.0)
print(result)
output = numpy.array(result.outputs['out'].float_val)

# output = tf.contrib.util.make_ndarray(result.outputs['out'])
# print(output)

nu = numpy.array(output)
ma = numpy.argmax(nu)
print("Best classification", classes.imagenet_classess[ma-1])

