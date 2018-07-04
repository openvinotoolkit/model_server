from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import cv2
import classes
import datetime
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2



parser = argparse.ArgumentParser(description='Do requests to ie_serving and tf_serving using images in numpy format')
parser.add_argument('--images_numpy_path', required=True, help='numpy in shape [n,w,h,c]')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service')
args = vars(parser.parse_args())

channel = implementations.insecure_channel(args['grpc_address'], int(args['grpc_port']))

stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
processing_times = np.zeros((0),int)
imgs = np.load(args['images_numpy_path'], mmap_mode='r', allow_pickle=False)
for x in range(10):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.inputs['in'].CopyFrom(
    tf.contrib.util.make_tensor_proto(imgs[x,:,:,:], shape=[1, 224, 224, 3]))
    start_time = datetime.datetime.now()
    result = stub.Predict(request, 10.0)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    #output = numpy.array(result.outputs['out'].float_val)
    output = tf.contrib.util.make_ndarray(result.outputs['out'])

    nu = np.array(output)
    ma = np.argmax(nu)
    print("Best classification", classes.imagenet_classess[ma-1])

print('processing time for all iterations')
for x in processing_times:
    print(x)
print('processing_statistics')
print('average time:',round(np.average(processing_times),1), 'ms; average speed:', round(1000/np.average(processing_times),1),'fps')
print('median time:',round(np.median(processing_times),1), 'ms; median speed:',round(1000/np.median(processing_times),1),'fps')
print('max time:',round(np.max(processing_times),1), 'ms; max speed:',round(1000/np.max(processing_times),1),'fps')
print('min time:',round(np.min(processing_times),1),'ms; min speed:',round(1000/np.min(processing_times),1),'fps')
print('time percentile 90:',round(np.percentile(processing_times,90),1),'ms; speed percentile 90:',round(1000/np.percentile(processing_times,90),1),'fps')
print('time percentile 50:',round(np.percentile(processing_times,50),1),'ms; speed percentile 50:',round(1000/np.percentile(processing_times,50),1),'fps')
print('time standard deviation:',round(np.std(processing_times)))
print('time variance:',round(np.var(processing_times)))
