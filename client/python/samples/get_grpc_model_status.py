import argparse
from ovmsclient.tfs_compat.grpc.requests import make_status_request
from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client

parser = argparse.ArgumentParser(description='Get information about the status of served models over gRPC interace')
parser.add_argument('--grpc_address',required=False, default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=0, type=int, help='Model version to query. Lists all versions if omitted',
                    dest='model_version')
args = vars(parser.parse_args())

# configuration
address = args.get('grpc_address')   #default='localhost'
port = args.get('grpc_port') #default=9000
model_name = args.get('model_name') #default='resnet'
model_version = args.get('model_version')   #default=0

# creating grpc client
config = {
    "address": address,
    "port": port
}
client = make_grpc_client(config)

# creating status request
request = make_status_request(model_name, model_version)

# getting model status from the server
status = client.get_model_status(request)
status_dict = status.to_dict()
print(status_dict)
