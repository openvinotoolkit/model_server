import argparse
import json
from ie_serving.server.start import serve as start_server
from ie_serving.models.model import Model


def open_config(path):
    with open(path) as f:
        data = json.load(f)
    print(data)


def parse_config(args):
    print('config_parsed')
    #model = Model(model_name=args.name, model_directory=args.path)
    #start_server(model)
    open_config(path=args.model_path)


def parse_one_model(args):
    model = Model(model_name=args.model_name, model_directory=args.model_path)
    start_server({args.model_name: model})


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('config', help='Allows you to share multiple models using a configuration file')
    parser_a.add_argument('--path', type=str, help='absolute path to json configuration file', required=True)
    parser_a.set_defaults(func=parse_config)

    parser_b = subparsers.add_parser('model', help='Allows you to share one type of model')
    parser_b.add_argument('--model_name', type=str, help='name of the model', required=True)
    parser_b.add_argument('--model_path', type=str, help='absolute path to model,as in tf serving', required=True)
    parser_b.set_defaults(func=parse_one_model)
    args = parser.parse_args()
    parser.parse_args().func(args)


if __name__ == '__main__':
    main()