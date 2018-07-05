import argparse
import json
from ie_serving.server.start import serve as start_server
from ie_serving.models.model import Model


def open_config(path):
    with open(path) as f:
        data = json.load(f)
    return data


def parse_config(args):
    config = open_config(path=args.config_path)
    models = {}
    for model in config['model_config_list']:
        modelin = Model(model_name=model['config']['name'], model_directory=model['config']['base_path'])
        models[model['config']['name']] = modelin
    start_server(models)


def parse_one_model(args):
    model = Model(model_name=args.model_name, model_directory=args.model_path)
    start_server({args.model_name: model})


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('config', help='Allows you to share multiple models using a configuration file')
    parser_a.add_argument('--config_path', type=str, help='absolute path to json configuration file', required=True)
    parser_a.set_defaults(func=parse_config)

    parser_b = subparsers.add_parser('model', help='Allows you to share one type of model')
    parser_b.add_argument('--model_name', type=str, help='name of the model', required=True)
    parser_b.add_argument('--model_path', type=str, help='absolute path to model,as in tf serving', required=True)
    parser_b.set_defaults(func=parse_one_model)
    args = parser.parse_args()
    parser.parse_args().func(args)


if __name__ == '__main__':
    main()