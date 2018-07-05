import argparse
from ie_serving.server.start import serve as start_server
from ie_serving.models.model import Model


def parse_config(args):
    print('config_parsed')
    model = Model(model_name=args.name, model_directory=args.path)
    start_server(model)


def parse_one_model(args):
    model = Model(model_name=args.model_name, model_directory=args.model_path)
    start_server({args.model_name: model})


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('config', help='Models in configmap',)
    parser_a.add_argument('--path', type=str, help='configmap path', required=True)
    parser_a.set_defaults(func=parse_config)

    parser_b = subparsers.add_parser('model', help='One model')
    parser_b.add_argument('--model_name', type=str, help='bar help', required=True)
    parser_b.add_argument('--model_path', type=str, help='bar help', required=True)
    parser_b.set_defaults(func=parse_one_model)
    args = parser.parse_args()
    parser.parse_args().func(args)


if __name__ == '__main__':
    main()