#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import sys
from os import path

from tests.functional.utils.logger import get_logger


def parse_args(args=sys.argv):
    """Parse cli arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c",
                        "--config",
                        dest="config_path",
                        metavar="path",
                        default="./reservation_manager.json",
                        help="Path to YAML configuration file")

    parser.add_argument("-i",
                        "--init-config",
                        default=False,
                        action="store_true",
                        help="Init default config under path "
                        "pointed by --config")

    parser.add_argument("--pool-range-start",
                        type=int,
                        metavar="port",
                        help="Port number starting "
                        "allowed port reservation range")

    parser.add_argument("--pool-range-stop",
                        type=int,
                        metavar="port",
                        help="Port number ending "
                        "allowed port reservation range")

    parser.add_argument("--pool-part-size",
                        type=int,
                        metavar="number",
                        help="Port pool part size")

    parser.add_argument("--locks-dir",
                        type=str,
                        metavar="path",
                        help="Directory where lock files will be present")

    parser.add_argument("--reservation-file-json",
                        type=str,
                        metavar="filepath",
                        default="./reservation.json",
                        help="Path to file where the reservation information "
                        "will be saved in JSON file")

    parser.add_argument("--reservation-file-env",
                        type=str,
                        metavar="filepath",
                        default="./reservation.env",
                        help="Path to file where the reservation information "
                        "will be saved in ENV Variables format")

    parser.add_argument("--keep-env",
                        default=False,
                        action="store_true",
                        help="Keep environment variables when running "
                        "command, pass with configured reservation envs.")

    parser.add_argument("-l",
                        "--log-level",
                        choices=[
                            "DEBUG",
                            "INFO",
                            "WARNING",
                        ],
                        default="INFO",
                        help="Enable debug mode")

    # Create subparser for functions
    subparsers = parser.add_subparsers(
        help="Action to be taken with %(prog)s",
        dest="reservation_action",
    )

    # subparser for create
    parser_for_create = subparsers.add_parser(
        "create",
        help="Create a long living reservation with reservation "
        "files for JSON and Shell Environments. See --reservation-file-json "
        "and --reservation-file-env.",
    )
    parser_for_create.add_argument(
        dest="reservation_create",
        action="store_true",
    )

    # subparser for remove
    parser_for_remove = subparsers.add_parser(
        "remove",
        help="Delete reservation pointed by --reservation-file-json on host.",
    )
    parser_for_remove.add_argument(
        dest="reservation_remove",
        action="store_true",
    )

    # subparser for cleanup
    parser_for_cleanup = subparsers.add_parser(
        "cleanup",
        help="Delete all reservations on host.",
    )
    parser_for_cleanup.add_argument(
        dest="reservation_cleanup",
        action="store_true",
    )

    # subparser for command
    parser_for_command = subparsers.add_parser(
        "command",
        help="Command to be run. Reservation will be "
        "automatically cleaned up after command completion. ",
    )
    parser_for_command.add_argument(
        dest="reservation_command",
        metavar="cmd_arg",
        nargs="*",
        help="example: "
        "'%(prog)s -- pytest --mypytest opt'",
    )

    args = parser.parse_args(args=args)

    log = get_logger(__name__)

    log.info(f"Arguments: {args}")

    # Copy default configuration file if requested
    if args.init_config:
        if path.isfile(args.config_path):
            raise Exception(
                f"Can't init config in '{args.config_path}', file exists.")
       # copyfile(paths.package_data_conf_filepath, args.config_path)
        log.info("Config file initialized")

    # # If configuration not exists in config_path, use default from package
    # if not path.exists(args.config_path):
    #     args.config_path = paths.package_data_conf_filepath

    return args
