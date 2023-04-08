#
# Copyright (c) 2021 Intel Corporation
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

"""Base inference client for benchmarks"""

import os
import re
import io
import abc
import sys
import time
import math
import json
import random
import traceback
import datetime
import uuid
import png
import grpc
import numpy
import tensorflow
from retry.api import retry_call
try:
    from ovms_benchmark_client.metrics import XSeries
except ModuleNotFoundError:
    from metrics import XSeries


class BaseClient(metaclass=abc.ABCMeta):

    STATEFUL_WIP = 0
    STATEFUL_START = 1
    STATEFUL_STOP = 2

    indent = 1 * " "
    indent2 = 2 * " "
    json_prefix = "JSON-OUTPUT:"
    grpc_options = [
        ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ("grpc.max_receive_message_length", 1024 * 1024 * 1024)
    ]
    retry_setup = {
        "tries": 48,
        "delay": 1
    }

    ###########################################################################
    ##################### Abstract Methods & Properties #######################
    ###########################################################################

    @property
    @abc.abstractmethod
    def status_endpoint(self):
        pass

    @property
    @abc.abstractmethod
    def DTYPE_FLOAT_16(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_FLOAT_32(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_FLOAT_64(self): pass

    @property
    @abc.abstractmethod
    def DTYPE_INT_8(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_INT_16(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_INT_32(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_INT_64(self): pass

    @property
    @abc.abstractmethod
    def DTYPE_UINT_8(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_UINT_16(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_UINT_32(self): pass
    @property
    @abc.abstractmethod
    def DTYPE_UINT_64(self): pass

    @abc.abstractmethod
    def get_stub(self):
        pass

    @abc.abstractmethod
    def show_server_status(self):
        pass

    @abc.abstractmethod
    def get_model_metadata(self, model_name, model_version=None, timeout=60):
        pass

    @abc.abstractmethod
    def prepare_batch_requests(self):
        pass

    @abc.abstractmethod
    def predict(self, request, timeout):
        pass

    ###########################################################################
    ############################# End Of Abstract #############################
    ###########################################################################

    def _get_print_prefix(self):
        if not self.printtime: return ""
        dt = datetime.datetime.now().strftime("%H:%M:%S.%f")
        return f"[{dt}] "

    def print_info(self, *info, force=False):
        if self.printall or force:
            time_prefix = self._get_print_prefix()
            info = ' '.join(map(str, info))
            sys.stdout.write(f"{time_prefix:10}XI {self.worker_id}: {info}\n")

    def print_warning(self, *info):
        time_prefix = self._get_print_prefix()
        info = ' '.join(map(str, info))
        sys.stderr.write(f"{time_prefix:10}XW {self.worker_id}: {info}\n")

    def print_error(self, *info):
        time_prefix = self._get_print_prefix()
        info = ' '.join(map(str, info))
        sys.stderr.write(f"{time_prefix:10}XE {self.worker_id}: {info}\n")

    def make_credentials(self, certs_dir):
        server_cert_path = os.path.join(certs_dir, "server.pem")
        self.print_info(f"secure channel: {server_cert_path}")
        assert os.path.exists(server_cert_path), server_cert_path
        with open(server_cert_path, "rb") as cert:
            server_cert = cert.read()

        client_key_path = os.path.join(certs_dir, "client.key")
        self.print_info(f"secure channel: {client_key_path}")
        assert os.path.exists(client_key_path), client_key_path
        with open(client_key_path, "rb") as key:
            client_key = key.read()

        client_cert_path = os.path.join(certs_dir, "client.pem")
        self.print_info(f"secure channel: {client_cert_path}")
        assert os.path.exists(client_cert_path), client_cert_path
        with open(client_cert_path, "rb") as cert_file:
            client_cert = cert_file.read()

        credentials = grpc.ssl_channel_credentials(
            root_certificates=server_cert,
            certificate_chain=client_cert,
            private_key=client_key)
        return credentials

    def __init__(self, worker_id, address, grpc_port=None, rest_port=None, certs_dir=None):
        assert grpc_port is not None or rest_port is not None
        assert address is not None

        self.address, self.grpc_port, self.rest_port = address, grpc_port, rest_port
        self.full_grpc_address = f"{address}:{grpc_port}"
        self.worker_id = worker_id
        self.final_status = True
        self.jsonout = False
        self.printall = False
        self.printtime = False

        if certs_dir is None:
            func = grpc.insecure_channel
            cargs = (self.full_grpc_address, self.grpc_options)
        else:
            func = grpc.secure_channel
            credentials = self.make_credentials(certs_dir)
            cargs = (self.full_grpc_address, credentials, self.grpc_options)
        try:
            self.channel = retry_call(func, cargs, **self.retry_setup)
            self.print_info("create inference channel...")
        except Exception as err:
            self.print_error(f"worker could not be started: {err}")
            self.final_status = False
            sys.exit(-1)
        try:
            sargs = (self.channel,)
            func = self.get_stub()
            self.stub = retry_call(func, sargs, **self.retry_setup)
            self.print_info("init inference channel...")
        except Exception as err:
            self.print_error(f"worker could not be inited: {err}")
            self.final_status = False
            sys.exit(-1)

        # init empty attributes
        self.stateful_counter = 0
        self.stateful_length = 0
        self.stateful_hop = 0
        self.stateful_id = 1
        self.min_value = 0.0
        self.max_value = 1.0
        self.dump_png = False
        self.xrandom_number = 8
        self.model_name = None
        self.model_version = None
        self.dataset_length = 0
        self.report_warmup = False
        self.forced_shape = None
        self.batchsizes = []
        self.requests = []
        self.outputs = {}
        self.inputs = {}
        self.xdata = {}

    def set_flags(self, jsonout=False, printall=False, printtimed=False, reportwarmup=False):
        self.report_warmup = reportwarmup
        self.printtime = printtimed
        self.printall = printall
        self.jsonout = jsonout
        if self.printall:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    def set_stateful(self, stateful_id, stateful_length=0, stateful_hop=0):
        self.stateful_length = int(stateful_length)
        self.stateful_hop = int(stateful_hop)
        self.stateful_id = int(stateful_id)
        self.stateful_counter = 0

    def set_random_range(self, min_value, max_value):
        self.print_info(f"new random range: {min_value}, {max_value}")
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def set_xrandom_number(self, number):
        assert int(number) > 0
        self.xrandom_number = int(number)

    def set_dump_png(self):
        self.dump_png = True

    def prepare_data(self, data_description, bs_list, dataset_length=None, forced_shape=None):
        self.forced_shape = forced_shape
        if data_description is None:
            data_description = []
        for batch_size in bs_list:
            assert int(batch_size) > 0
            self.batchsizes.append(int(batch_size))
        if dataset_length is None:
            dataset_length = sum(self.batchsizes)
        assert self.batchsizes, "batchsize has to be defined!"
        self.print_info(f"batchsize sequence: {self.batchsizes}")
        super_method_name, parsed_data_description = None, {}
        for data_item in data_description:
            if ":" not in data_item:
                info = "only one definition of super method is allowed"
                assert super_method_name is None, info
                super_method_name = data_item
            else:
                input_name, method_name = data_item.split(":")
                parsed_data_description[input_name] = method_name
        if super_method_name is None:
            super_method_name = "random"

        for input_name in self.inputs:
            method_name = parsed_data_description.get(input_name, super_method_name)
            if "." in method_name:
                self.load_data(input_name, method_name, dataset_length)
            elif method_name in ("random", "urandom"):
                self.generate_urandom_data(input_name, dataset_length)
            elif re.match(r"^png([0-9]*)(-random|-urandom)*$", method_name):
                radius = re.match(r"png([0-9]*)", method_name).group(1)
                try: radius = int(radius)
                except ValueError: radius = 1
                self.set_random_range(0, 255)
                self.print_warning(f"Forced random range (0, 255) for method {method_name}")
                self.generate_png_urandom_data(input_name, dataset_length, radius)
            elif method_name in ("vehicle-jpeg", "vehicle-jpg"):
                self.load_vehicle_jpeg_data(input_name, dataset_length)
            elif method_name in ("png-xrandom", "xrandom", "xrand"):
                self.generate_png_xrandom_data(input_name, dataset_length)
            else: raise ValueError(f"unknown method: {method_name}")
        self.prepare_batch_requests()

    def create_batch_from_data(self, content, content_offset, input_name, batch_length):
        batch = []
        for i in range(batch_length):
            try: batch.append(content[content_offset + i])
            except IndexError:
                if len(batch) == 0:
                    self.print_warning(f"Not enough data in create all batches. " +
                                       f"Number of entries in file ({content_offset + i}) " +
                                       f"is smaller then required batches {self.batchsizes} " +
                                       f"with --dataset_length {self.dataset_length}")
                else:
                    self.print_warning(f"Not enought data in create all batches. " +
                                       f"Cannot fill batch of size {batch_length}. " +
                                       f"Created {len(self.xdata[input_name])} batches. ")
                return None
        return batch

    def load_data(self, input_name, file_name, dataset_length):
        assert os.path.exists(file_name), f"File does not exist: {file_name}"
        content = numpy.load(file_name)
        content_offset = 0
        assert isinstance(content, numpy.ndarray)
        if dataset_length is None or dataset_length <= 0:
            self.dataset_length = content.shape[0]
        else: self.dataset_length = dataset_length

        self.xdata[input_name] = []
        for i in range(self.dataset_length):
            batch_length = self.batchsizes[i % len(self.batchsizes)]
            batch = self.create_batch_from_data(content, content_offset, input_name, batch_length)
            if batch is None: break
            content_offset += len(batch)
            xargs = batch, {"shape": list(content.shape[1:])}
            self.xdata[input_name].append(xargs)
        self.dataset_length = len(self.xdata[input_name])

    def __fix_dataset_length(self, input_name, dataset_length):
        assert input_name in self.inputs.keys()
        if dataset_length is None:
            dataset_length = int(self.batchsizes)
        self.print_info(f"dataset length ({input_name}): {dataset_length}")
        if int(dataset_length) < len(self.batchsizes):
            dataset_length = len(self.batchsizes)
            self.print_warning(
                f"dataset length ({input_name}): fixed to batchsize")
        if int(dataset_length) % len(self.batchsizes) != 0:
            self.print_warning("dataset_length is not matched!")
        self.dataset_length = int(dataset_length)
        return self.dataset_length

    def __fix_shape_and_type(self, input_name):
        if input_name in self.forced_shape:
            shape = self.forced_shape[input_name]
        elif None in self.forced_shape and self.forced_shape[None]:
            shape = self.forced_shape[None]
        else: shape = self.inputs[input_name]["shape"]

        dtype = self.inputs[input_name]["dtype"]
        if dtype == self.DTYPE_FLOAT_32:
            dtype = tensorflow.dtypes.float32
        elif dtype == self.DTYPE_INT_8:
            dtype = tensorflow.dtypes.int8
        elif dtype == self.DTYPE_INT_32:
            dtype = tensorflow.dtypes.int32
        elif dtype == self.DTYPE_INT_64:
            dtype = tensorflow.dtypes.int64
        elif dtype == self.DTYPE_UINT_8:
            dtype = tensorflow.dtypes.uint8
        else: raise ValueError(f"not supported type: {dtype}")
        return shape, dtype

    def __rand_single_png(self, width, height, radius):
        png_image = []
        for _ in range(height):
            row = ()
            for _ in range(width):
                red = random.randint(self.min_value, self.max_value)
                green = random.randint(self.min_value, self.max_value)
                blue = random.randint(self.min_value, self.max_value)
                row = row + (red, green, blue)
            png_image.append(row)
        for _ in range(radius):
            png_tmp = []
            for y in range(height):
                row = ()
                yn = (y + 1) % height
                yp = (y - 1) % height
                for x in range(width):
                    xo = 3 * x
                    xn = (xo + 3) % (3*width)
                    xp = (xo - 3) % (3*width)
                    rsum = png_image[y][xo] + png_image[yp][xo] + png_image[yn][xo] + png_image[y][xn] + png_image[y][xp]
                    gsum = png_image[y][xo+1] + png_image[yp][xo+1] + png_image[yn][xo+1] + png_image[y][xn+1] + png_image[y][xp+1]
                    bsum = png_image[y][xo+2] + png_image[yp][xo+2] + png_image[yn][xo+2] + png_image[y][xn+2] + png_image[y][xp+2]
                    row = row + (int(rsum/5), int(gsum/5), int(bsum/5))
                png_tmp.append(row)
            png_image = png_tmp

        writ = png.Writer(width, height, greyscale=False)
        with io.BytesIO() as bytes_file:
            writ.write(bytes_file, png_image)
            binary_png = bytes_file.getvalue()

        if self.dump_png:
            filename = "/tmp/xcli-" + uuid.uuid4().hex + ".png"
            with open(filename, "wb") as fd:
                writ.write(fd, png_image)
        return binary_png

    def load_vehicle_jpeg_data(self, input_name, dataset_length):
        images_list = ["/data/road1.jpg"]
        self.__fix_dataset_length(input_name, dataset_length)
        shape, _ = self.__fix_shape_and_type(input_name)

        counter = 0
        self.xdata[input_name] = []
        for index in range(self.dataset_length):
            batch_index = index % len(self.batchsizes)
            batch_length = self.batchsizes[batch_index]
            batch = []
            for _ in range(batch_length):
                image_path = images_list[counter % len(images_list)]
                with open(image_path, "rb") as fd:
                    data = fd.read()
                batch.append(data)
                counter += 1
            xargs = batch, {"shape": [batch_length]}
            self.xdata[input_name].append(xargs)

    def generate_png_urandom_data(self, input_name, dataset_length, radius=0):
        self.__fix_dataset_length(input_name, dataset_length)
        shape, dtype = self.__fix_shape_and_type(input_name)
        assert shape[-1] == 3, f"PNG has to have RGB channel - set NHWC - (now: {shape})"
        width, height = shape[-3], shape[-2]

        self.xdata[input_name] = []
        for index in range(self.dataset_length):
            batch_index = index % len(self.batchsizes)
            batch_length = self.batchsizes[batch_index]
            batch = []
            for _ in range(batch_length):
               batch.append(self.__rand_single_png(width, height, radius))
            xargs = batch, {"shape": [batch_length]}
            self.xdata[input_name].append(xargs)

    def generate_urandom_data(self, input_name, dataset_length):
        self.__fix_dataset_length(input_name, dataset_length)
        shape, dtype = self.__fix_shape_and_type(input_name)

        self.xdata[input_name] = []
        for index in range(self.dataset_length):
            batch_index = index % len(self.batchsizes)
            batch_length = self.batchsizes[batch_index]
            xshape = [batch_length, *shape[1:]]
            for dim in xshape:
                self.print_info(f"--> dim: {dim}")
            try:
                batch = numpy.random.uniform(
                    self.min_value, self.max_value, xshape)
                self.print_info(f"Generated data shape: {batch.shape}")
            except ValueError as err:
                self.print_warning(f"data cannot be generated! shape: {xshape}")
                raise ValueError(err)
            xargs = batch, {"dtype": dtype, "shape": batch.shape}
            self.xdata[input_name].append(xargs)

    def __xrand_single_png(self, width, height):
        xvector = []
        for _ in range(self.xrandom_number):
            rgb = []
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            for _ in range(3):
                s = random.randint(0, 3)
                if s == 0: rgb.append(0)
                elif s == 1: rgb.append(255)
                else: rgb.append(random.randint(self.min_value, self.max_value))
            xvector.append((x, y, rgb))
        png_image = []
        for oy in range(height):
            row = ()
            for ox in range(width):
                min_d = math.inf
                min_rgb = (0, 0, 0)
                for x, y, rgb in xvector:
                    d = (ox - x) ** 2 + (oy - y) ** 2
                    if d < min_d: min_rgb, min_d = tuple(rgb), d
                row = row + min_rgb
            png_image.append(row)

        writ = png.Writer(width, height, greyscale=False)
        with io.BytesIO() as bytes_file:
            writ.write(bytes_file, png_image)
            binary_png = bytes_file.getvalue()

        if self.dump_png:
            filename = "/tmp/xcli-" + uuid.uuid4().hex + ".png"
            with open(filename, "wb") as fd:
                writ.write(fd, png_image)
        return binary_png

    def generate_png_xrandom_data(self, input_name, dataset_length):
        self.__fix_dataset_length(input_name, dataset_length)
        shape, dtype = self.__fix_shape_and_type(input_name)
        assert shape[-1] == 3, f"PNG has to have RGB channel - set NHWC - (now: {shape})"
        width, height = shape[-3], shape[-2]

        self.xdata[input_name] = []
        for index in range(self.dataset_length):
            batch_index = index % len(self.batchsizes)
            batch_length = self.batchsizes[batch_index]
            batch = []
            for _ in range(batch_length):
               batch.append(self.__xrand_single_png(width, height))
            xargs = batch, {"shape": [batch_length]}
            self.xdata[input_name].append(xargs)

    def run_workload(self, steps_number, duration, timeout=30, errors_limits=(0,0), warmup=0,
                     window=None, hist_base=10, hist_factor=1000, max_throughput=None, concurr=1):
        assert self.dataset_length, "no data to inference!"
        errors_limit, errors_exposition = errors_limits
        fail_counter, counter = 0, 0

        # ONly for limited throughput
        if max_throughput is not None:
            max_throughput = float(max_throughput)
            single_max_throughput = max_throughput / int(concurr)

        if errors_limit is None: errors_limit = 0
        if steps_number is None: steps_number = 0
        if window is None: window = math.inf

        window_series = XSeries("window", hist_base, hist_factor)
        warmup_series = XSeries("warmup")
        total_series = XSeries("")

        start_workload = time.perf_counter()
        if duration is None:
            stop_workload = math.inf
            error_msg = "one from step number or duration has to be set"
            assert int(steps_number) > 0, error_msg
        else: stop_workload = start_workload + float(duration)
        stop_window = start_workload + float(warmup) + float(window)
        stop_warmup = start_workload + float(warmup)
        self.print_info(f"stop warmup: {stop_warmup}")
        self.print_info(f"stop window: {stop_window}")

        self.print_info("Workload started!")
        while time.perf_counter() <= stop_workload:
            if int(steps_number) > 0 and counter >= int(steps_number): break

            index = counter % self.dataset_length
            batch_size, request = self.requests[index]
            start_step = time.perf_counter()
            try:
                response = self.predict(request, int(timeout))
                end_step = time.perf_counter()
                request_time = end_step - start_step
                current_status = True

                # no response validation to save time

            except Exception:
                end_step = time.perf_counter()
                request_time = end_step - start_step
                self.final_status = False
                current_status = False

                fail_counter += 1

                if errors_exposition in (0, None) or fail_counter <= int(errors_exposition):
                    ex_type, ex_value, ex_tback = sys.exc_info()
                    traceback.print_exception(
                        ex_type, ex_value, ex_tback, file=sys.stderr)
                    current_offset = time.perf_counter() - start_workload
                    info = f"Failed inference ({counter}) after: {current_offset}"
                    self.print_error(info)

                if errors_limit != 0 and fail_counter > int(errors_limit):
                    info = f"Limit of erros {errors_limit} is reached!"
                    self.print_error(info)
                    break
            counter += 1

            total_series.add(current_status, request_time, batch_size)
            if time.perf_counter() < stop_warmup:
                warmup_series.add(current_status, request_time, batch_size)
            elif time.perf_counter() < stop_window:
                window_series.add(current_status, request_time, batch_size)
                if warmup_series.stop():
                    warmup_ts = warmup_series.stop_timestamp
                    self.print_info(f"Warmup normally stopped: {warmup_ts}")
                    window_series.start()
                    window_ts = window_series.start_timestamp
                    self.print_info(f"Window normally start: {window_ts}")
            else:
                if warmup_series.stop():
                    warmup_ts = warmup_series.stop_timestamp
                    self.print_info(f"Warmup stopped: {warmup_ts}")
                    window_series.start()
                    window_ts = window_series.start_timestamp
                    self.print_info(f"Window start: {window_ts}")
                if window_series.stop():
                    window_ts = window_series.stop_timestamp
                    self.print_info(f"Window normally stopped: {window_ts}")

            if max_throughput is not None:
                current_throughput = float(batch_size) / request_time
                if current_throughput > single_max_throughput:
                    extra_throughput = current_throughput - single_max_throughput
                    interval_to_sleep = request_time * extra_throughput / single_max_throughput
                    time.sleep(interval_to_sleep)

        if warmup_series.stop():
            warmup_ts = warmup_series.stop_timestamp
            self.print_info(f"Warmup unnormally stopped: {warmup_ts}")
            window_series.start()
            window_ts = window_series.start_timestamp
            self.print_info(f"Window unnormally start: {window_ts}")
        if window_series.stop():
            window_ts = window_series.stop_timestamp
            self.print_info(f"Window stopped: {window_ts}")
        total_series.stop()

        workload_statistics = {}
        total_stats = total_series.analyze()
        workload_statistics.update(total_stats)
        window_stats = window_series.analyze()
        workload_statistics.update(window_stats)
        if self.report_warmup:
            warmup_stats = warmup_series.analyze()
            workload_statistics.update(warmup_stats)
        if self.printall:
            for key, value in workload_statistics.items():
                self.print_info(f"{key}: {value}")

        if not self.jsonout:
            return workload_statistics
        jout = json.dumps(workload_statistics)
        print(f"{self.json_prefix}###{self.worker_id}###STATISTICS###{jout}")
        return workload_statistics

    def get_status(self):
        return self.final_status
