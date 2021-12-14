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

import time
import math
import re

class XMetrics(dict):
    def __init__(self, init_data=None, submetrics=1):
        if init_data is None: init_data = {}
        dict.__init__(self, init_data)
        self["submetrics"] = submetrics
        self.__mean_counter_cache = {}
        self.__mean_sum_cache = {}

    def calc_sum(self, results, key):
        if key not in results: return
        if key not in self: self[key] = 0
        self[key] += results[key]

    def calc_max(self, results, key):
        if key not in results: return
        if key not in self:
            self[key] = results[key]
        elif self[key] < results[key]:
            self[key] = results[key]

    def calc_min(self, results, key):
        if key not in results: return
        if key not in self:
            self[key] = results[key]
        elif self[key] > results[key]:
            self[key] = results[key]

    def calc_mean(self, results, key):
        if key not in results: return
        if key not in self:
            self.__mean_counter_cache[key] = 0
            self.__mean_sum_cache[key] = 0.0
        self.__mean_counter_cache[key] += 1
        self.__mean_sum_cache[key] += results[key]
        self[key] = self.__mean_sum_cache[key] / self.__mean_counter_cache[key]

    def __iadd_series(self, prefix, other):
        self.calc_sum(other, f"{prefix}total_batches")
        self.calc_sum(other, f"{prefix}total_frames")
        self.calc_sum(other, f"{prefix}pass_batches")
        self.calc_sum(other, f"{prefix}fail_batches")
        self.calc_sum(other, f"{prefix}pass_frames")
        self.calc_sum(other, f"{prefix}fail_frames")

        self.calc_sum(other, f"{prefix}netto_batch_rate")
        self.calc_sum(other, f"{prefix}netto_frame_rate")
        self.calc_mean(other, f"{prefix}mean_latency")
        self.calc_max(other, f"{prefix}pass_max_latency")
        self.calc_max(other, f"{prefix}fail_max_latency")

        self.calc_min(other, f"{prefix}start_timestamp")
        self.calc_max(other, f"{prefix}stop_timestamp")

        self[f"{prefix}total_duration"] = self[f"{prefix}stop_timestamp"] - self[f"{prefix}start_timestamp"]
        self[f"{prefix}brutto_batch_rate"] = self[f"{prefix}pass_batches"] / self[f"{prefix}total_duration"]
        self[f"{prefix}brutto_frame_rate"] = self[f"{prefix}pass_frames"] / self[f"{prefix}total_duration"]

        self.calc_mean(other, f"{prefix}batch_passrate")
        self.calc_mean(other, f"{prefix}frame_passrate")
        self.calc_mean(other, f"{prefix}pass_mean_latency")
        self.calc_mean(other, f"{prefix}fail_mean_latency")

        N = int(self["submetrics"])
        assert N > 1, "need at least 1 client/concurrency"
        if f"{prefix}mean_latency" not in self: self[f"{prefix}mean_latency"] = 0.0
        if f"{prefix}mean_latency2" not in self: self[f"{prefix}mean_latency2"] = 0.0
        if f"{prefix}mean_latency" not in other: other[f"{prefix}mean_latency"] = 0.0
        if f"{prefix}mean_latency2" not in other: other[f"{prefix}mean_latency2"] = 0.0
        mean_lat2 = ((N - 1) * self[f"{prefix}mean_latency2"] + other[f"{prefix}mean_latency2"]) / N
        mean_lat = ((N - 1) * self[f"{prefix}mean_latency"] + other[f"{prefix}mean_latency"]) / N
        try:
            self[f"{prefix}stdev_latency"] = math.sqrt(mean_lat2 - mean_lat ** 2)
            self[f"{prefix}cv_latency"] = self[f"{prefix}stdev_latency"] / mean_lat
        except:
            self[f"{prefix}stdev_latency"] = 0.0
            self[f"{prefix}cv_latency"] = 0.0
        self[f"{prefix}mean_latency2"] = mean_lat2
        self[f"{prefix}mean_latency"] = mean_lat

        if f"{prefix}pass_mean_latency" not in self: self[f"{prefix}pass_mean_latency"] = 0.0
        if f"{prefix}pass_mean_latency2" not in self: self[f"{prefix}pass_mean_latency2"] = 0.0
        if f"{prefix}pass_mean_latency" not in other: other[f"{prefix}pass_mean_latency"] = 0.0
        if f"{prefix}pass_mean_latency2" not in other: other[f"{prefix}pass_mean_latency2"] = 0.0
        mean_lat2 = ((N - 1) * self[f"{prefix}pass_mean_latency2"] + other[f"{prefix}pass_mean_latency2"]) / N
        mean_lat = ((N - 1) * self[f"{prefix}pass_mean_latency"] + other[f"{prefix}pass_mean_latency"]) / N
        try:
            self[f"{prefix}pass_stdev_latency"] = math.sqrt(mean_lat2 - mean_lat ** 2)
            self[f"{prefix}pass_cv_latency"] = self[f"{prefix}pass_stdev_latency"] / mean_lat
        except:
            self[f"{prefix}pass_stdev_latency"] = 0.0
            self[f"{prefix}pass_cv_latency"] = 0.0
        self[f"{prefix}pass_mean_latency2"] = mean_lat2
        self[f"{prefix}pass_mean_latency"] = mean_lat

        if f"{prefix}fail_mean_latency" not in self: self[f"{prefix}fail_mean_latency"] = 0.0
        if f"{prefix}fail_mean_latency2" not in self: self[f"{prefix}fail_mean_latency2"] = 0.0
        if f"{prefix}fail_mean_latency" not in other: other[f"{prefix}fail_mean_latency"] = 0.0
        if f"{prefix}fail_mean_latency2" not in other: other[f"{prefix}fail_mean_latency2"] = 0.0
        mean_lat2 = ((N - 1) * self[f"{prefix}fail_mean_latency2"] + other[f"{prefix}fail_mean_latency2"]) / N
        mean_lat = ((N - 1) * self[f"{prefix}fail_mean_latency"] + other[f"{prefix}fail_mean_latency"]) / N
        try:
            self[f"{prefix}fail_stdev_latency"] = math.sqrt(mean_lat2 - mean_lat ** 2)
            self[f"{prefix}fail_cv_latency"] = self[f"{prefix}pass_stdev_latency"] / mean_lat
        except:
            self[f"{prefix}fail_stdev_latency"] = 0.0
            self[f"{prefix}fail_cv_latency"] = 0.0
        self[f"{prefix}fail_mean_latency2"] = mean_lat2
        self[f"{prefix}fail_mean_latency"] = mean_lat

        for key in other.keys():
            if re.search(f"^{prefix}hist_latency_", key):
                try: self[key] += other[key]
                except KeyError: self[key] = other[key]

    def __iadd__(self, other):
        self.calc_sum(other, "submetrics")
        self.__iadd_series("warmup_", other)
        self.__iadd_series("window_", other)
        self.__iadd_series("", other)
        return self


    def calc_quantile_value(self, latlist, total, quantile):
        assert quantile >= 0 and quantile <= 1
        minimal_samples_number = 6
        if (1 - quantile) * total < minimal_samples_number:
            return None, 0

        gen = reversed(sorted(latlist))
        current, req = total, total * quantile
        for num, val, fval_o, fval_e in gen:
            current -= val
            if current <= req: break

        offset = req - current
        error = fval_e - fval_o
        delta = error * float(offset) / val
        return fval_o + delta, error

    
    def recalculate_quantiles(self, prefix, base, factor, quantile_list):
        total = 0
        latlist = []
        for key, val in self.items():
            if re.search(f"^{prefix}hist_latency_", key):
                num = int(key.replace(f"{prefix}hist_latency_", ""))
                fval_e = ((num+1) ** (1/base)) / factor
                fval_o = (num ** (1/base)) / factor 
                latlist.append((num, val, fval_o, fval_e))
                total += int(val)

        keys_to_rm = []
        for key in self.keys():
            if re.search(f"^{prefix}hist_latency_", key):
                keys_to_rm.append(key)
        for key in keys_to_rm: del self[key]
                
        for index, quantile in enumerate(quantile_list):
            latency, error = self.calc_quantile_value(latlist, total, float(quantile))
            if latency is None: latency = ""
            self[f"qos_quantile_{index}"] = quantile
            self[f"qos_latency_{index}"] = latency
            self[f"qos_error_{index}"] = error

    
class XSeries:
    def __init__(self, prefix, hist_base=None, hist_factor=None):
        self.start_timestamp = None
        self.stop_timestamp = None
        self.prefix = prefix

        self.hist_register = {}
        if hist_factor is not None:
            self.hist_factor = float(hist_factor)
        else: self.hist_factor = None        
        if hist_base is not None:
            self.hist_base = float(hist_base)
        else: self.hist_base = None

        self.pass_counter = 0
        self.fail_counter = 0
        self.counter = 0

        self.pass_xcounter = 0
        self.fail_xcounter = 0
        self.xcounter = 0

        self.pass_latency_sum = 0.0
        self.fail_latency_sum = 0.0
        self.latency_sum = 0.0 

        self.pass_latency_sum2 = 0.0
        self.fail_latency_sum2 = 0.0
        self.latency_sum2 = 0.0 

        self.pass_latency_max = 0.0
        self.fail_latency_max = 0.0
        self.start()

    def add_to_hist(self, lat, bs):
        if self.hist_base is None: return
        if self.hist_factor is None: return
        index = int((self.hist_factor * lat) ** self.hist_base)
        try: self.hist_register[index] += int(bs)
        except KeyError:
            self.hist_register[index] = int(bs)

    def add(self, status, lat, bs):
        self.add_to_hist(lat, bs)
        self.latency_sum += float(lat) 
        self.latency_sum2 += float(lat) ** 2
        self.xcounter += int(bs)
        self.counter += 1

        if status:
            if self.pass_latency_max < lat:
                self.pass_latency_max = lat
            self.pass_counter += 1
            self.pass_xcounter += int(bs)
            self.pass_latency_sum += float(lat) 
            self.pass_latency_sum2 += float(lat) ** 2
        else:
            if self.fail_latency_max < lat:
                self.fail_latency_max = lat
            self.fail_counter += 1
            self.fail_xcounter += int(bs)
            self.fail_latency_sum += float(lat) 
            self.fail_latency_sum2 += float(lat) ** 2

    def start(self):
        self.start_timestamp = time.time()

    def stop(self):
        if self.stop_timestamp is None:
            self.stop_timestamp = time.time()
            return True
        return False

    def analyze(self):
        if self.stop_timestamp is None: self.stop()
        if self.prefix: prefix = f"{self.prefix}_"
        else: prefix = ""
            
        stats = {}
        duration = self.stop_timestamp - self.start_timestamp
        stats[f"{prefix}total_duration"] = duration
        stats[f"{prefix}total_batches"] = self.counter
        stats[f"{prefix}total_frames"] = self.xcounter
        stats[f"{prefix}start_timestamp"] = self.start_timestamp
        stats[f"{prefix}stop_timestamp"] = self.stop_timestamp
        
        stats[f"{prefix}pass_batches"] = self.pass_counter
        stats[f"{prefix}fail_batches"] = self.fail_counter
        stats[f"{prefix}pass_frames"] = self.pass_xcounter
        stats[f"{prefix}fail_frames"] = self.fail_xcounter

        stats[f"{prefix}pass_max_latency"] = self.pass_latency_max
        stats[f"{prefix}fail_max_latency"] = self.fail_latency_max
        if duration > 0.0:
            stats[f"{prefix}brutto_batch_rate"] = float(self.pass_counter) / duration
            stats[f"{prefix}brutto_frame_rate"] = float(self.pass_xcounter) / duration
        else:
            stats[f"{prefix}brutto_batch_rate"] = 0
            stats[f"{prefix}brutto_frame_rate"] = 0
        if self.latency_sum > 0.0:
            stats[f"{prefix}netto_batch_rate"] = float(self.pass_counter) / self.latency_sum
            stats[f"{prefix}netto_frame_rate"] = float(self.pass_xcounter) / self.latency_sum
        else:
            stats[f"{prefix}netto_batch_rate"] = 0.0
            stats[f"{prefix}netto_frame_rate"] = 0.0

        if self.xcounter > 0:
            stats[f"{prefix}frame_passrate"] = float(self.pass_xcounter) / self.xcounter
        else: stats[f"{prefix}frame_passrate"] = 0.0
        
        if self.counter > 0:
            stats[f"{prefix}batch_passrate"] = float(self.pass_counter) / self.counter
            latency_mean2 = self.latency_sum2 / self.counter
            latency_mean = self.latency_sum / self.counter
            latency_var = latency_mean2 - latency_mean ** 2
            try: latency_stdev = math.sqrt(latency_var)
            except: latency_stdev = 0.0
            try: latency_cv = latency_stdev / latency_mean
            except: latency_cv = 0.0
            stats[f"{prefix}mean_latency"] = latency_mean
            stats[f"{prefix}mean_latency2"] = latency_mean2
            stats[f"{prefix}stdev_latency"] = latency_stdev
            stats[f"{prefix}cv_latency"] = latency_cv
        else:
            stats[f"{prefix}batch_passrate"] = 0.0
            stats[f"{prefix}mean_latency"] = 0.0
            stats[f"{prefix}mean_latency2"] = 0.0
            stats[f"{prefix}stdev_latency"] = 0.0
            stats[f"{prefix}cv_latency"] = 0.0

        if self.pass_counter > 0:
            pass_latency_mean2 = self.pass_latency_sum2 / self.pass_counter
            pass_latency_mean = self.pass_latency_sum / self.pass_counter
            pass_latency_var = pass_latency_mean2 - pass_latency_mean ** 2
            try: pass_latency_stdev = math.sqrt(pass_latency_var)
            except: pass_latency_stdev = 0.0
            try: pass_latency_cv = pass_latency_stdev / pass_latency_mean
            except: pass_latency_cv = 0.0
            stats[f"{prefix}pass_mean_latency"] = pass_latency_mean
            stats[f"{prefix}pass_mean_latency2"] = pass_latency_mean2
            stats[f"{prefix}pass_stdev_latency"] = pass_latency_stdev
            stats[f"{prefix}pass_cv_latency"] = pass_latency_cv            
        else:
            stats[f"{prefix}pass_mean_latency"] = 0.0
            stats[f"{prefix}pass_mean_latency2"] = 0.0
            stats[f"{prefix}pass_stdev_latency"] = 0.0
            stats[f"{prefix}pass_cv_latency"] = 0.0
        
        if self.fail_counter > 0:
            fail_latency_mean2 = self.fail_latency_sum2 / self.fail_counter
            fail_latency_mean = self.fail_latency_sum / self.fail_counter
            fail_latency_var = fail_latency_mean2 - fail_latency_mean ** 2
            try: fail_latency_stdev = math.sqrt(fail_latency_var)
            except: fail_latency_stdev = 0.0
            try: fail_latency_cv = fail_latency_stdev / fail_latency_mean
            except: fail_latency_cv = 0.0
            stats[f"{prefix}fail_mean_latency"] = fail_latency_mean 
            stats[f"{prefix}fail_mean_latency2"] = fail_latency_mean2
            stats[f"{prefix}fail_stdev_latency"] = fail_latency_stdev
            stats[f"{prefix}fail_cv_latency"] = fail_latency_cv
        else:
            stats[f"{prefix}fail_mean_latency"] = 0.0
            stats[f"{prefix}fail_mean_latency2"] = 0.0
            stats[f"{prefix}fail_stdev_latency"] = 0.0
            stats[f"{prefix}fail_cv_latency"] = 0.0

        for key, val in self.hist_register.items():
            stats[f"{prefix}hist_latency_{key}"] = val 
        return stats
