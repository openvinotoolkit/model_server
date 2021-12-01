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

import yaml
import datetime
import pymongo
import uuid

class DBExporter(dict):
    data_format = "%d-%b-%Y %H:%M:%S"

    def __init__(self, args):
        dict.__init__(self)
        self.collection = None
        if args["db_config"] is not None:
            with open(args["db_config"], "r") as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)
                for key, val in config.items(): self[key] = val
        self.exec_date = datetime.datetime.now()
        self.args = args

    def connect_to_collection(self):
        db_full_address = self["endpoint"]["address"]
        db_client = pymongo.MongoClient(db_full_address)
        with db_client:
            db_database = db_client[self["endpoint"]["database"]]
            db_collection = getattr(db_database, self["endpoint"]["collection"])
        return db_collection

    def upload_results(self, results, return_code):
        # if endpoint is not specified, upload is skipped
        if self.args["db_config"] is None: return

        # internal results are not supported by this tool
        doc = {"internal_results": []}
        
        doc["final_results"] = {f"xcli_{key}": val for key, val in results.items()}
        doc["final_results"]["xcli_return_code"] = return_code
        doc["execution_date"] = self.exec_date
        if "model" not in self["metadata"]:
            self["metadata"]["model"] = self.args["model_name"]
        self["metadata"]["model_name"] = self["metadata"]["model"]
        if "version" not in self["metadata"]:
            self["metadata"]["model_version"] = self.args["model_version"]
        if "batchsize" not in self["metadata"]:
            self["metadata"]["batchsize"] = "-".join(map(str, self.args["bs"]))
        if "concurr" not in self["metadata"]:
            self["metadata"]["concurr"] = self.args["concurrency"]
        if "duration" not in self["metadata"]:
            self["metadata"]["duration"] = self.args["duration"]
        if "iterations" not in self["metadata"]:
            self["metadata"]["iterations"] = self.args["steps_number"]            
        doc.update(self["metadata"])
        prefix = self.get("prefix", "noprefix")
        dlist = prefix, doc["backend"], doc["model"], str(doc["batchsize"]), str(doc["concurr"])
        doc["description"] = "-".join(dlist)

        try:
            self.collection = self.connect_to_collection()
            self.collection.insert_one(doc)
        except pymongo.errors.ServerSelectionTimeoutError:
            filename = "/tmp/xcli-" + uuid.uuid4().hex + ".dump"
            print("dump file:", filename)
            with open(filename, "w") as fd:
                fd.write(str(self["endpoint"]))
                fd.write(f"\n{self.args}")
                fd.write(f"\n{doc}\n")
