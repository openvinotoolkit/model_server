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

import uuid
import json
import bson
import datetime
import pymongo


class DBExporter:
    json_prefix = "JSON-OUTPUT:"

    def __init__(self, args, worker_id):
        self.db_address = None
        self.collection_name = None
        self.database_name = None
        self.exec_date = None
        self.args = args
        self.dbid = None

        db_endpoint = args["db_endpoint"]
        if db_endpoint is not None:
            protocol, address, database, collection = db_endpoint.split(":")
            assert protocol == "mongodb", "only mongo db is supported!"
            self.exec_date = datetime.datetime.now()
            self.db_address = f"{protocol}:{address}"
            self.collection_name = collection
            self.database_name = database
            self.db_collection = None

            try:
                with pymongo.MongoClient(self.db_address) as db_client:
                    db_database = db_client[self.database_name]
                    self.db_collection = getattr(db_database, self.collection_name)
                    initial_doc = self.make_initial_doc()
                    ret = self.db_collection.insert_one(initial_doc)
                    self.dbid = bson.ObjectId(str(ret.inserted_id))
                    jout = json.dumps({"_id": str(ret.inserted_id)})
                    print(f"{self.json_prefix}###{worker_id}###DB-ID###{jout}")
            except pymongo.errors.ServerSelectionTimeoutError: self.dbid = None

    def make_initial_doc(self):
        # internal results are not supported by this tool but needed by externals
        doc = {"internal_results": []}

        if self.args["db_metadata"] is not None:
            for kv in self.args["db_metadata"]:
                parts = kv.split(":")
                if len(parts) < 2: continue
                elif len(parts) == 2:
                    key, value = parts
                else:
                    key = parts[0]
                    value = ":".join(parts[1:])
                doc[key] = value

        doc["final_results"] = {}
        doc["execution_date"] = self.exec_date
        doc["client_version"] = "2.7"
        doc["model_name"] = self.args["model_name"]
        doc["model_version"] = self.args["model_version"]

        if "model" not in doc: doc["model"] = self.args["model_name"]
        if "batchsize" not in doc: doc["batchsize"] = "-".join(map(str, self.args["bs"]))
        if "iterations" not in doc: doc["iterations"] = self.args["steps_number"]
        if "concurr" not in doc: doc["concurr"] = self.args["concurrency"]
        if "duration" not in doc: doc["duration"] = self.args["duration"]
        if "window" not in doc: doc["window"] = self.args["window"]
        if "warmup" not in doc: doc["warmup"] = self.args["warmup"]

        for karg, varg in self.args.items():
            if karg == "db_metadata": continue
            if karg == "db_endpoint": continue
            if karg == "quantile_list": continue
            if isinstance(varg, (str, int, float)):
                doc[f"opt_{karg}"] = varg
            elif isinstance(varg, (list, tuple)):
                val = "_".join(map(str, varg))
                doc[f"opt_{karg}"] = val
        return doc

    def upload_results(self, results, return_code):
        # if endpoint is not specified, upload is skipped
        if self.db_address is None: return

        doc = {}
        doc = {f"xcli_{k}": v for k, v in results.items()}
        doc["xcli_return_code"] = return_code
        cmd = {"$set": {"final_results": doc}}
        query = {"_id": self.dbid}

        try:
            if self.dbid is None:
                raise pymongo.errors.ServerSelectionTimeoutError()
            with pymongo.MongoClient(self.db_address) as db_client:
                db_database = db_client[self.database_name]
                db_collection = getattr(db_database, self.collection_name)
                ret = db_collection.update_one(query, cmd)

        except pymongo.errors.ServerSelectionTimeoutError:
            filename = "/tmp/xcli-" + uuid.uuid4().hex + ".dump"
            print("dump file:", filename)
            with open(filename, "w") as fd:
                endpoint = f"{self.db_address}: {self.database_name}\n"
                fd.write(str(endpoint))
                fd.write(f"\n{self.args}")
                fd.write(f"\n{doc}\n")
