#
# INTEL CONFIDENTIAL
# Copyright (c) 2021 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
#

import yaml
import datetime
import pymongo

class DBExporter(dict):
    data_format = "%d-%b-%Y %H:%M:%S"

    def __init__(self, args):
        dict.__init__(self)
        self.collection = None
        if args["db_config"] is not None:
            with open(args["db_config"], "r") as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)
                for key, val in config.items(): self[key] = val
            self.collection = self.connect_to_collection()            
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
        if self.collection is None: return

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
        self.collection.insert_one(doc)
