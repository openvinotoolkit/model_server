/*
 * Copyright (C) 2020-2021 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <assert.h>
#include <rapidjson/document.h>

#include "../../customloaderinterface.hpp"

using namespace std;
using namespace rapidjson;
using namespace ovms;

#define PATH_SIZE 10
#define RSIZE_MAX_STR 4096

#define CUSTLOADER_OK 1
#define CUSTLOADER_ERROR 40

typedef pair<string, int> model_id_t;

class custSampleLoader : public CustomLoaderInterface {
private:
    vector<model_id_t> models_blacklist;
    vector<model_id_t> models_loaded;

protected:
    int extract_input_params(char* basePath, int version, char* loaderOptions, char** binFile,
        char** xmlFile);
    int load_files(char* binFile, char* xmlFile, char** xmlBuffer, char** binBuffer, int* xmlLen,
        int* binLen);

    std::promise<void> exitSignal;
    std::future<void> futureObj;
    int watchIntervalSec = 0;
    bool watcherStarted = false;
    std::thread watcher_thread;

public:
    custSampleLoader();
    ~custSampleLoader();
    int loaderInit(char* loader_path);
    void loaderDeInit();
    int unloadModel(const char* modelName, int version);
    int loadModel(const char* modelName, const char* basePath, const int version,
        const char* loaderOptions, char** xmlBuffer, int* xmlLen, char** binBuffer,
        int* binLen);
    bool getModelBlacklistStatus(const char* modelName, int version);

    void threadFunction(std::future<void> futureObj);
    void startWatcher(int intervalSec);
    void watcherJoin();
};

extern "C" CustomLoaderInterface* create() {
    return new custSampleLoader();
}

extern "C" void destroy(CustomLoaderInterface* p) {
    delete p;
}

custSampleLoader::custSampleLoader() {
    cout << "custSampleLoader: Instance of Custom SampleLoader created" << endl;
}

custSampleLoader::~custSampleLoader() {
    cout << "custSampleLoader: Instance of Custom SampleLoader deleted" << endl;
}

int custSampleLoader::loaderInit(char* loader_path) {
    cout << "custSampleLoader: Custom loaderInit" << loader_path << endl;
    // if error return CUSTLOADER_INIT_FAIL;
    return CUSTLOADER_OK;
}

int custSampleLoader::load_files(char* binFile, char* xmlFile, char** xmlBuffer,
    char** binBuffer, int* xmlLen, int* binLen) {
    streampos size;
    ifstream bfile(binFile, ios::in | ios::binary | ios::ate);
    if (bfile.is_open()) {
        size = bfile.tellg();
        *binLen = size;
        *binBuffer = new char[size];
        bfile.seekg(0, ios::beg);
        bfile.read(*binBuffer, size);
        bfile.close();
    } else {
        cout << "Unable to open bin file" << endl;
        return CUSTLOADER_ERROR;
    }
    ifstream xfile(xmlFile, ios::in | ios::ate);
    if (xfile.is_open()) {
        size = xfile.tellg();
        *xmlLen = size;
        *xmlBuffer = new char[size];
        xfile.seekg(0, ios::beg);
        xfile.read(*xmlBuffer, size);
        xfile.close();
    } else {
        cout << "Unable to open xml file" << endl;
        return CUSTLOADER_ERROR;
    }
    return CUSTLOADER_OK;
}
int custSampleLoader::extract_input_params(char* basePath, int version, char* loaderOptions,
    char** binFile, char** xmlFile) {
    size_t str_len = 0;
    size_t path_len = 0;
    int ret = CUSTLOADER_OK;
    char path[RSIZE_MAX_STR];
    Document doc;

    if (basePath == NULL || loaderOptions == NULL) {
        cout << "custSampleLoader: Invalid input parameters to loadModel" << endl;
        return CUSTLOADER_ERROR;
    }

    snprintf(path, RSIZE_MAX_STR, "%s/%d", basePath, version);
    string fullpath = string(path);
    path_len = fullpath.length() + PATH_SIZE;

    // parse jason input string
    if (doc.Parse(loaderOptions).HasParseError()) {
        return CUSTLOADER_ERROR;
    }

    for (Value::ConstMemberIterator itr = doc.MemberBegin(); itr != doc.MemberEnd(); ++itr)
        printf("Type of member %s is %s\n", itr->name.GetString(), itr->value.GetString());

    if (doc.HasMember("bin_file")) {
        string binName = doc["bin_file"].GetString();
        size_t size = path_len + binName.length();
        string binPath = fullpath + "/" + binName;
        *binFile = new char[binPath.length() + 1];
        if (binFile == NULL) {
            cout << "Error: could not allocate memory " << endl;
            return CUSTLOADER_ERROR;
        }
        snprintf(*binFile, size, "%s", binPath.c_str());
        cout << "binFile:" << *binFile << endl;
    }

    if (doc.HasMember("xml_file")) {
        string xmlName = doc["xml_file"].GetString();
        size_t size = path_len + xmlName.length();
        string xmlPath = fullpath + "/" + xmlName;
        *xmlFile = new char[xmlPath.length() + 1];
        if (xmlFile == NULL) {
            cout << "Error: could not allocate memory " << endl;
            return CUSTLOADER_ERROR;
        }
        snprintf(*xmlFile, size, "%s", xmlPath.c_str());
        cout << "xmlFile:" << *xmlFile << endl;
    }
    return ret;
}

int custSampleLoader::loadModel(const char* modelName, const char* basePath, const int version,
    const char* loaderOptions, char** xmlBuffer, int* xmlLen,
    char** binBuffer, int* binLen) {
    cout << "custSampleLoader: Custom loadModel" << endl;

    char* type = NULL;
    char* binFile = NULL;
    char* xmlFile = NULL;

    int ret =
        extract_input_params((char*)basePath, version, (char*)loaderOptions, &binFile, &xmlFile);
    if (ret != CUSTLOADER_OK || binFile == NULL || binFile == NULL) {
        cout << "custSampleLoader: Invalid custom loader options" << endl;
        return CUSTLOADER_ERROR;
    }

    // load models
    ret = load_files(binFile, xmlFile, xmlBuffer, binBuffer, xmlLen, binLen);
    if (ret != CUSTLOADER_OK || xmlBuffer == NULL || xmlBuffer == NULL) {
        cout << "custSampleLoader: Could not read model files" << endl;
        return CUSTLOADER_ERROR;
    }

    /* Start the watcher thread after first moel load */
    if (watcherStarted == false) {
        int interval = 30;
        startWatcher(interval);
        this_thread::sleep_for(chrono::seconds(1));
    }

    models_loaded.push_back(make_pair(string(modelName), version));
    return CUSTLOADER_OK;
}

int custSampleLoader::unloadModel(const char* modelName, int version) {
    cout << "custSampleLoader: Custom unloadModel" << endl;

    model_id_t toFind = make_pair(string(modelName), version);

    auto it = models_loaded.begin();
    for (; it != models_loaded.end(); it++) {
        if (*it == toFind)
            break;
    }

    if (it == models_loaded.end()) {
        cout << modelName << " is not loaded" << endl;
    } else {
        models_loaded.erase(it);
    }
    return CUSTLOADER_OK;
}

void custSampleLoader::loaderDeInit() {
    cout << "custSampleLoader: Custom loaderDeInit" << endl;
    watcherJoin();
    return;
}

bool custSampleLoader::getModelBlacklistStatus(const char* modelName, int version) {
    cout << "custSampleLoader: Custom getModelBlacklistStatus" << endl;

    if (models_blacklist.size() == 0)
        return false;

    model_id_t toFind = make_pair(string(modelName), version);

    auto it = models_blacklist.begin();
    for (; it != models_blacklist.end(); it++) {
        if (*it == toFind)
            break;
    }

    if (it == models_blacklist.end()) {
        return false;
    }

    /* model name and version in blacklist.. return true */
    return true;
}

void custSampleLoader::threadFunction(future<void> futureObj) {
    cout << "custSampleLoader: Thread Start" << endl;
    int count = 1;
    while (futureObj.wait_for(chrono::milliseconds(1)) == future_status::timeout) {
        cout << "custSampleLoader: Doing Some Work " << count++ << endl;
        this_thread::sleep_for(std::chrono::seconds(watchIntervalSec));
        // After 1 cycles forcing the thread to add first model as blacklisted
        if (count == 2) {
            if (models_loaded.size() > 0) {
                models_blacklist.push_back(models_loaded[0]);
                cout << "custSampleLoader: Blacklisting the model " << endl;
            }
        }

        //After 2 cycles clear the blacklist and break
        if (count == 3) {
            cout << "custSampleLoader: Clearing the blacklist " << endl;
            models_blacklist.clear();
        }
    }
    cout << "custSampleLoader: Thread END" << endl;
    watcherJoin();
}

void custSampleLoader::startWatcher(int interval) {
    watchIntervalSec = interval;

    if ((!watcherStarted) && (watchIntervalSec > 0)) {
        future<void> futureObj = exitSignal.get_future();
        thread th(thread(&custSampleLoader::threadFunction, this, move(futureObj)));
        watcherStarted = true;
        watcher_thread = move(th);
    }
    cout << "custSampleLoader: StartWatcher" << endl;
}

void custSampleLoader::watcherJoin() {
    cout << "custSampleLoader: watcherJoin()" << endl;
    if (watcherStarted) {
        exitSignal.set_value();
        if (watcher_thread.joinable()) {
            watcher_thread.detach();
            watcherStarted = false;
        }
    }
}
