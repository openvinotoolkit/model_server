//*****************************************************************************
// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cxxopts.hpp>
#include <signal.h>
#include <stdio.h>
#include <sysexits.h>

#ifdef _WIN32
#include <csignal>

#include <ntstatus.h>
#include <windows.h>
#endif

#include "ovms.h"  // NOLINT

namespace {

using signed_shape_t = std::vector<int64_t>;
class BenchmarkCLIParser {
    std::unique_ptr<cxxopts::Options> options;

public:
    std::unique_ptr<cxxopts::ParseResult> result;

    BenchmarkCLIParser() = default;
    void parse(int argc, char** argv);
    void prepare(OVMS_ServerSettings*, OVMS_ModelsSettings*);
};

void BenchmarkCLIParser::parse(int argc, char** argv) {
    try {
        options = std::make_unique<cxxopts::Options>(argv[0], "OpenVINO Model Server");

        // clang-format off
        options->add_options()
            ("h, help",
                "Show this help message and exit")
            // server options
            ("log_level",
                "serving log level - one of TRACE, DEBUG, INFO, WARNING, ERROR",
                cxxopts::value<std::string>()->default_value("ERROR"),
                "LOG_LEVEL")
            ("config_path",
                "Config file path for OVMS to read",
                cxxopts::value<std::string>()->default_value("/ovms/src/test/configs/config_benchmark.json"),
                "CONFIG_PATH")
// benchmark options
            ("niter",
                "number of inferences to conduct",
                cxxopts::value<uint32_t>()->default_value("1000"),
                "NITER")
            ("nstreams",
                "number of execution streams to be performed simultaneously (suggested for best throughput is NUM_STREAMS of a ovms model config)",
                cxxopts::value<uint32_t>()->default_value("1"),
                "NSTREAMS")
            // inference data
            ("servable_name",
                "Model name to sent request to",
                cxxopts::value<std::string>(),
                "MODEL_NAME")
            ("servable_version",
                "workload threads per ireq, if not set version will be set by default model version policy",
                cxxopts::value<int64_t>()->default_value("0"),
                "MODEL_VERSION")
            ("mode",
                "Workload mode. Possible values: INFERENCE_ONLY, RESET_BUFFER, RESET_REQUEST",
                cxxopts::value<std::string>()->default_value("INFERENCE_ONLY"),
                "MODE")
            ("seed",
                "Random values generator seed.",
                cxxopts::value<uint64_t>(),
                "SEED");

        result = std::make_unique<cxxopts::ParseResult>(options->parse(argc, argv));

        if (result->count("help") || result->arguments().size() == 0) {
            std::cout << options->help({"", "multi model", "single model"}) << std::endl;
            exit(EX_OK);
        }
    } catch (const std::exception& e) {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(EX_USAGE);
    }
}

volatile sig_atomic_t shutdown_request = 0;

static void onInterrupt(int status) {
    shutdown_request = 1;
}

static void onTerminate(int status) {
    shutdown_request = 1;
}

static void onIllegal(int status) {
    shutdown_request = 2;
}
#ifdef __linux__

static void installSignalHandlers() {
    static struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = onInterrupt;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    static struct sigaction sigTermHandler;
    sigTermHandler.sa_handler = onTerminate;
    sigemptyset(&sigTermHandler.sa_mask);
    sigTermHandler.sa_flags = 0;
    sigaction(SIGTERM, &sigTermHandler, NULL);

    static struct sigaction sigIllHandler;
    sigIllHandler.sa_handler = onIllegal;
    sigemptyset(&sigIllHandler.sa_mask);
    sigIllHandler.sa_flags = 0;
    sigaction(SIGILL, &sigIllHandler, NULL);
}

#elif _WIN32

static BOOL WINAPI onConsoleEvent(DWORD event) {
    switch (event) {
    case CTRL_C_EVENT:
        onInterrupt(SIGINT);
        return TRUE;
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        onTerminate(SIGTERM);
        return TRUE;
    default:
        return FALSE;
    }
}

static void installSignalHandlers() {
    SetConsoleCtrlHandler(onConsoleEvent, TRUE);
    signal(SIGINT, onInterrupt);
    signal(SIGTERM, onTerminate);
    signal(SIGILL, onIllegal);
}

#endif

size_t DataTypeToByteSize(OVMS_DataType datatype) {
    static std::unordered_map<OVMS_DataType, size_t> datatypeSizeMap{
        {OVMS_DATATYPE_BOOL, 1},
        {OVMS_DATATYPE_U1, 1},
        {OVMS_DATATYPE_U4, 1},
        {OVMS_DATATYPE_U8, 1},
        {OVMS_DATATYPE_U16, 2},
        {OVMS_DATATYPE_U32, 4},
        {OVMS_DATATYPE_I4, 1},
        {OVMS_DATATYPE_I8, 1},
        {OVMS_DATATYPE_I16, 2},
        {OVMS_DATATYPE_I32, 4},
        {OVMS_DATATYPE_FP16, 2},
        {OVMS_DATATYPE_FP32, 4},
        {OVMS_DATATYPE_BF16, 2},
    };
    auto it = datatypeSizeMap.find(datatype);
    if (it == datatypeSizeMap.end()) {
        return 0;
    }
    return it->second;
}

OVMS_InferenceRequest* prepareRequest(OVMS_Server* server, const std::string& servableName, int64_t servableVersion, OVMS_DataType datatype, const signed_shape_t& shape, const std::string& inputName, const void* data) {
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceRequestNew(&request, server, servableName.c_str(), servableVersion);
    OVMS_InferenceRequestAddInput(request, inputName.c_str(), datatype, shape.data(), shape.size());
    auto elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<signed_shape_t::value_type>());
    OVMS_InferenceRequestInputSetData(request, inputName.c_str(), data, DataTypeToByteSize(datatype) * elementsCount, OVMS_BUFFERTYPE_CPU, 0);
    return request;
}

void triggerInferenceInALoopInferenceOnly(
    std::future<void>& startSignal,
    std::promise<void>& readySignal,
    const size_t niterPerThread,
    size_t& wholeThreadTimeUs,
    double& averageWholeLatency,
    double& averagePureLatency,
    OVMS_Server* server,
    const std::string& servableName, int64_t servableVersion, OVMS_DataType datatype, const signed_shape_t& shape, const std::string& inputName,
    std::optional<uint64_t> seed) {
    OVMS_InferenceResponse* response{nullptr};
    std::vector<uint64_t> latenciesWhole(niterPerThread);
    std::vector<uint64_t> latenciesPure(niterPerThread);
    auto elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<signed_shape_t::value_type>());
    std::vector<float> data(elementsCount, 1.0);
    OVMS_InferenceRequest* request = prepareRequest(server, servableName, servableVersion, datatype, shape, inputName, (const void*)data.data());
    readySignal.set_value();
    startSignal.get();
    auto workloadStart = std::chrono::high_resolution_clock::now();
    size_t iter = niterPerThread;
    while (iter-- > 0) {
        auto iterationWholeStart = std::chrono::high_resolution_clock::now();
        auto iterationPureStart = std::chrono::high_resolution_clock::now();
        OVMS_Inference(server, request, &response);
        auto iterationPureEnd = std::chrono::high_resolution_clock::now();
        OVMS_InferenceResponseDelete(response);
        auto iterationWholeEnd = std::chrono::high_resolution_clock::now();
        latenciesWhole[iter] = std::chrono::duration_cast<std::chrono::microseconds>(iterationWholeEnd - iterationWholeStart).count();
        latenciesPure[iter] = std::chrono::duration_cast<std::chrono::microseconds>(iterationPureEnd - iterationPureStart).count();
    }
    auto workloadEnd = std::chrono::high_resolution_clock::now();
    wholeThreadTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(workloadEnd - workloadStart).count();
    averageWholeLatency = std::accumulate(latenciesWhole.begin(), latenciesWhole.end(), 0) / (double(niterPerThread) * 1'000);
    averagePureLatency = std::accumulate(latenciesPure.begin(), latenciesPure.end(), 0) / (double(niterPerThread) * 1'000);
    OVMS_InferenceRequestDelete(request);
}

void prepareData(std::vector<std::vector<float>>& preparedData, const size_t& numberOfVectors, const size_t& vectorElementCount, std::optional<uint64_t> seed){
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    std::default_random_engine generator;
    uint64_t seedValue;
    if (seed.has_value()) {
        seedValue = seed.value();
    } else {
        std::random_device rd;
        seedValue = rd();
    }
    std::cout << "Seed used to generate random values: " << seedValue << std::endl;
    generator = std::default_random_engine(seedValue);
    for (size_t i = 0; i < numberOfVectors; i ++) {
        float random = distribution(generator);
        std::vector<float> data(vectorElementCount, random);
        preparedData.push_back(data);
    }
}

void triggerInferenceInALoopResetBuffer(
    std::future<void>& startSignal,
    std::promise<void>& readySignal,
    const size_t niterPerThread,
    size_t& wholeThreadTimeUs,
    double& averageWholeLatency,
    double& averagePureLatency,
    OVMS_Server* server,
    const std::string& servableName, int64_t servableVersion, OVMS_DataType datatype, const signed_shape_t& shape, const std::string& inputName,
    std::optional<uint64_t> seed) {
    OVMS_InferenceResponse* response{nullptr};
    std::vector<uint64_t> latenciesWhole(niterPerThread);
    std::vector<uint64_t> latenciesPure(niterPerThread);
    std::vector<std::vector<float>> preparedData;
    auto elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<signed_shape_t::value_type>());
    prepareData(preparedData, niterPerThread, elementsCount, seed);
    std::vector<float> data(elementsCount, 1.0);
    OVMS_InferenceRequest* request = prepareRequest(server, servableName, servableVersion, datatype, shape, inputName, (const void*)data.data());
    readySignal.set_value();
    startSignal.get();
    auto workloadStart = std::chrono::high_resolution_clock::now();
    size_t iter = 0;
    while (iter < niterPerThread) {
        auto iterationWholeStart = std::chrono::high_resolution_clock::now();
        OVMS_InferenceRequestInputRemoveData(request, inputName.c_str());
        OVMS_InferenceRequestInputSetData(request, inputName.c_str(), (const void*)preparedData[iter].data(), elementsCount * sizeof(float), OVMS_BUFFERTYPE_CPU, 0);
        auto iterationPureStart = std::chrono::high_resolution_clock::now();
        OVMS_Inference(server, request, &response);
        auto iterationPureEnd = std::chrono::high_resolution_clock::now();
        OVMS_InferenceResponseDelete(response);
        auto iterationWholeEnd = std::chrono::high_resolution_clock::now();
        latenciesWhole[iter] = std::chrono::duration_cast<std::chrono::microseconds>(iterationWholeEnd - iterationWholeStart).count();
        latenciesPure[iter] = std::chrono::duration_cast<std::chrono::microseconds>(iterationPureEnd - iterationPureStart).count();
        iter++;
    }
    auto workloadEnd = std::chrono::high_resolution_clock::now();
    wholeThreadTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(workloadEnd - workloadStart).count();
    averageWholeLatency = std::accumulate(latenciesWhole.begin(), latenciesWhole.end(), 0) / (double(niterPerThread) * 1'000);
    averagePureLatency = std::accumulate(latenciesPure.begin(), latenciesPure.end(), 0) / (double(niterPerThread) * 1'000);
    OVMS_InferenceRequestDelete(request);
}

void triggerInferenceInALoopResetRequest(
    std::future<void>& startSignal,
    std::promise<void>& readySignal,
    const size_t niterPerThread,
    size_t& wholeThreadTimeUs,
    double& averageWholeLatency,
    double& averagePureLatency,
    OVMS_Server* server,
    const std::string& servableName, int64_t servableVersion, OVMS_DataType datatype, const signed_shape_t& shape, const std::string& inputName,
    std::optional<uint64_t> seed) {
    OVMS_InferenceResponse* response{nullptr};
    auto elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<signed_shape_t::value_type>());
    std::vector<uint64_t> latenciesWhole(niterPerThread);
    std::vector<uint64_t> latenciesPure(niterPerThread);
    std::vector<std::vector<float>> preparedData;
    prepareData(preparedData, niterPerThread, elementsCount, seed);
    readySignal.set_value();
    startSignal.get();
    auto workloadStart = std::chrono::high_resolution_clock::now();
    size_t iter = 0;
    while (iter < niterPerThread) {
        OVMS_InferenceRequest* request = prepareRequest(server, servableName, servableVersion, datatype, shape, inputName, (const void*)preparedData[iter].data());
        auto iterationWholeStart = std::chrono::high_resolution_clock::now();
        auto iterationPureStart = std::chrono::high_resolution_clock::now();
        OVMS_Inference(server, request, &response);
        auto iterationPureEnd = std::chrono::high_resolution_clock::now();
        OVMS_InferenceResponseDelete(response);
        OVMS_InferenceRequestDelete(request);
        auto iterationWholeEnd = std::chrono::high_resolution_clock::now();
        latenciesWhole[iter] = std::chrono::duration_cast<std::chrono::microseconds>(iterationWholeEnd - iterationWholeStart).count();
        latenciesPure[iter] = std::chrono::duration_cast<std::chrono::microseconds>(iterationPureEnd - iterationPureStart).count();
        iter++;
    }
    auto workloadEnd = std::chrono::high_resolution_clock::now();
    wholeThreadTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(workloadEnd - workloadStart).count();
    averageWholeLatency = std::accumulate(latenciesWhole.begin(), latenciesWhole.end(), 0) / (double(niterPerThread) * 1'000);
    averagePureLatency = std::accumulate(latenciesPure.begin(), latenciesPure.end(), 0) / (double(niterPerThread) * 1'000);
}

}  // namespace

enum Mode {
    INFERENCE_ONLY,
    RESET_BUFFER,
    RESET_REQUEST
};

int main(int argc, char** argv) {
    installSignalHandlers();
    BenchmarkCLIParser cliparser;
    cliparser.parse(argc, argv);

    OVMS_ServerSettings* serverSettings = 0;
    OVMS_ModelsSettings* modelsSettings = 0;
    OVMS_Server* srv;

    OVMS_ServerSettingsNew(&serverSettings);
    OVMS_ModelsSettingsNew(&modelsSettings);
    OVMS_ServerNew(&srv);

    uint32_t grpcPort = 9178;
    OVMS_ServerSettingsSetGrpcPort(serverSettings, grpcPort);

    std::string cliLogLevel(cliparser.result->operator[]("log_level").as<std::string>());
    OVMS_LogLevel_enum logLevel;
    if (cliLogLevel == "TRACE") {
        logLevel = OVMS_LOG_TRACE;
    } else if (cliLogLevel == "DEBUG") {
        logLevel = OVMS_LOG_DEBUG;
    } else if (cliLogLevel == "INFO") {
        logLevel = OVMS_LOG_INFO;
    } else if (cliLogLevel == "WARN") {
        logLevel = OVMS_LOG_WARNING;
    } else if (cliLogLevel == "ERROR") {
        logLevel = OVMS_LOG_ERROR;
    } else {
        std::cout << __LINE__ << std::endl;
        return EX_USAGE;
    }
    OVMS_ServerSettingsSetLogLevel(serverSettings, logLevel);
    OVMS_ModelsSettingsSetConfigPath(modelsSettings, cliparser.result->operator[]("config_path").as<std::string>().c_str());

    std::string modeParam = cliparser.result->operator[]("mode").as<std::string>();
    Mode mode;
    if (modeParam == "INFERENCE_ONLY") {
        mode = Mode::INFERENCE_ONLY;
    }else if (modeParam == "RESET_BUFFER") {
        mode = Mode::RESET_BUFFER;
    }else if (modeParam == "RESET_REQUEST") {
        mode = Mode::RESET_REQUEST;
    }else {
        std::cerr << "Invalid mode requested: " <<  modeParam << std::endl;
        return 1;
    }
    std::cout << "Mode requested: " <<  modeParam << std::endl;

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings);

    if (res) {
        uint32_t code = 0;
        const char* details = nullptr;
        OVMS_StatusCode(res, &code);
        OVMS_StatusDetails(res, &details);
        std::cout << "Error starting the server. Code:" << code
                  << "; details:" << details << std::endl;
        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return 1;
    }

    std::cout << "Server ready for inference" << std::endl;

    ///////////////////////
    // model parameters
    ///////////////////////
    std::string servableName(cliparser.result->operator[]("servable_name").as<std::string>());
    int64_t servableVersion(cliparser.result->operator[]("servable_version").as<int64_t>());
    if (servableVersion < 0) {
        std::cerr << "servableVersion cannot be negative" << std::endl;
        return EX_USAGE;
    }
    // input names handling
    OVMS_ServableMetadata* metadata;
    OVMS_GetServableMetadata(srv, servableName.c_str(), servableVersion, &metadata);
    const char* name;
    OVMS_DataType dt;
    size_t dimCount;
    int64_t* shapeMinArray;
    int64_t* discarded;
    OVMS_ServableMetadataInput(metadata, 0, &name, &dt, &dimCount, &shapeMinArray, &discarded);
    std::string inputName(name);
    // datatype handling
    OVMS_DataType datatype;
    if (dt == OVMS_DATATYPE_STRING || dt == OVMS_DATATYPE_U64 || dt == OVMS_DATATYPE_I64 || dt == OVMS_DATATYPE_FP64) {
        std::cerr << "Benchmarking models with following input types is currently unsupported: STRING, U64, I64, FP64" << std::endl;
        return 1;
    }
    if (dt != OVMS_DATATYPE_UNDEFINED) {
        datatype = dt;
    } else {
        datatype = OVMS_DATATYPE_FP32;
    }
    // shape handling
    signed_shape_t shape;
    for (size_t i = 0; i < dimCount; i++) {
        shape.push_back(shapeMinArray[i]);
    }
    ///////////////////////
    // benchmark parameters
    ///////////////////////
    size_t niter = cliparser.result->operator[]("niter").as<uint32_t>();
    size_t threadCount = cliparser.result->operator[]("nstreams").as<uint32_t>();
    size_t niterPerThread = std::max(niter / threadCount, size_t(1));

    auto elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<signed_shape_t::value_type>());
    std::vector<float> data(elementsCount, 0.1);

    ///////////////////////
    // prepare time measuring
    ///////////////////////
    std::vector<double> wholeTimes;
    wholeTimes.resize(threadCount);
    std::vector<double> pureTimes;
    pureTimes.resize(threadCount);

    ///////////////////////
    // prepare requests
    ///////////////////////
    OVMS_InferenceRequest* request = prepareRequest(srv, servableName, servableVersion, datatype, shape, inputName, (const void*)data.data());
    ///////////////////////
    // check request
    ///////////////////////
    OVMS_InferenceResponse* response;
    res = OVMS_Inference(srv, request, &response);
    if (res != nullptr) {
        uint32_t code = 0;
        const char* details = 0;
        OVMS_StatusCode(res, &code);
        OVMS_StatusDetails(res, &details);
        std::cerr << "Error occurred during inference. Code:" << code
                  << ", details:" << details << std::endl;
        OVMS_StatusDelete(res);
        OVMS_InferenceRequestDelete(request);
        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        exit(EX_CONFIG);
    }
    OVMS_InferenceResponseDelete(response);
    std::optional<uint64_t> seed;
    if (cliparser.result->count("seed")) {
        seed = cliparser.result->operator[]("seed").as<uint64_t>();
    }

    ///////////////////////
    // setup workload machinery
    ///////////////////////
    std::vector<std::unique_ptr<std::thread>> workerThreads;
    std::vector<std::promise<void>> startSignals(threadCount);
    std::vector<std::promise<void>> readySignals(threadCount);
    std::vector<std::future<void>> futureStartSignals;
    std::vector<std::future<void>> futureReadySignals;
    std::vector<size_t> wholeThreadsTimesUs(threadCount, 0);
    std::transform(startSignals.begin(),
        startSignals.end(),
        std::back_inserter(futureStartSignals),
        [](auto& p) { return p.get_future(); });
    std::transform(readySignals.begin(),
        readySignals.end(),
        std::back_inserter(futureReadySignals),
        [](auto& p) { return p.get_future(); });

    ///////////////////////
    // prepare threads
    ///////////////////////
    void (*triggerInferenceInALoop)(std::future<void>&, std::promise<void>&, const size_t, size_t&, double&, double&, OVMS_Server*, const std::string&, int64_t, OVMS_DataType, const signed_shape_t&, const std::string&, std::optional<uint64_t>);
    if (mode == Mode::INFERENCE_ONLY) {
        triggerInferenceInALoop = triggerInferenceInALoopInferenceOnly;
    } else if (mode == Mode::RESET_BUFFER){
        triggerInferenceInALoop = triggerInferenceInALoopResetBuffer;
    } else if (mode == Mode::RESET_REQUEST) {
        triggerInferenceInALoop = triggerInferenceInALoopResetRequest;
    }
    for (size_t i = 0; i < threadCount; ++i) {
        workerThreads.emplace_back(std::make_unique<std::thread>(
            [&futureStartSignals,
            &readySignals,
            &niterPerThread,
            &wholeThreadsTimesUs,
            &wholeTimes,
            &pureTimes,
            &srv,
            &servableName,
            &servableVersion,
            &datatype,
            &shape,
            &inputName,
            &triggerInferenceInALoop,
            &seed,
            i]() {
                triggerInferenceInALoop(
                    futureStartSignals[i],
                    readySignals[i],
                    niterPerThread,
                    wholeThreadsTimesUs[i],
                    wholeTimes[i],
                    pureTimes[i],
                    srv,
                    servableName, servableVersion, datatype, shape, inputName,
                    seed);
            }));
    }

    // allow all threads to initialize
    std::for_each(futureReadySignals.begin(), futureReadySignals.end(), [](auto& readySignal) { readySignal.get(); });
    ///////////////////////
    // start workload
    ///////////////////////
    std::cout << "Benchmark starting workload" << std::endl;
    auto workloadStart = std::chrono::high_resolution_clock::now();
    std::for_each(startSignals.begin(), startSignals.end(), [](auto& startSignal) { startSignal.set_value(); });
    ///////////////////////
    // end workload
    ///////////////////////
    std::for_each(workerThreads.begin(), workerThreads.end(), [](auto& t) { t->join(); });
    auto workloadEnd = std::chrono::high_resolution_clock::now();
    auto wholeTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(workloadEnd - workloadStart).count();
    std::cout << "FPS: " << double(niter) / wholeTimeUs * 1'000'000 << std::endl;
    OVMS_InferenceRequestDelete(request);
    double totalWhole = std::accumulate(wholeTimes.begin(), wholeTimes.end(), double(0)) / threadCount;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average latency : " << totalWhole << "ms" << std::endl;
    // OVMS cleanup
    OVMS_ServerDelete(srv);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);
    std::cout << "main() exit" << std::endl;
    return 0;
}
// adjustable nireq, adjustable shape, model name
