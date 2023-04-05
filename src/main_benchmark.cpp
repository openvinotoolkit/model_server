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
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include <cxxopts.hpp>
#include <signal.h>
#include <stdio.h>
#include <sysexits.h>

#include "ovms.h"  // NOLINT
#include "stringutils.hpp"

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
            ("port",
                "gRPC server port",
                cxxopts::value<uint32_t>()->default_value("9178"),
                "PORT")
            ("rest_port",
                "REST server port, the REST server will not be started if rest_port is blank or set to 0",
                cxxopts::value<uint32_t>()->default_value("0"),
                "REST_PORT")
            ("log_level",
                "serving log level - one of TRACE, DEBUG, INFO, WARNING, ERROR",
                cxxopts::value<std::string>()->default_value("ERROR"),
                "LOG_LEVEL")
            ("config_path",
                "Config file path for OVMS to read",
                cxxopts::value<std::string>()->default_value("/ovms/src/test/c_api/config_benchmark.json"),
                "CONFIG_PATH")
// benchmark options
            ("niter",
                "number of inferences to conduct",
                cxxopts::value<uint32_t>()->default_value("1000"),
                "NITER")
            ("nireq",
                "nireq from OVMS configuration",
                cxxopts::value<uint32_t>()->default_value("1"),
                "NIREQ")
            ("threads_per_ireq",
                "workload threads per ireq",
                cxxopts::value<uint32_t>()->default_value("2"),
                "THREADS_PER_IREQ")
            // inference data
            ("servable_name",
                "Model name to sent request to",
                cxxopts::value<std::string>(),
                "MODEL_NAME")
            ("servable_version",
                "workload threads per ireq",
                cxxopts::value<int64_t>()->default_value("0"),
                "MODEL_VERSION")
            ("inputs_names",
                "Comma separated list of inputs names",
                cxxopts::value<std::string>(),
                "INPUTS_NAMES")
            ("shape",
                "Semicolon separated list of inputs names followed by their shapes in brackers. For example: \"inputA[1,3,224,224],inputB[1,10]\"",
                cxxopts::value<std::string>(),
                "INPUTS_NAMES");

        result = std::make_unique<cxxopts::ParseResult>(options->parse(argc, argv));

        if (result->count("help") || result->arguments().size() == 0) {
            std::cout << options->help({"", "multi model", "single model"}) << std::endl;
            exit(EX_OK);
        }
    } catch (const cxxopts::OptionException& e) {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(EX_USAGE);
    }
}

signed_shape_t parseShapes(const std::string& cliInputShapes) {
    auto inputShapes = ovms::tokenize(cliInputShapes, ';');
    if (inputShapes.size() != 1) {
        std::cout << __LINE__ << std::endl;
        throw std::invalid_argument("Invalid shape argument");
    }
    std::string firstShape = inputShapes[0];
    size_t leftBracket = firstShape.find("[");
    size_t rightBracket = firstShape.find("]");
    if ((leftBracket == std::string::npos) ||
        (rightBracket == std::string::npos) ||
        (leftBracket > rightBracket)) {
        std::cout << __LINE__ << std::endl;
        throw std::invalid_argument("Invalid shape argument");
    }
    std::string shapeString = firstShape.substr(leftBracket + 1, rightBracket - leftBracket - 1);
    auto dimsString = ovms::tokenize(shapeString, ',');
    signed_shape_t shape;
    std::transform(dimsString.begin(), dimsString.end(), std::back_inserter(shape),
                                   [](const std::string& s) -> signed_shape_t::value_type {
        auto dimOpt = ovms::stoi64(s);
        if (!dimOpt.has_value() || dimOpt.value() <= 0) {
            std::cout << __LINE__ << " " << s << std::endl;
            throw std::invalid_argument("Invalid shape argument");
        }
                                   return dimOpt.value(); });
    return shape;
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

OVMS_InferenceRequest* prepareRequest(OVMS_Server* server, const std::string& servableName, int64_t servableVersion, OVMS_DataType datatype, const signed_shape_t& shape, const std::string& inputName, const void* data) {
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceRequestNew(&request, server, servableName.c_str(), servableVersion);
    OVMS_InferenceRequestAddInput(request, inputName.c_str(), datatype, shape.data(), shape.size());
    auto elementsCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<signed_shape_t::value_type>());
    OVMS_InferenceRequestInputSetData(request, inputName.c_str(), data, sizeof(float) * elementsCount, OVMS_BUFFERTYPE_CPU, 0);
    return request;
}

void triggerInferenceInALoop(
    std::future<void>& startSignal,
    std::future<void>& stopSignal,
    const size_t niterPerThread,
    size_t& wholeThreadTimeUs,
    double& averageWholeLatency,
    double& averagePureLatency,
    OVMS_Server* server,
    OVMS_InferenceRequest* request) {
    OVMS_InferenceResponse* response{nullptr};
    std::vector<uint64_t> latenciesWhole(niterPerThread);
    std::vector<uint64_t> latenciesPure(niterPerThread);
    startSignal.get();
    auto workloadStart = std::chrono::high_resolution_clock::now();
    size_t iter = niterPerThread;
    while (iter-- > 0) {
        // stopSignal will be used with ctrl-c app stopping or with total requestCount
        // stopSignal.wait_for(std::chrono::milliseconds(0));
        auto iterationWholeStart = std::chrono::high_resolution_clock::now();
        auto iterationPureStart = std::chrono::high_resolution_clock::now();
        // aternatively we are changing request
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
}
}  // namespace

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

    uint32_t grpcPort = cliparser.result->operator[]("port").as<uint32_t>();
    uint32_t restPort = cliparser.result->operator[]("rest_port").as<uint32_t>();
    OVMS_ServerSettingsSetGrpcPort(serverSettings, grpcPort);
    OVMS_ServerSettingsSetRestPort(serverSettings, restPort);

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

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings);

    if (res) {
        uint32_t code = 0;
        const char* details = nullptr;
        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
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
    std::string cliInputsNames(cliparser.result->operator[]("inputs_names").as<std::string>());
    auto inputsNames = ovms::tokenize(cliInputsNames, ',');
    if (inputsNames.size() != 1) {
        std::cout << __LINE__ << std::endl;
        return EX_USAGE;
    }
    std::string inputName = inputsNames[0];
    // datatype handling
    OVMS_DataType datatype = OVMS_DATATYPE_FP32;
    // shape handling
    signed_shape_t shape = parseShapes(cliparser.result->operator[]("shape").as<std::string>());
    ///////////////////////
    // benchmark parameters
    ///////////////////////
    size_t nireq = cliparser.result->operator[]("nireq").as<uint32_t>();
    size_t niter = cliparser.result->operator[]("niter").as<uint32_t>();
    size_t threadsPerIreq = cliparser.result->operator[]("threads_per_ireq").as<uint32_t>();
    size_t threadCount = nireq * threadsPerIreq;
    size_t niterPerThread = niter / threadCount;

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
        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cerr << "Error occured during inference. Code:" << code
                  << ", details:" << details << std::endl;
        OVMS_StatusDelete(res);
        OVMS_InferenceRequestDelete(request);
        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        exit(EX_CONFIG);
    }
    OVMS_InferenceResponseDelete(response);

    ///////////////////////
    // setup workload machinery
    ///////////////////////
    std::vector<std::unique_ptr<std::thread>> workerThreads;
    std::vector<std::promise<void>> startSignals(threadCount);
    std::vector<std::promise<void>> stopSignals(threadCount);
    std::vector<std::future<void>> futureStartSignals;
    std::vector<std::future<void>> futureStopSignals;
    std::vector<size_t> wholeThreadsTimesUs(threadCount, 0);
    std::transform(startSignals.begin(),
        startSignals.end(),
        std::back_inserter(futureStartSignals),
        [](auto& p) { return p.get_future(); });
    std::transform(stopSignals.begin(),
        stopSignals.end(),
        std::back_inserter(futureStopSignals),
        [](auto& p) { return p.get_future(); });

    ///////////////////////
    // prepare threads
    ///////////////////////
    for (size_t i = 0; i < threadCount; ++i) {
        workerThreads.emplace_back(std::make_unique<std::thread>(
            [&futureStartSignals,
             &futureStopSignals,
             &niterPerThread,
             &wholeThreadsTimesUs,
             &wholeTimes,
             &pureTimes,
             &srv,
             &request,
             i]() {
                triggerInferenceInALoop(
                    futureStartSignals[i],
                    futureStopSignals[i],
                    niterPerThread,
                    wholeThreadsTimesUs[i],
                    wholeTimes[i],
                    pureTimes[i],
                    srv,
                    request);
            }));
    }
    // sleep to allow all threads to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
    auto totalUs = std::accumulate(wholeThreadsTimesUs.begin(), wholeThreadsTimesUs.end(), 0);
    std::cout << "Average per thread FPS: " << double(niter) * threadCount/totalUs * 1'000'000 << std::endl;
    OVMS_InferenceRequestDelete(request);
    double totalWhole = std::accumulate(wholeTimes.begin(), wholeTimes.end(), double(0)) / threadCount;
    double totalPure = std::accumulate(pureTimes.begin(), pureTimes.end(), double(0)) / threadCount;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average latency whole prediction path:" << totalWhole << "ms" << std::endl;
    std::cout << "Average latency pure C-API inference:" << totalPure << "ms" << std::endl;
    // OVMS cleanup
    OVMS_ServerDelete(srv);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);
    std::cout << "main() exit" << std::endl;
    return 0;
}
// adjustable nireq, adjustable shape, model name
