//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#ifndef SRC_MAIN_WINDOWS_HPP_
#define SRC_MAIN_WINDOWS_HPP_
#endif  // SRC_MAIN_WINDOWS_HPP_

#include <memory>
#include <string>
#include <utility>
#include <Windows.h>
#include <tchar.h>

#include "server.hpp"
using ovms::Server;

int main_windows(int argc, char** argv);

struct ConsoleParameters {
    int argc;
    char** argv;
};

class OvmsService {
private:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<std::thread> t;

public:
    bool started;
    int error;
    void TearDown();
    int SetUp(int argc, char** argv);
};

class WindowsServiceManager {
public:
    static std::string getCurrentTimeString();
    WindowsServiceManager();
    ConsoleParameters ovmsParams;
    static LPSTR serviceName;
    VOID WINAPI serviceMain(DWORD argc, LPTSTR* argv);

private:
    static SERVICE_STATUS serviceStatus;
    static SERVICE_STATUS_HANDLE statusHandle;
    static HANDLE serviceStopEvent;

    static VOID WINAPI serviceCtrlHandler(DWORD);
    static DWORD WINAPI serviceWorkerThread(LPVOID lpParam);

    static void setServiceStopStatusPending();
    void setServiceStartStatus();
    void setServiceStopStatusWithSuccess();
    void setServiceStopStatusWithError();
    void setServiceRunningStatus();
};
