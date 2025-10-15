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
    bool isReady();
    bool isLive(const std::string& moduleName);
};

struct WinServiceStatusWrapper {
    SERVICE_STATUS_HANDLE handle;
    WinServiceStatusWrapper();
    ~WinServiceStatusWrapper();
};

struct WinServiceEventWrapper {
    HANDLE handle;
    WinServiceEventWrapper();
    ~WinServiceEventWrapper();
};

class OvmsWindowsServiceManager {
public:
    OvmsWindowsServiceManager();
    ~OvmsWindowsServiceManager();

    // Members
    ConsoleParameters ovmsParams;
    static LPSTR serviceName;
    static LPSTR serviceDisplayName;
    static LPSTR serviceDesc;

    // Methods
    static std::string getCurrentTimeString();
    static void serviceInstall();
    static void logParameters(DWORD argc, LPTSTR* argv, const std::string& logText);
    static void serviceReportEvent(LPSTR szFunction);
    void WINAPI serviceMain(DWORD argc, LPTSTR* argv);

private:
    // Members
    static SERVICE_STATUS serviceStatus;
    static std::unique_ptr<WinServiceStatusWrapper> statusHandle;
    static std::unique_ptr<WinServiceEventWrapper> serviceStopEvent;

    // Methods
    static void WINAPI serviceCtrlHandler(DWORD);
    static DWORD WINAPI serviceWorkerThread(LPVOID lpParam);

    // Service status update
    static void setServiceStopStatusPending();
    void setServiceStartStatus();
    void setServiceStopStatusWithSuccess();
    void setServiceStopStatusWithError();
    void setServiceRunningStatus();
};
