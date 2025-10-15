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
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <utility>

#include <strsafe.h>
#include <windows.h>
#include <tchar.h>
#include <errors.h>

#include "server.hpp"
#include "main_windows.hpp"

std::string OvmsWindowsServiceManager::getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S  ");
    return oss.str();
}

// #define DEBUG_LOG(msg) { std::wstringstream ss; ss << msg; OutputDebugStringW(ss.str().c_str()); }
// TODO: Implement windows logging mechanism with events
std::ofstream logFile("C:\\test2\\ovms.log");
#define DEBUG_LOG(msg)                                                           \
    {                                                                            \
        std::stringstream ss;                                                    \
        ss << OvmsWindowsServiceManager::getCurrentTimeString() << msg << std::endl; \
        logFile << ss.rdbuf();                                                   \
        logFile.flush();                                                         \
    }

using ovms::Server;

OvmsWindowsServiceManager manager;

// Need this original function pointer type expected by the Windows Service API (LPSERVICE_MAIN_FUNCTIONA),
void WINAPI WinServiceMain(DWORD argc, LPTSTR* argv) {
    manager.serviceMain(argc, argv);
}

int main_windows(int argc, char** argv) {
    //std::ofstream logFile2("C:\\test2\\ovms.log", std::ios::out | std::ios::trunc);
    //logFile2.close();
    DEBUG_LOG("Windows Main - Entry");
    manager.ovmsParams.argc = argc;
    manager.ovmsParams.argv = argv;
    OvmsWindowsServiceManager::logParameters(argc, argv, "OVMS Main Argument");

    // Install service with ovms.exe
    if( CompareString(LOCALE_INVARIANT, NORM_IGNORECASE, argv[1], -1, TEXT("install"), -1) == CSTR_EQUAL )
    {
        OvmsWindowsServiceManager::serviceInstall();
        return 0;
    }

    SERVICE_TABLE_ENTRY ServiceTable[] =
        {
            {OvmsWindowsServiceManager::serviceName, (LPSERVICE_MAIN_FUNCTION)WinServiceMain},
            {NULL, NULL}};

    // Service start on windows success
    if (StartServiceCtrlDispatcher(ServiceTable) == TRUE) {
        DEBUG_LOG("StartServiceCtrlDispatcher returned success");
        return 0;
    } else {
        DWORD error = GetLastError();
        // Not running as a service; run as console app.
        if (error == ERROR_FAILED_SERVICE_CONTROLLER_CONNECT) {
            DEBUG_LOG("StartServiceCtrlDispatcher ERROR_FAILED_SERVICE_CONTROLLER_CONNECT starting as console application.")
            Server& server = Server::instance();
            return server.start(argc, argv);
        } else {
            // Error trying to start as service
            std::string message = std::system_category().message(error);
            DEBUG_LOG("StartServiceCtrlDispatcher failed.")
            DEBUG_LOG(message.c_str())
            return -1;
        }
    }

    DEBUG_LOG("Windows Main - Exit");
    return 0;
}

SERVICE_STATUS OvmsWindowsServiceManager::serviceStatus = {0};
std::unique_ptr<WinServiceStatusWrapper> OvmsWindowsServiceManager::statusHandle = std::make_unique<WinServiceStatusWrapper>();
std::unique_ptr<WinServiceEventWrapper> OvmsWindowsServiceManager::serviceStopEvent = std::make_unique<WinServiceEventWrapper>();
LPSTR OvmsWindowsServiceManager::serviceName = _T("ovms");
LPSTR OvmsWindowsServiceManager::serviceDisplayName = _T("OpenVino Model Server");
LPSTR OvmsWindowsServiceManager::serviceDesc = _T("Hosts models and makes them accessible to software components over standard network protocols.");
OvmsWindowsServiceManager::OvmsWindowsServiceManager() {
    DEBUG_LOG("OvmsWindowsServiceManager constructor");
    ovmsParams = {};
}
OvmsWindowsServiceManager::~OvmsWindowsServiceManager() {
    DEBUG_LOG("OvmsWindowsServiceManager destructor");
}

struct WinHandleDeleter
{
    typedef HANDLE pointer;
    void operator()(HANDLE h)
    {
        DEBUG_LOG("WinHandleDeleter: closing handle");
        if(h != NULL && h != INVALID_HANDLE_VALUE)
        {
            CloseHandle(h);
        }
    }
};

void WINAPI OvmsWindowsServiceManager::serviceMain(DWORD argc, LPTSTR* argv) {
    DEBUG_LOG("ServiceMain: Entry");

    statusHandle->handle = RegisterServiceCtrlHandler(OvmsWindowsServiceManager::serviceName, OvmsWindowsServiceManager::serviceCtrlHandler);
    if (this->statusHandle->handle == NULL || this->statusHandle->handle == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("ServiceMain: RegisterserviceCtrlHandler returned error");
        serviceReportEvent("RegisterServiceCtrlHandler");
        return;
    }

    this->setServiceStartStatus();

    DEBUG_LOG("ServiceMain: Performing Service Start Operations");
    OvmsWindowsServiceManager::logParameters(argc, argv, "ServiceMain Argument");

    if (argv && argc > 1) {
        DEBUG_LOG("ServiceMain: Setting new parameters for service after service start.");
        manager.ovmsParams.argc = argc;
        manager.ovmsParams.argv = argv;
    }

    std::unique_ptr<HANDLE, WinHandleDeleter> mainThread(CreateThread(NULL, 0, OvmsWindowsServiceManager::serviceWorkerThread, &manager.ovmsParams, 0, NULL));
    if (mainThread.get() == NULL || mainThread.get() == INVALID_HANDLE_VALUE) {
        // Handle error
        DEBUG_LOG("ServiceMain: mainThread == NULL || mainThread == INVALID_HANDLE_VALUE");
        serviceReportEvent("CreateThread");
        return;
    }

    // Create stop event to wait on later.
    serviceStopEvent->handle = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (serviceStopEvent->handle == NULL || serviceStopEvent->handle == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("ServiceMain: CreateEvent(serviceStopEvent) returned error");
        serviceReportEvent("CreateEvent");
        this->setServiceStopStatusWithError();
        return;
    }

    // Tell the service controller we are started
    this->setServiceRunningStatus();

    DEBUG_LOG("ServiceMain: Waiting for Worker Thread to complete");

    WaitForSingleObject(mainThread.get(), INFINITE);
    DEBUG_LOG("ServiceMain: Worker Thread Stop Event signaled after we leave the WaitForSingle call");

    this->setServiceStopStatusWithSuccess();
    DEBUG_LOG("ServiceMain: Exit");

    return;
}

struct WinSCHandleDeleter
{
    typedef SC_HANDLE pointer;
    void operator()(SC_HANDLE h)
    {
        DEBUG_LOG("WinSCHandleDeleter: closing handle");
        if(h != NULL && h != INVALID_HANDLE_VALUE)
        {
            CloseServiceHandle(h);
        }
    }
};

void OvmsWindowsServiceManager::serviceInstall() {
    TCHAR szUnquotedPath[MAX_PATH];
    DEBUG_LOG("Installing Openvino Model Server service");
    std::cout << "Installing Openvino Model Server service" << std::endl;
    if( !GetModuleFileName( NULL, szUnquotedPath, MAX_PATH )) {
        DEBUG_LOG("serviceInstall, GetModuleFileName failed.");
        serviceReportEvent("GetModuleFileName");
        return;
    }

    // In case the path contains a space, it must be quoted so that
    // it is correctly interpreted. For example,
    // "d:\my share\myservice.exe" should be specified as
    // ""d:\my share\myservice.exe"".
    TCHAR szPath[MAX_PATH];
    StringCbPrintf(szPath, MAX_PATH, TEXT("\"%s\""), szUnquotedPath);

    // Get a handle to the SCM database. 
    std::unique_ptr<SC_HANDLE, WinSCHandleDeleter> schSCManager(OpenSCManager(
        NULL,                    // local computer
        NULL,                    // ServicesActive database 
        SC_MANAGER_ALL_ACCESS)); // full access rights 

    if (schSCManager.get() == NULL || schSCManager.get() == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("OpenSCManager failed");
        serviceReportEvent("OpenSCManager");
        return;
    }

    // Create the service
    std::unique_ptr<SC_HANDLE, WinSCHandleDeleter> schService(CreateService( 
        schSCManager.get(),                 // SCM database 
        OvmsWindowsServiceManager::serviceName, // name of service 
        OvmsWindowsServiceManager::serviceDisplayName, // service name to display 
        SERVICE_ALL_ACCESS,                 // desired access 
        SERVICE_WIN32_OWN_PROCESS,          // service type 
        SERVICE_DEMAND_START,               // start type 
        SERVICE_ERROR_NORMAL,               // error control type 
        szPath,                             // path to service's binary 
        NULL,                               // no load ordering group 
        NULL,                               // no tag identifier 
        NULL,                               // no dependencies 
        NULL,                               // LocalSystem account 
        NULL));                             // no password 
 
    if (schService.get() == NULL || schService.get() == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("CreateService failed");
        serviceReportEvent("CreateService");
        return;
    }

    SERVICE_DESCRIPTION sd;
    sd.lpDescription =  OvmsWindowsServiceManager::serviceDesc;
    if (!ChangeServiceConfig2(schService.get(), SERVICE_CONFIG_DESCRIPTION, &sd)) {
        DEBUG_LOG("ChangeServiceConfig2 failed");
        serviceReportEvent("ChangeServiceConfig2");
        return;
    }
    DEBUG_LOG("Openvino Model Server service installed successfully.");
    std::cout << "Openvino Model Server service installed successfully" << std::endl;
    return;
}

void OvmsWindowsServiceManager::logParameters(DWORD argc, LPTSTR* argv, const std::string& logText) {
    for (int i = 0; i < (int)manager.ovmsParams.argc; ++i) {
        std::stringstream ss2;
        ss2 << logText << " " << i << ": " << manager.ovmsParams.argv[i];
        DEBUG_LOG(ss2.rdbuf());
    }
}

struct WinESHandleDeleter
{
    typedef HANDLE  pointer;
    void operator()(HANDLE  h)
    {
        DEBUG_LOG("WinESHandleDeleter: closing handle");
        if(h != NULL && h != INVALID_HANDLE_VALUE)
        {
            DeregisterEventSource(h);
        }
    }
};

void OvmsWindowsServiceManager::serviceReportEvent(LPSTR szFunction) {
    LPCTSTR lpszStrings[2];
    TCHAR Buffer[200];
    std::unique_ptr<SC_HANDLE, WinESHandleDeleter> hEventSource(RegisterEventSource(NULL, OvmsWindowsServiceManager::serviceName));
    if(hEventSource.get() != NULL)
    {
        DWORD errcode = GetLastError();
        std::string message = std::system_category().message(errcode);
        StringCchPrintf(Buffer, 200, TEXT("%s failed with %lu error: %s"), szFunction, errcode, message.c_str());
        lpszStrings[0] = OvmsWindowsServiceManager::serviceName;
        lpszStrings[1] = Buffer;
        ReportEvent(hEventSource.get(),  // event log handle
                    EVENTLOG_ERROR_TYPE, // event type
                    0,                   // event category
                    0,                   // event identifier
                    NULL,                // no security identifier
                    2,                   // size of lpszStrings array
                    0,                   // no binary data
                    lpszStrings,         // array of strings
                    NULL);               // no binary data

    } else {
        DEBUG_LOG("RegisterEventSource failed");
        DEBUG_LOG(std::system_category().message(GetLastError()));
    }
}

void WINAPI OvmsWindowsServiceManager::serviceCtrlHandler(DWORD CtrlCode) {
    DEBUG_LOG("serviceCtrlHandler: Entry");

    switch (CtrlCode) {
    case SERVICE_CONTROL_STOP:
        DEBUG_LOG("serviceCtrlHandler: SERVICE_CONTROL_STOP Request");
        if (serviceStatus.dwCurrentState != SERVICE_RUNNING)
            break;

        setServiceStopStatusPending();
        // Signal the worker thread to start shutting down
        SetEvent(serviceStopEvent->handle);
        break;
    // Currently not supported controls
    case SERVICE_CONTROL_INTERROGATE:
        break;
    case SERVICE_CONTROL_CONTINUE:
        break;
    case SERVICE_CONTROL_PAUSE:
        break;
    default:
        break;
    }

    DEBUG_LOG("serviceCtrlHandler: Exit");
}

DWORD WINAPI OvmsWindowsServiceManager::serviceWorkerThread(LPVOID lpParam) {
    DEBUG_LOG("serviceWorkerThread: Entry");
    std::unique_ptr<OvmsService> ovmsService = std::make_unique<OvmsService>();
    ovmsService->error = 0;
    ovmsService->started = false;
    //  Start OVMS and check for stop
    while (WaitForSingleObject(serviceStopEvent->handle, 0) != WAIT_OBJECT_0) {
        // TODO: Check ovms running with OVMS serverLive and Ready
        if (!ovmsService->started) {
            ConsoleParameters* params = (ConsoleParameters*)lpParam;
            if (params && params->argc > 1) {
                DEBUG_LOG("serviceWorkerThread: Starting ovms from start parameters.");
                OvmsWindowsServiceManager::logParameters(params->argc, params->argv, "OVMS Main Argument");
                ovmsService->SetUp(params->argc, params->argv);
            } else {
                DEBUG_LOG("serviceWorkerThread: Error - No parameters passed to ovms service.");
                break;
            }
        } else {
            DEBUG_LOG("serviceWorkerThread: Ovms running ...")
        }

        // Check for events
        // TODO: Implement CreateWaitableTimer and SetWaitableTimerEx to save cpu
        Sleep(3000);
    }

    if (ovmsService->started) {
        ovmsService->TearDown();
        DEBUG_LOG("serviceWorkerThread: Stopping ovms service.");
    } else {
        DEBUG_LOG("serviceWorkerThread: Ovms service could not be started.");
    }

    if (ovmsService->error) {
        // TODO: Catch parsing errors from OVMS - currently we do not have this info
        DEBUG_LOG("serviceWorkerThread: Ovms start returned error.");
        DEBUG_LOG(ovmsService->error);
        serviceReportEvent("Ovms start returned error.");
        return ovmsService->error;
    }

    DEBUG_LOG("serviceWorkerThread: Exit");
    return ERROR_SUCCESS;
}

void OvmsWindowsServiceManager::setServiceStartStatus() {
    ZeroMemory(&serviceStatus, sizeof(serviceStatus));
    serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_START_PENDING;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwServiceSpecificExitCode = 0;
    serviceStatus.dwCheckPoint = 0;

    if (SetServiceStatus(this->statusHandle->handle, &serviceStatus) == FALSE) {
        DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
        serviceReportEvent("SetServiceStatus");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus start");
}

void OvmsWindowsServiceManager::setServiceStopStatusWithError() {
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_STOPPED;
    serviceStatus.dwWin32ExitCode = GetLastError();
    serviceStatus.dwCheckPoint = 1;
    if (SetServiceStatus(this->statusHandle->handle, &serviceStatus) == FALSE) {
        DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
        serviceReportEvent("SetServiceStatus");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus stop with error");
}

void OvmsWindowsServiceManager::setServiceRunningStatus() {
    serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP;
    serviceStatus.dwCurrentState = SERVICE_RUNNING;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwCheckPoint = 0;

    if (SetServiceStatus(this->statusHandle->handle, &serviceStatus) == FALSE) {
        DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
        serviceReportEvent("SetServiceStatus");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus running");
}

void OvmsWindowsServiceManager::setServiceStopStatusPending() {
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_STOP_PENDING;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwCheckPoint = 4;

    if (SetServiceStatus(OvmsWindowsServiceManager::statusHandle->handle, &serviceStatus) == FALSE) {
        DEBUG_LOG("serviceCtrlHandler: SetServiceStatus returned error");
        serviceReportEvent("SetServiceStatus");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus stop pending");
}

void OvmsWindowsServiceManager::setServiceStopStatusWithSuccess() {
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_STOPPED;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwCheckPoint = 3;

    if (SetServiceStatus(this->statusHandle->handle, &serviceStatus) == FALSE) {
        DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
        serviceReportEvent("SetServiceStatus");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus stop with success");
}

void OvmsService::TearDown() {
    DEBUG_LOG("OvmsService::TearDown");
    server.setShutdownRequest(1);
    if (t)
        t->join();
    server.setShutdownRequest(0);
    this->started = false;
}

int OvmsService::SetUp(int argc, char** argv) {
    DEBUG_LOG("OvmsService::SetUp");
    t.reset(new std::thread([argc, argv, this]() {
        this->started = true;
        this->error = server.start(argc, argv);
    }));
    return 0;
}

bool OvmsService::isReady() {
    return server.isReady();
}

bool OvmsService::isLive(const std::string& moduleName) {
    return server.isLive(moduleName);
}

WinServiceStatusWrapper::WinServiceStatusWrapper() {
    handle = NULL;
}
WinServiceStatusWrapper::~WinServiceStatusWrapper() {
    DEBUG_LOG("WinServiceStatusWrapper: closing handle");
    if(handle != NULL && handle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(handle);
    }
}

WinServiceEventWrapper::WinServiceEventWrapper() {
    handle = INVALID_HANDLE_VALUE;
}
WinServiceEventWrapper::~WinServiceEventWrapper() {
    DEBUG_LOG("WinServiceEventWrapper: closing handle");
    if(handle != NULL && handle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(handle);
    }
}

