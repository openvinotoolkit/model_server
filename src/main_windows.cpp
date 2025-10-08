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
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <utility>
#include <Windows.h>
#include <tchar.h>

#include "server.hpp"
#include "main_windows.hpp"

std::string WindowsServiceManager::getCurrentTimeString() {
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
#define DEBUG_LOG(msg) { std::stringstream ss; ss << WindowsServiceManager::getCurrentTimeString() << msg << std::endl; logFile << ss.rdbuf(); logFile.flush(); }

using ovms::Server;

WindowsServiceManager manager;

VOID WINAPI WinServiceMain(DWORD argc, LPTSTR *argv){
    manager.serviceMain(argc, argv);
}

int main_windows(int argc, char** argv) { 
    std::ofstream logFile2("C:\\test2\\ovms.log", std::ios::out | std::ios::trunc);
    logFile2.close();
    // TODO: Write storing default arguments for restart
    DEBUG_LOG("Windows Main - Entry");
    manager.ovmsParams.argc = argc;
    manager.ovmsParams.argv = argv;
    for (int i = 0; i < manager.ovmsParams.argc; ++i) {
        std::stringstream ss2; ss2 << "OVMS Main Argument " << i << ": " << manager.ovmsParams.argv[i];
        DEBUG_LOG(ss2.rdbuf());
    }

    SERVICE_TABLE_ENTRY ServiceTable[] = 
    {
        {WindowsServiceManager::serviceName, (LPSERVICE_MAIN_FUNCTION) WinServiceMain},
        {NULL, NULL}
    };

    // Service start on windows success
    if (StartServiceCtrlDispatcher (ServiceTable) == TRUE)
    {
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

SERVICE_STATUS WindowsServiceManager::serviceStatus = {0};
SERVICE_STATUS_HANDLE WindowsServiceManager::statusHandle = NULL;
HANDLE WindowsServiceManager::serviceStopEvent = INVALID_HANDLE_VALUE;
LPSTR WindowsServiceManager::serviceName = _T("OpenVino Model Server");
WindowsServiceManager::WindowsServiceManager() {
    ovmsParams = {};
}

VOID WINAPI WindowsServiceManager::serviceMain(DWORD argc, LPTSTR *argv)
{
    DEBUG_LOG("ServiceMain: Entry");

    statusHandle = RegisterServiceCtrlHandler(WindowsServiceManager::serviceName, WindowsServiceManager::serviceCtrlHandler);
    if (statusHandle == NULL) 
    {
        DEBUG_LOG("ServiceMain: RegisterserviceCtrlHandler returned error");
        return;
    }

    this->setServiceStartStatus();

    DEBUG_LOG("ServiceMain: Performing Service Start Operations");
    for (DWORD i = 0; i < argc; ++i) {
        std::stringstream ss2; ss2 << "ServiceMain Argument " << i << ": " << argv[i];
        DEBUG_LOG(ss2.rdbuf());
    }

    if (argv && argc > 1) {
        DEBUG_LOG("ServiceMain: Setting new parameters for service after service start.");
        manager.ovmsParams.argc = argc;
        manager.ovmsParams.argv = argv;
    }

    HANDLE mainThread = CreateThread (NULL, 0, WindowsServiceManager::serviceWorkerThread, &manager.ovmsParams, 0, NULL);

    if (mainThread == NULL || mainThread == INVALID_HANDLE_VALUE) {
        // Handle error
        DEBUG_LOG("ServiceMain: mainThread == NULL || mainThread == INVALID_HANDLE_VALUE");
        CloseHandle (statusHandle);
        return;
    }

    // Create stop event to wait on later.
    serviceStopEvent = CreateEvent (NULL, TRUE, FALSE, NULL);
    if (serviceStopEvent == NULL) 
    {
        DEBUG_LOG("ServiceMain: CreateEvent(serviceStopEvent) returned error");
        this->setServiceStopStatusWithError();
        CloseHandle (statusHandle);
        return;
    }    

    // Tell the service controller we are started
    this->setServiceRunningStatus();

    DEBUG_LOG("ServiceMain: Waiting for Worker Thread to complete");

    WaitForSingleObject (mainThread, INFINITE);
    DEBUG_LOG("ServiceMain: Worker Thread Stop Event signaled after we leave the WaitForSingle call");

    CloseHandle (serviceStopEvent);
    this->setServiceStopStatusWithSuccess();
    DEBUG_LOG("ServiceMain: Exit");

    return;
}

VOID WINAPI WindowsServiceManager::serviceCtrlHandler(DWORD CtrlCode)
{
    DEBUG_LOG("serviceCtrlHandler: Entry");

    switch (CtrlCode) 
	{
     case SERVICE_CONTROL_STOP :

        DEBUG_LOG("serviceCtrlHandler: SERVICE_CONTROL_STOP Request");

        if (serviceStatus.dwCurrentState != SERVICE_RUNNING)
           break;
        
        setServiceStopStatusPending();

        // Signal the worker thread to start shutting down
        SetEvent (serviceStopEvent);

        break;

     default:
         break;
    }

    DEBUG_LOG("serviceCtrlHandler: Exit");
}

DWORD WINAPI WindowsServiceManager::serviceWorkerThread(LPVOID lpParam)
{
    DEBUG_LOG("serviceWorkerThread: Entry");
    std::unique_ptr<OvmsService> ovmsService = std::make_unique<OvmsService>();
    ovmsService->error = 0;
    ovmsService->started = false;
    //  Start OVMS and check for stop
    while (WaitForSingleObject(serviceStopEvent, 0) != WAIT_OBJECT_0)
    {
        // TODO: Check ovms running with OVMS serverLive and Ready
        if (!ovmsService->started) {
            ConsoleParameters* params = (ConsoleParameters*)lpParam;
            if (params && params->argc > 1) {
                DEBUG_LOG("serviceWorkerThread: Starting ovms from start parameters.");
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
        DEBUG_LOG("serviceWorkerThread: Stoping ovms service.");
    } else {
        DEBUG_LOG("serviceWorkerThread: Ovms service could not be started.");
    }

    if (ovmsService->error) {
        // TODO: Catch parsing erors from OVMS - currently we do not have this info
        DEBUG_LOG("serviceWorkerThread: Ovms start returned error.");
        DEBUG_LOG(ovmsService->error);
        return ovmsService->error;
    }

    DEBUG_LOG("serviceWorkerThread: Exit");
    return ERROR_SUCCESS;
}

void WindowsServiceManager::setServiceStartStatus() {
    ZeroMemory (&serviceStatus, sizeof (serviceStatus));
    serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_START_PENDING;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwServiceSpecificExitCode = 0;
    serviceStatus.dwCheckPoint = 0;

    if (SetServiceStatus (statusHandle, &serviceStatus) == FALSE) 
    {
        DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus start");
}

void WindowsServiceManager::setServiceStopStatusWithError() {
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_STOPPED;
    serviceStatus.dwWin32ExitCode = GetLastError();
    serviceStatus.dwCheckPoint = 1;
    if (SetServiceStatus (statusHandle, &serviceStatus) == FALSE)
    {
        DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus stop with error");
}

void WindowsServiceManager::setServiceRunningStatus() {
    serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP;
    serviceStatus.dwCurrentState = SERVICE_RUNNING;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwCheckPoint = 0;

    if (SetServiceStatus (statusHandle, &serviceStatus) == FALSE)
    {
	    DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus running");
}

void WindowsServiceManager::setServiceStopStatusPending() {
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_STOP_PENDING;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwCheckPoint = 4;

    if (SetServiceStatus (statusHandle, &serviceStatus) == FALSE)
    {
        DEBUG_LOG("serviceCtrlHandler: SetServiceStatus returned error");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus stop pending");
}

void WindowsServiceManager::setServiceStopStatusWithSuccess() {
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_STOPPED;
    serviceStatus.dwWin32ExitCode = 0;
    serviceStatus.dwCheckPoint = 3;

    if (SetServiceStatus (statusHandle, &serviceStatus) == FALSE)
    {
	    DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus stop with success");
}

void OvmsService::TearDown() {
    DEBUG_LOG("OvmsService::TearDown");
    server.setShutdownRequest(1);
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
