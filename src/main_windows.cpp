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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <utility>
#include <vector>
#pragma warning(push)
#pragma warning(disable : 6553)
#include <WinReg/WinReg.hpp>
#pragma warning(pop)
#include <strsafe.h>
#include <windows.h>
#include <tchar.h>
#include <errors.h>

#include "main_windows.hpp"
#include "module_names.hpp"
#include "ovms_exit_codes.hpp"
#include "server.hpp"

namespace ovms_service {
std::string OvmsWindowsServiceManager::getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S  ");
    return oss.str();
}

#define DEBUG_LOG_ENABLE 0
static std::ofstream logFile("C:\\temp\\ovms.log", std::ios::app);
#define DEBUG_LOG(msg)                                                                   \
    {                                                                                    \
        if (DEBUG_LOG_ENABLE) {                                                          \
            std::stringstream ss;                                                        \
            ss << OvmsWindowsServiceManager::getCurrentTimeString() << msg << std::endl; \
            logFile << ss.rdbuf();                                                       \
            logFile.flush();                                                             \
        }                                                                                \
    }

using ovms::Server;

OvmsWindowsServiceManager& OvmsWindowsServiceManager::instance() {
    static OvmsWindowsServiceManager global;
    return global;
}

// Need this original function pointer type expected by the Windows Service API (LPSERVICE_MAIN_FUNCTIONA),
void WINAPI WinServiceMain(DWORD argc, LPTSTR* argv) {
    OvmsWindowsServiceManager::instance().serviceMain(argc, argv);
}

std::string wstringToString(const std::wstring& wstr) {
    if (wstr.empty()) {
        return std::string();
    }

    // First, determine the required buffer size
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string strTo(size_needed, 0);

    // Perform the actual conversion
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, strTo.data(), size_needed, nullptr, nullptr);

    return strTo;
}

inline std::wstring stringToWstring(const std::string& str, UINT codePage = CP_THREAD_ACP) {
    if (str.empty()) {
        return std::wstring();
    }

    int required = ::MultiByteToWideChar(codePage, 0, str.data(), (int)str.size(), NULL, 0);
    if (0 == required) {
        return std::wstring();
    }

    std::wstring str2;
    str2.resize(required);

    int converted = ::MultiByteToWideChar(codePage, 0, str.data(), (int)str.size(), &str2[0], str2.capacity());
    if (0 == converted) {
        return std::wstring();
    }

    return str2;
}

int main_windows(int argc, char** argv) {
    DEBUG_LOG("Windows Main - Entry");
    OvmsWindowsServiceManager::instance().ovmsParams.argc = argc;
    OvmsWindowsServiceManager::instance().ovmsParams.argv = argv;
    OvmsWindowsServiceManager::logParameters(argc, argv, "OVMS Main Argument");

    // Install service with ovms.exe
    if (argc > 1 && CompareString(LOCALE_INVARIANT, NORM_IGNORECASE, argv[1], -1, TEXT("install"), -1) == CSTR_EQUAL) {
        if (!OvmsWindowsServiceManager::serviceSetDescription()) {
            DEBUG_LOG("serviceSetDescription returned failure");
            return -1;
        }

        OvmsWindowsServiceManager::setPythonPathRegistry();
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

struct WinHandleDeleter {
    typedef HANDLE pointer;
    void operator()(HANDLE h) {
        std::stringstream ss2;
        ss2 << "WinHandleDeleter: closing handle: " << h;
        DEBUG_LOG(ss2.str());
        if (h != NULL && h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
        }
    }
};

// Arguments for this function are the arguments from sc start ovms
// When no arguments are passed we use those from sc create ovms - during install service, and overwrite the parameters
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
    // argc = 1 equals ovms.exe
    if (argv && argc > 1) {
        DEBUG_LOG("ServiceMain: Setting new parameters for service after service start.");
        OvmsWindowsServiceManager::instance().ovmsParams.argc = argc;
        OvmsWindowsServiceManager::instance().ovmsParams.argv = argv;
    }
    OvmsWindowsServiceManager::logParameters(argc, argv, "ServiceMain Argument");

    // Parse arguments before server start
    auto paramsOrExit = ovms::Server::parseArgs(OvmsWindowsServiceManager::instance().ovmsParams.argc, OvmsWindowsServiceManager::instance().ovmsParams.argv);
    // Check for error in parsing
    if (std::holds_alternative<std::pair<int, std::string>>(paramsOrExit)) {
        auto printAndExit = std::get<std::pair<int, std::string>>(paramsOrExit);
        // Check retcode other than success
        if (printAndExit.first > 0) {
            DEBUG_LOG("ServiceMain: Server::parseArgs returned error");
            serviceReportEventWithExitCode("ovms::Server::parseArgs", printAndExit.second, printAndExit.first);
            this->setServiceStopStatusWithExitCode(printAndExit.first);
        } else {
            // Check retcode 0 but service not started [--help, --version] arguments
            DEBUG_LOG("ServiceMain: Server::parseArgs returned success, no valid parameters to start the service provided.");
            serviceReportEventWithExitCode("ovms::Server::parseArgs", printAndExit.second, printAndExit.first);
            this->setServiceStopStatusWithExitCode(printAndExit.first);
        }

        return;
    } else {
        OvmsWindowsServiceManager::instance().parsedParameters = std::get<std::pair<ovms::ServerSettingsImpl, ovms::ModelsSettingsImpl>>(paramsOrExit);
    }

    std::unique_ptr<HANDLE, WinHandleDeleter> mainThread(CreateThread(NULL, 0, OvmsWindowsServiceManager::serviceWorkerThread, &OvmsWindowsServiceManager::instance().parsedParameters, 0, NULL));
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

    DEBUG_LOG("ServiceMain: Waiting for Worker Thread to complete");

    WaitForSingleObject(mainThread.get(), INFINITE);
    DEBUG_LOG("ServiceMain: Worker Thread Stop Event signaled after we leave the WaitForSingle call");

    this->setServiceStopStatusWithSuccess();
    DEBUG_LOG("ServiceMain: Exit");

    return;
}

struct WinSCHandleDeleter {
    typedef SC_HANDLE pointer;
    void operator()(SC_HANDLE h) {
        std::stringstream ss2;
        ss2 << "WinSCHandleDeleter: closing handle: " << h;
        DEBUG_LOG(ss2.str());
        if (h != NULL && h != INVALID_HANDLE_VALUE) {
            CloseServiceHandle(h);
        }
    }
};

// Deprecated ovms self install method
// Use sc create ... instead
// Cannot be used as it does not create the registry entry for the service
// Registry entry required to add ovms\python to PATH
void OvmsWindowsServiceManager::serviceInstall() {
    TCHAR szUnquotedPath[MAX_PATH];
    DEBUG_LOG("Installing Openvino Model Server service");
    std::cout << "Installing Openvino Model Server service" << std::endl;
    if (!GetModuleFileName(NULL, szUnquotedPath, MAX_PATH)) {
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
        NULL,                     // local computer
        NULL,                     // ServicesActive database
        SC_MANAGER_ALL_ACCESS));  // full access rights

    if (schSCManager.get() == NULL || schSCManager.get() == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("OpenSCManager failed");
        serviceReportEvent("OpenSCManager");
        return;
    }

    // Create the service
    std::unique_ptr<SC_HANDLE, WinSCHandleDeleter> schService(CreateService(
        schSCManager.get(),                             // SCM database
        OvmsWindowsServiceManager::serviceName,         // name of service
        OvmsWindowsServiceManager::serviceDisplayName,  // service name to display
        SERVICE_ALL_ACCESS,                             // desired access
        SERVICE_WIN32_OWN_PROCESS,                      // service type
        SERVICE_DEMAND_START,                           // start type
        SERVICE_ERROR_NORMAL,                           // error control type
        szPath,                                         // path to service's binary
        NULL,                                           // no load ordering group
        NULL,                                           // no tag identifier
        NULL,                                           // no dependencies
        NULL,                                           // LocalSystem account
        NULL));                                         // no password

    if (schService.get() == NULL || schService.get() == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("CreateService failed");
        serviceReportEvent("CreateService");
        return;
    }

    SERVICE_DESCRIPTION sd;
    sd.lpDescription = OvmsWindowsServiceManager::serviceDesc;
    if (!ChangeServiceConfig2(schService.get(), SERVICE_CONFIG_DESCRIPTION, &sd)) {
        DEBUG_LOG("ChangeServiceConfig2 failed");
        serviceReportEvent("ChangeServiceConfig2");
        return;
    }
    DEBUG_LOG("Openvino Model Server service installed successfully.");
    std::cout << "Openvino Model Server service installed successfully" << std::endl;
    return;
}

bool OvmsWindowsServiceManager::serviceSetDescription() {
    // Get a handle to the SCM database.
    std::unique_ptr<SC_HANDLE, WinSCHandleDeleter> schSCManager(OpenSCManager(
        NULL,                     // local computer
        NULL,                     // ServicesActive database
        SC_MANAGER_ALL_ACCESS));  // full access rights

    if (schSCManager.get() == NULL || schSCManager.get() == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("OpenSCManager failed");
        std::cout << "OpenSCManager failed" << std::endl;
        return false;
    }

    // Create the service
    std::unique_ptr<SC_HANDLE, WinSCHandleDeleter> schService(OpenServiceA(
        schSCManager.get(),                      // SCM database
        OvmsWindowsServiceManager::serviceName,  // name of service
        SERVICE_ALL_ACCESS));                    // desired access

    if (schService.get() == NULL || schService.get() == INVALID_HANDLE_VALUE) {
        DEBUG_LOG("OpenService failed");
        std::cout << "OpenService failed" << std::endl;
        return false;
    }

    SERVICE_DESCRIPTION sd;
    sd.lpDescription = OvmsWindowsServiceManager::serviceDesc;
    if (!ChangeServiceConfig2(schService.get(), SERVICE_CONFIG_DESCRIPTION, &sd)) {
        DEBUG_LOG("ChangeServiceConfig2 failed");
        std::cout << "ChangeServiceConfig2 failed" << std::endl;
        return false;
    }
    DEBUG_LOG("Openvino Model Server service description updated.");
    std::cout << "Openvino Model Server service description updated." << std::endl;
    return true;
}

void OvmsWindowsServiceManager::logParameters(DWORD argc, LPTSTR* argv, const std::string& logText) {
    for (int i = 0; i < (int)OvmsWindowsServiceManager::instance().ovmsParams.argc; ++i) {
        std::stringstream ss2;
        ss2 << logText << " " << i << ": " << OvmsWindowsServiceManager::instance().ovmsParams.argv[i];
        DEBUG_LOG(ss2.str());
    }
}

struct WinESHandleDeleter {
    typedef HANDLE pointer;
    void operator()(HANDLE h) {
        std::stringstream ss2;
        ss2 << "WinESHandleDeleter: closing handle: " << h;
        DEBUG_LOG(ss2.str());
        if (h != NULL && h != INVALID_HANDLE_VALUE) {
            DeregisterEventSource(h);
        }
    }
};

void OvmsWindowsServiceManager::serviceReportEvent(const std::string& szFunction) {
    serviceReportEvent(const_cast<LPSTR>(szFunction.c_str()));
}

void OvmsWindowsServiceManager::serviceReportEvent(LPSTR szFunction) {
    LPCTSTR lpszStrings[2];
    TCHAR Buffer[200];
    std::unique_ptr<SC_HANDLE, WinESHandleDeleter> hEventSource(RegisterEventSource(NULL, OvmsWindowsServiceManager::serviceName));
    if (hEventSource.get() != NULL) {
        DWORD errcode = GetLastError();
        std::string message = std::system_category().message(errcode);
        StringCchPrintf(Buffer, 200, TEXT("%s failed with %lu error: %s"), szFunction, errcode, message.c_str());
        lpszStrings[0] = OvmsWindowsServiceManager::serviceName;
        lpszStrings[1] = Buffer;
        ReportEvent(hEventSource.get(),  // event log handle
            EVENTLOG_ERROR_TYPE,         // event type
            0,                           // event category
            0,                           // event identifier
            NULL,                        // no security identifier
            2,                           // size of lpszStrings array
            0,                           // no binary data
            lpszStrings,                 // array of strings
            NULL);                       // no binary data

    } else {
        DEBUG_LOG("RegisterEventSource failed");
        DEBUG_LOG(std::system_category().message(GetLastError()));
    }
}

void OvmsWindowsServiceManager::serviceReportEventWithExitCode(const std::string& szFunction, const std::string& message, const int& exitCode) {
    serviceReportEventWithExitCode(const_cast<LPSTR>(szFunction.c_str()), message, exitCode);
}

void OvmsWindowsServiceManager::serviceReportEventWithExitCode(LPSTR szFunction, const std::string& message, const int& exitCode) {
    // TODO: Write message file for ovms: https://learn.microsoft.com/en-us/windows/win32/eventlog/sample-message-text-file
    LPCTSTR lpszStrings[2];
    TCHAR Buffer[200];
    std::unique_ptr<SC_HANDLE, WinESHandleDeleter> hEventSource(RegisterEventSource(NULL, OvmsWindowsServiceManager::serviceName));
    if (hEventSource.get() != NULL) {
        StringCchPrintf(Buffer, 200, TEXT("%s failed with %d error: %s"), szFunction, exitCode, message.c_str());
        lpszStrings[0] = OvmsWindowsServiceManager::serviceName;
        lpszStrings[1] = Buffer;
        ReportEvent(hEventSource.get(),  // event log handle
            EVENTLOG_ERROR_TYPE,         // event type
            exitCode,                    // event category
            0,                           // event identifier
            NULL,                        // no security identifier
            2,                           // size of lpszStrings array
            0,                           // no binary data
            lpszStrings,                 // array of strings
            NULL);                       // no binary data

    } else {
        DEBUG_LOG("RegisterEventSource failed");
        DEBUG_LOG(std::system_category().message(GetLastError()));
    }
}

void OvmsWindowsServiceManager::serviceReportEventSuccess(const std::string& szFunction, const std::string& message) {
    serviceReportEventSuccess(const_cast<LPSTR>(szFunction.c_str()), message);
}

void OvmsWindowsServiceManager::serviceReportEventSuccess(LPSTR szFunction, const std::string& message) {
    // TODO: Write message file for ovms: https://learn.microsoft.com/en-us/windows/win32/eventlog/sample-message-text-file
    LPCTSTR lpszStrings[2];
    TCHAR Buffer[200];
    std::unique_ptr<SC_HANDLE, WinESHandleDeleter> hEventSource(RegisterEventSource(NULL, OvmsWindowsServiceManager::serviceName));
    if (hEventSource.get() != NULL) {
        StringCchPrintf(Buffer, 200, TEXT("%s success. Status: %s"), szFunction, message.c_str());
        lpszStrings[0] = OvmsWindowsServiceManager::serviceName;
        lpszStrings[1] = Buffer;
        ReportEvent(hEventSource.get(),  // event log handle
            EVENTLOG_SUCCESS,            // event type
            0,                           // event category
            0,                           // event identifier
            NULL,                        // no security identifier
            2,                           // size of lpszStrings array
            0,                           // no binary data
            lpszStrings,                 // array of strings
            NULL);                       // no binary data

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
    ovmsService->setup = false;

    //  Start OVMS and check for stop
    while (WaitForSingleObject(serviceStopEvent->handle, 0) != WAIT_OBJECT_0) {
        // Already started
        if (!ovmsService->setup) {
            std::pair<ovms::ServerSettingsImpl, ovms::ModelsSettingsImpl>* params = (std::pair<ovms::ServerSettingsImpl, ovms::ModelsSettingsImpl>*)lpParam;
            DEBUG_LOG("serviceWorkerThread: Starting ovms from parameters.");
            ovmsService->SetUp(params);
        }
        // Check thread not exited
        if (!ovmsService->isRunning()) {
            DEBUG_LOG("serviceWorkerThread: Server thread is not running.")
            break;
        }

        if (!ovmsService->started && ovmsService->checkModulesStarted()) {
            // Tell the service controller we are started
            OvmsWindowsServiceManager::setServiceRunningStatus();
            ovmsService->started = true;
        }
    }

    if (ovmsService->started || ovmsService->setup) {
        ovmsService->TearDown();
        DEBUG_LOG("serviceWorkerThread: Stopping ovms service.");
    } else {
        DEBUG_LOG("serviceWorkerThread: Ovms service could not be started.");
    }

    if (ovmsService->error) {
        DEBUG_LOG("serviceWorkerThread: Ovms start returned error.");
        DEBUG_LOG(ovmsService->error);
        serviceReportEventWithExitCode("serviceWorkerThread", "Ovms exited with error. Check windows events log and ovms server log for details.", ovmsService->error);
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

void OvmsWindowsServiceManager::setServiceStopStatusWithExitCode(const int& exitCode) {
    DWORD exitToError = static_cast<DWORD>(exitCode);
    // Map known exit code to known win errors for proper service status report on error
    // Check https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499- for details
    switch (exitCode) {
    case OVMS_EX_USAGE: {
        exitToError = ERROR_BAD_ARGUMENTS;
        break;
    }
    case OVMS_EX_OK: {
        exitToError = ERROR_BAD_ARGUMENTS;
        break;
    }
    case OVMS_EX_FAILURE: {
        exitToError = ERROR_INVALID_FUNCTION;
        break;
    }
    case OVMS_EX_WARNING: {
        exitToError = ERROR_INVALID_FUNCTION;
        break;
    }
    default:
        exitToError = ERROR_INVALID_FUNCTION;
    }

    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwCurrentState = SERVICE_STOPPED;
    serviceStatus.dwWin32ExitCode = exitToError;
    serviceStatus.dwCheckPoint = 1;
    if (SetServiceStatus(this->statusHandle->handle, &serviceStatus) == FALSE) {
        DEBUG_LOG("ServiceMain: SetServiceStatus returned error");
        serviceReportEvent("SetServiceStatus");
    }
    DEBUG_LOG("ServiceMain: SetServiceStatus stop with exit code");
}

void OvmsWindowsServiceManager::setServiceRunningStatus() {
    OvmsWindowsServiceManager::serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP;
    OvmsWindowsServiceManager::serviceStatus.dwCurrentState = SERVICE_RUNNING;
    OvmsWindowsServiceManager::serviceStatus.dwWin32ExitCode = 0;
    OvmsWindowsServiceManager::serviceStatus.dwCheckPoint = 0;

    if (SetServiceStatus(OvmsWindowsServiceManager::statusHandle->handle, &OvmsWindowsServiceManager::serviceStatus) == FALSE) {
        DEBUG_LOG("OvmsWindowsServiceManager: SetServiceStatus returned error");
        serviceReportEvent("SetServiceStatus");
    }
    DEBUG_LOG("OvmsWindowsServiceManager: SetServiceStatus running");
}

std::string OvmsWindowsServiceManager::getRegValue(const winreg::RegKey& key, const std::wstring& name, const DWORD& regType) {
    std::string retStr = "";
    switch (regType) {
    case REG_SZ: {
        if (auto testVal = key.TryGetStringValue(name)) {
            retStr = wstringToString(testVal.GetValue());
        }

        return retStr;
    }
    case REG_EXPAND_SZ: {
        if (auto testVal = key.TryGetExpandStringValue(name)) {
            retStr = wstringToString(testVal.GetValue());
        }

        return retStr;
    }
    case REG_MULTI_SZ: {
        if (auto testVal = key.TryGetMultiStringValue(name)) {
            for (auto elem : testVal.GetValue()) {
                retStr += wstringToString(elem) + ",";
            }
        }

        return retStr;
    }
    case REG_DWORD: {
        if (auto testVal = key.TryGetDwordValue(name)) {
            retStr = std::to_string(testVal.GetValue());
        }

        return retStr;
    }
    case REG_QWORD: {
        if (auto testVal = key.TryGetQwordValue(name)) {
            retStr = std::to_string(testVal.GetValue());
        }

        return retStr;
    }
    case REG_BINARY: {
        if (auto testVal = key.TryGetBinaryValue(name)) {
            for (auto elem : testVal.GetValue()) {
                retStr += std::to_string(elem) + ",";
            }
        }

        return retStr;
    }
    default:
        return retStr;
    }
    return retStr;
}

void OvmsWindowsServiceManager::logRegistryEntry(HKEY keyType, const std::wstring& keyPath) {
    DEBUG_LOG(wstringToString(keyPath));
    winreg::RegKey key{keyType, keyPath};
    std::vector<std::wstring> subKeyNames = key.EnumSubKeys();
    DEBUG_LOG("SubKeys:");
    for (const auto& s : subKeyNames) {
        DEBUG_LOG(wstringToString(s));
    }
    std::vector<std::pair<std::wstring, DWORD>> values = key.EnumValues();
    DEBUG_LOG("Values:");
    for (const auto& [valueName, valueType] : values) {
        std::stringstream ss2;
        // Try string reg value
        ss2 << "  [" << wstringToString(valueName) << "](" << wstringToString(winreg::RegKey::RegTypeToString(valueType)) << "): " << getRegValue(key, valueName, valueType);
        DEBUG_LOG(ss2.rdbuf());
    }
}

// Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\ovms
void OvmsWindowsServiceManager::setPythonPathRegistry() {
    try {
        const std::wstring ovmsServiceKey = L"SYSTEM\\CurrentControlSet\\Services\\ovms";
        winreg::RegKey key{HKEY_LOCAL_MACHINE, ovmsServiceKey};

        TCHAR szUnquotedPath[MAX_PATH];
        if (!GetModuleFileName(NULL, szUnquotedPath, MAX_PATH)) {
            DEBUG_LOG("setPythonPathRegistry, GetModuleFileName failed.");
            return;
        }
        //  create PATH=c:\test2\ovms\python;%PATH%
        std::string ovmsDirectory = std::filesystem::path(szUnquotedPath).parent_path().string();
        std::stringstream ss3;
        DEBUG_LOG("Adding Service Environment setting:")
        ss3 << "PATH=" << ovmsDirectory << "\\python;%PATH%";
        DEBUG_LOG(ss3.str());
        std::vector<std::wstring> multiString;
        multiString.push_back(stringToWstring(ss3.str()));
        key.Open(HKEY_LOCAL_MACHINE, ovmsServiceKey);
        key.SetMultiStringValue(L"Environment", multiString);
        key.Close();
        OvmsWindowsServiceManager::logRegistryEntry(HKEY_LOCAL_MACHINE, ovmsServiceKey);
    } catch (const std::exception& e) {
        DEBUG_LOG("setPythonPathRegistry: Add python path variable Failed:");
        DEBUG_LOG(e.what());
        std::cout << "Installing Openvino Model Server service PATH environment variable failed." << std::endl;
    }
    std::cout << "Installed Openvino Model Server service PATH environment variable." << std::endl;
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
    if (t && t->joinable())
        t->join();
    server.setShutdownRequest(0);
    this->started = false;
    this->setup = false;
    OvmsWindowsServiceManager::serviceReportEventSuccess("[INFO]Modules", "Openvino Model Server is stopped.");
}

int OvmsService::SetUp(std::pair<ovms::ServerSettingsImpl, ovms::ModelsSettingsImpl>* parameters) {
    DEBUG_LOG("OvmsService::SetUp");
    this->setup = true;
    OvmsWindowsServiceManager::serviceReportEventSuccess("[INFO]Modules", "Openvino Model Server is starting ...");
    t.reset(new std::thread([parameters, this]() {
        this->error = server.startServerFromSettings(parameters->first, parameters->second);
    }));
    return 0;
}

bool OvmsService::isReady() {
    return server.isReady();
}

bool OvmsService::isRunning() {
    // Check if server thread exited
    return (t && t->joinable() && server.getShutdownStatus() == 0 && server.getExitStatus() == 0);
}

bool OvmsService::isLive(const std::string& moduleName) {
    return server.isLive(moduleName);
}

bool OvmsService::checkModulesStarted() {
    // TODO: HF_MODEL_PULL_MODULE_LIVE - add logic for pull and start - now we are ready when pull starts in pull and start
    // Currently we return true on isReady = true;
    // Currently we return true on HF_MODEL_PULL_MODULE_LIVE = true;
    // Currently we return true on SERVABLES_CONFIG_MANAGER_MODULE_LIVE = true;
    static bool SERVER_READY = false;
    static bool PROFILER_MODULE_LIVE = false;
    static bool GRPC_SERVER_MODULE_LIVE = false;
    static bool HTTP_SERVER_MODULE_LIVE = false;
    static bool SERVABLE_MANAGER_MODULE_LIVE = false;
    static bool HF_MODEL_PULL_MODULE_LIVE = false;
    static bool METRICS_MODULE_LIVE = false;
    static bool PYTHON_INTERPRETER_MODULE_LIVE = false;
    static bool CAPI_MODULE_LIVE = false;
    static bool SERVABLES_CONFIG_MANAGER_MODULE_LIVE = false;

    if (!SERVABLE_MANAGER_MODULE_LIVE && this->isLive(ovms::SERVABLE_MANAGER_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service SERVABLE_MANAGER_MODULE is live.");
        SERVABLE_MANAGER_MODULE_LIVE = true;
    }
    // TODO: Add timeout for server ready ?
    if (!SERVER_READY && this->isReady()) {
        DEBUG_LOG("serviceWorkerThread: Ovms service is ready and running.");
        SERVER_READY = true;
        OvmsWindowsServiceManager::serviceReportEventSuccess("[INFO]Modules", "Openvino Model Server is ready.");
    }
    if (!PROFILER_MODULE_LIVE && this->isLive(ovms::PROFILER_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service PROFILER_MODULE is live.");
        PROFILER_MODULE_LIVE = true;
    }
    if (!GRPC_SERVER_MODULE_LIVE && this->isLive(ovms::GRPC_SERVER_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service GRPC_SERVER_MODULE is live.");
        GRPC_SERVER_MODULE_LIVE = true;
        OvmsWindowsServiceManager::serviceReportEventSuccess("[INFO]Modules", "Openvino Model Server GRPC module is live.");
    }
    if (!HTTP_SERVER_MODULE_LIVE && this->isLive(ovms::HTTP_SERVER_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service HTTP_SERVER_MODULE is live.");
        HTTP_SERVER_MODULE_LIVE = true;
        OvmsWindowsServiceManager::serviceReportEventSuccess("[INFO]Modules", "Openvino Model Server HTTP module is live.");
    }
    if (!METRICS_MODULE_LIVE && this->isLive(ovms::METRICS_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service METRICS_MODULE is live.");
        METRICS_MODULE_LIVE = true;
    }
    if (!PYTHON_INTERPRETER_MODULE_LIVE && this->isLive(ovms::PYTHON_INTERPRETER_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service PYTHON_INTERPRETER_MODULE is live.");
        PYTHON_INTERPRETER_MODULE_LIVE = true;
    }
    if (!CAPI_MODULE_LIVE && this->isLive(ovms::CAPI_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service CAPI_MODULE is live.");
        CAPI_MODULE_LIVE = true;
    }
    if (!HF_MODEL_PULL_MODULE_LIVE && this->isLive(ovms::HF_MODEL_PULL_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service HF_MODEL_PULL_MODULE is live.");
        HF_MODEL_PULL_MODULE_LIVE = true;
        return true;
    }
    if (!SERVABLES_CONFIG_MANAGER_MODULE_LIVE && this->isLive(ovms::SERVABLES_CONFIG_MANAGER_MODULE_NAME)) {
        DEBUG_LOG("serviceWorkerThread: Ovms service SERVABLES_CONFIG_MANAGER_MODULE is live.");
        SERVABLES_CONFIG_MANAGER_MODULE_LIVE = true;
        return true;
    }

    return SERVER_READY;
}

WinServiceStatusWrapper::WinServiceStatusWrapper() {
    handle = NULL;
}
WinServiceStatusWrapper::~WinServiceStatusWrapper() {
    std::stringstream ss2;
    ss2 << "WinServiceStatusWrapper: closing handle: " << handle;
    DEBUG_LOG(ss2.str());
    if (handle != NULL && handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
}

WinServiceEventWrapper::WinServiceEventWrapper() {
    handle = INVALID_HANDLE_VALUE;
}
WinServiceEventWrapper::~WinServiceEventWrapper() {
    std::stringstream ss2;
    ss2 << "WinServiceEventWrapper: closing handle: " << handle;
    DEBUG_LOG(ss2.str());
    if (handle != NULL && handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
}

}  // namespace ovms_service
