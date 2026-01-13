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
#include "cmd_exec.hpp"

#include <iostream>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cctype>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

#include "src/utils/env_guard.hpp"

namespace ovms {

#ifndef _WIN32
// Internal helper to parse command string into arguments (handles quoted strings)
// Only needed on Linux for execvp which requires an argument array
static std::vector<std::string> parseArguments(const std::string& input) {
    std::vector<std::string> args;
    std::string current;
    bool inDoubleQuotes = false;
    bool inSingleQuotes = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '"' && !inSingleQuotes) {
            inDoubleQuotes = !inDoubleQuotes;
        } else if (c == '\'' && !inDoubleQuotes) {
            inSingleQuotes = !inSingleQuotes;
        } else if (std::isspace(static_cast<unsigned char>(c)) && !inDoubleQuotes && !inSingleQuotes) {
            if (!current.empty()) {
                args.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        args.push_back(current);
    }
    return args;
}
#endif

// Internal secure execution - bypasses shell to prevent command injection
static std::string exec_secure_internal(const std::string& command,
    int& returnCode,
    bool setUtf8Encoding = false) {
    std::string result;
    returnCode = -1;
    std::unique_ptr<EnvGuard> envGuard;
    if (setUtf8Encoding) {
        envGuard = std::make_unique<EnvGuard>();
        envGuard->set("PYTHONIOENCODING", "utf-8");
    }

#ifdef _WIN32
    // Windows: CreateProcess doesn't use a shell, so we can pass the command directly
    // No shell metacharacter interpretation occurs
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        return "Error: CreatePipe failed.";
    }
    SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = {0};
    si.cb = sizeof(STARTUPINFOA);
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;
    si.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi = {0};

    // CreateProcess takes a mutable string, make a copy
    std::string cmdCopy = command;
    if (!CreateProcessA(NULL, const_cast<char*>(cmdCopy.c_str()),
            NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(hReadPipe);
        CloseHandle(hWritePipe);
        return "Error: CreateProcess failed.";
    }

    CloseHandle(hWritePipe);

    char buffer[256];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buffer[bytesRead] = '\0';
        result += buffer;
    }

    CloseHandle(hReadPipe);
    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exitCode;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    returnCode = static_cast<int>(exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

#else  // Linux
    // Linux: Must use fork/execvp to avoid shell interpretation
    // Parse command into arguments for execvp
    std::vector<std::string> args = parseArguments(command);
    if (args.empty()) {
        return "Error: empty command.";
    }
    std::string executable = args[0];
    args.erase(args.begin());

    int pipefd[2];
    if (pipe(pipefd) == -1) {
        return "Error: pipe creation failed.";
    }

    pid_t pid = fork();
    if (pid == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        return "Error: fork failed.";
    }

    if (pid == 0) {
        // Child process
        close(pipefd[0]);  // Close read end

        // Redirect stdout and stderr to pipe
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        // Build argv array
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        execvp(executable.c_str(), argv.data());

        // If execvp returns, it failed
        _exit(127);
    } else {
        // Parent process
        close(pipefd[1]);  // Close write end

        char buffer[256];
        ssize_t bytesRead;
        while ((bytesRead = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
            buffer[bytesRead] = '\0';
            result += buffer;
        }
        close(pipefd[0]);

        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            returnCode = WEXITSTATUS(status);
        }
    }
#endif

    return result;
}

std::string exec_cmd(const std::string& command, int& returnCode) {
    return exec_secure_internal(command, returnCode, false);
}

std::string exec_cmd_utf8(const std::string& command, int& returnCode) {
    return exec_secure_internal(command, returnCode, true);
}

}  // namespace ovms
