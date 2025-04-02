//****************************************************************************
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
#include "network_utils.hpp"

//#include <signal.h>
//#include <stdlib.h>
#ifdef __linux__
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#elif _WIN32
#include <winsock2.h>
#endif

#include "logging.hpp"

namespace ovms {
#ifdef __linux__
bool isPortAvailable(uint64_t port) {
    struct sockaddr_in addr;
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == -1) {
        return false;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(s);
        return false;
    }
    close(s);
    return true;
}
#else  //  not __linux__

struct WSAStartupCleanupGuard {
    ~WSAStartupCleanupGuard() {
        WSACleanup();
    }
};
struct SocketOpenCloseGuard {
    SOCKET socket;
    SocketOpenCloseGuard(SOCKET socket) :
        socket(socket) {}
    ~SocketOpenCloseGuard() {
        closesocket(socket);
    }
};
bool isPortAvailable(uint64_t port) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        SPDLOG_ERROR("WSAStartup error.");
        return false;
    }
    WSAStartupCleanupGuard wsaGuard;
    // Create a socket
    this->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (this->sock == INVALID_SOCKET) {
        SPDLOG_ERROR("INVALID_SOCKET error.");
        return false;
    }

    // Bind to port
    sockaddr_in addr;
    addr.sin_family = AF_INET;
#pragma warning(disable : 4996)
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(port);
    SocketOpenCloseGuard socketGuard(this->sock);
    if (bind(this->sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        SPDLOG_ERROR("Bind port {} error: {}", port, WSAGetLastError());
        return false;
    }
    return true;
}
#endif  //  not __linux__
}  // namespace ovms
