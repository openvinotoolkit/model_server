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
#ifdef __linux__
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#elif _WIN32
#include <winsock2.h>
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/network_utils.hpp"
#include "src/logging.hpp"

#include "test_utils.hpp"

TEST(NetworkUtils, IsPortAvailable_Positive) {
    uint64_t availablePort = 12345;
    EXPECT_TRUE(ovms::isPortAvailable(availablePort));
}

TEST(NetworkUtils, IsPortAvailable_Negative) {
    std::string portString = "9000";
    uint64_t takenPort = 0;
    int tryCount = 3;
#ifdef __linux__
    int s = -1;
    while (tryCount--) {
        s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) {
            SPDLOG_ERROR("Failed to create socket for test: {}", s);
            continue;
        }
        randomizePort(portString);
        takenPort = std::stoi(portString);
        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(takenPort);
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            SPDLOG_DEBUG("Failed to bind socket for test, port: {}", takenPort);
            close(s);
            s = -1;
            continue;
        } else {
            SPDLOG_DEBUG("Successfully bound socket for test, port: {}", takenPort);
        }
        break;
    }
    ASSERT_NE(s, -1) << "Failed to bind to any port after multiple attempts";
    EXPECT_FALSE(ovms::isPortAvailable(takenPort));
    close(s);
#elif _WIN32
    WSADATA wsaData;
    ASSERT_EQ(WSAStartup(MAKEWORD(2, 2), &wsaData), 0) << "WSAStartup failed";
    SOCKET s = INVALID_SOCKET;
    while (tryCount--) {
        s = socket(AF_INET, SOCK_STREAM, 0);
        if (s == INVALID_SOCKET) {
            SPDLOG_ERROR("Failed to create socket for test");
            continue;
        }
        randomizePort(portString);
        takenPort = std::stoi(portString);
        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(takenPort);
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            closesocket(s);
            s = INVALID_SOCKET;
            continue;
        } else {
            SPDLOG_DEBUG("Successfully bound socket for test, port: {}", takenPort);
        }
        break;
    }
    ASSERT_NE(s, INVALID_SOCKET) << "Failed to bind to any port after multiple attempts";
    EXPECT_FALSE(ovms::isPortAvailable(takenPort));
    closesocket(s);  // Clean up
    WSACleanup();
#endif
}
