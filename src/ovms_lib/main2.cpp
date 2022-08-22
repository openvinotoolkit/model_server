#include <future>
#include <iostream>

#include "server.hpp"

using ovms::Server;

int main(int argc, char** argv) {
    Server& server = Server::instance();
    std::thread t([&server, &argv, &argc](){
                    std::cout << server.start(argc, argv) << std::endl;
                    });
    t.join();
    return 0;
}
