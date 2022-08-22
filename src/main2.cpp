#include <future>
#include <memory>
#include <iostream>
#include <utility>

#include "server.hpp"

#include "modelmanager.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "servablemanagermodule.hpp"

using ovms::Server;

int main(int argc, char** argv) {
    Server& server = Server::instance();
    std::thread t([&server, &argv, &argc](){
                    std::cout << server.start(argc, argv) << std::endl;
                    });
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << __LINE__ << "AERO" << std::endl;
    // get model instance and have a lock on reload
    std::shared_ptr<ovms::ModelInstance> instance;
    std::unique_ptr<ovms::ModelInstanceUnloadGuard> modelInstanceUnloadGuardPtr;
    auto module = server.getModule(ovms::SERVABLE_MANAGER_MODULE_NAME);
    std::cout << __LINE__ << "AERO" << std::endl;
    if (nullptr == module) {
        return 0;
    }
    std::cout << __LINE__ << "AERO" << std::endl;
    auto servableManagerModule = dynamic_cast<const ovms::ServableManagerModule*>(module);
    auto& manager = servableManagerModule->getServableManager();
    manager.getModelInstance("dummy", 0, instance, modelInstanceUnloadGuardPtr); // status code
    float a[10] = {1,2,3,4,5,6,7,8,9,10};
    float b[10] = {1,2,3,4,5,6,7,8,9,10};
    std::cout << __LINE__ << "AERO" << std::endl;
    instance->infer(a,b);
    std::cout << __LINE__ << "AERO" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;
    std::cout << __LINE__ << "AERO" << std::endl;
    t.join();
    std::cout << __LINE__ << "AERO" << std::endl;
    std::cout << __LINE__ << "AERO" << std::endl;
    return 0;
}
