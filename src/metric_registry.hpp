#include <prometheus/counter.h>
#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

namespace ovms {

class MetricRegistry {
    std::shared_ptr<prometheus::Registry> registry;
    prometheus::Family<prometheus::Counter>& grpcRequestsTotal;

    MetricRegistry() :
        registry(std::make_shared<prometheus::Registry>()),
        grpcRequestsTotal(prometheus::BuildCounter()
            .Name("grpc_requests_total")
            .Help("Total number of gRPC requests")
            .Register(*this->registry)) {
    }

    MetricRegistry(const MetricRegistry&) = delete;
    MetricRegistry(MetricRegistry&&) = delete;


public:
    static MetricRegistry& getInstance() {
        static MetricRegistry instance;
        return instance;
    }

    prometheus::Registry& getRegistry() {
        return *this->registry;
    }

    prometheus::Family<prometheus::Counter>& getGrpcRequestsTotalCounter() {
        return this->grpcRequestsTotal;
    }

    std::string serializeToString() const {
        prometheus::TextSerializer serializer;
        return serializer.Serialize(this->registry->Collect());
    }
};

}  // namespace ovms
