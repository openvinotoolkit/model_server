//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <cstdio>

#include <ie_iextension.h>
#include <ngraph/opsets/opset.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class OvmsOperation : public ILayerExecImpl {
public:
    explicit OvmsOperation(const std::shared_ptr<ngraph::Node>& node) {
        ::printf("OvmsOperation(node)\n");
    }
    StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc* resp) noexcept override;
    StatusCode init(LayerConfig& config, ResponseDesc* resp) noexcept override;
    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc* resp) noexcept override;

private:
    int64_t add;
    ngraph::Shape inShape;
    ngraph::Shape outShape;
    std::string error;
};

StatusCode OvmsOperation::getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc* resp) noexcept {
    ::printf("CPU_EXTENSIONS: getSupportedConfigurations()\n");
    return OK;
}

StatusCode OvmsOperation::init(LayerConfig& config, ResponseDesc* resp) noexcept {
    ::printf("CPU_EXTENSIONS: init()\n");
    return OK;
}

StatusCode OvmsOperation::execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc* resp) noexcept {
    ::printf("CPU_EXTENSIONS: execute()\n");
    return OK;
}
class INFERENCE_ENGINE_API_CLASS(OvmsMinimalCpuExtension) :
    public IExtension {
public:
    OvmsMinimalCpuExtension() = default;
    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType);

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        ::printf("OvmsMinimalCpuExtension::GetVersion()\n");
        static InferenceEngine::Version ExtensionDescription = {
            {1, 0},  // extension API version
            "1.0",
            "template_ext"  // extension description message
        };
        versionInfo = &ExtensionDescription;
    }

    void Unload() noexcept override { ::printf("OvmsMinimalCpuExtension::Unload()\n"); }
    void Release() noexcept override {
        ::printf("OvmsMinimalCpuExtension::Release()\n");
        delete this;
    }

    std::map<std::string, ngraph::OpSet> getOpSets() override;
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override;
};

std::map<std::string, ngraph::OpSet> OvmsMinimalCpuExtension::getOpSets() {
    ::printf("OvmsMinimalCpuExtension::getOpSets()\n");
    std::map<std::string, ngraph::OpSet> opsets;
    ngraph::OpSet opset;
    //    opset.insert<OvmsOperation>();
    opsets["custom_opset"] = opset;
    return opsets;
}

std::vector<std::string> OvmsMinimalCpuExtension::getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
    ::printf("OvmsMinimalCpuExtension::getImplTypes()\n");
    if (std::dynamic_pointer_cast<OvmsOperation>(node)) {
        ::printf("OvmsMinimalCpuExtension::getImplTypes() -> dyncast ok, returning 'CPU'\n");
        return {"CPU"};
    }
    ::printf("OvmsMinimalCpuExtension::getImplTypes() -> dyncast failed, returning empty vector.\n");
    return {};
}

InferenceEngine::ILayerImpl::Ptr OvmsMinimalCpuExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    ::printf("OvmsMinimalCpuExtension::getImplementation()\n");
    if (std::dynamic_pointer_cast<OvmsOperation>(node) && implType == "CPU") {
        ::printf("OvmsMinimalCpuExtension::getgetImplementation() -> dyncast ok, returning new OvmsOperation.\n");
        return std::make_shared<OvmsOperation>(node);
    }
    ::printf("OvmsMinimalCpuExtension::getgetImplementation() -> dyncast failed, returning nullptr.\n");
    return nullptr;
}

// Exported function
INFERENCE_EXTENSION_API(StatusCode)
CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept {
    try {
        ::printf("CreateExtension()\n");
        ext = new OvmsMinimalCpuExtension;
        ::printf("CreateExtension - ok!\n");
        return OK;
    } catch (std::exception& ex) {
        ::printf("CreateExtension exception: %s\n", ex.what());
        if (resp) {
            std::string err = ((std::string) "Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return GENERAL_ERROR;
    } catch (...) {
        ::printf("CreateExtension exception!\n");
        return GENERAL_ERROR;
    }
}

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
