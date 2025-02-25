#pragma once
namespace ovms {
template <typename TensorType>
Status convertStringRequestToOVTensor2D(
    const TensorType& src,
    ov::Tensor& tensor,
    const std::string* buffer) {
    OVMS_PROFILE_FUNCTION();
    if (buffer != nullptr) {
        return convertStringRequestFromBufferToOVTensor2D(src, tensor, buffer);
    }
    int batchSize = getBinaryInputsSize(src);
    size_t maxStringLength = 0;
    for (int i = 0; i < batchSize; i++) {
        maxStringLength = std::max(maxStringLength, getBinaryInput(src, i).size());
    }
    size_t width = maxStringLength + 1;
    tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape{static_cast<size_t>(batchSize), width});
    for (int i = 0; i < batchSize; i++) {
        std::memcpy(
            tensor.data<unsigned char>() + i * width,
            reinterpret_cast<const unsigned char*>(getBinaryInput(src, i).c_str()),
            getBinaryInput(src, i).size());
        for (size_t j = getBinaryInput(src, i).size(); j < width; j++) {
            tensor.data<unsigned char>()[i * width + j] = 0;
        }
    }
    return StatusCode::OK;
}
} // namespace ovms
