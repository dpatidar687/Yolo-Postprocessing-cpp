
#include "ort_utility.hpp"

namespace Ort
{
ObjectDetectionOrtSessionHandler::ObjectDetectionOrtSessionHandler(
    const uint16_t numClasses,            //
    const std::string& modelPath,         //
    const std::string& provider,          //
    const std::optional<size_t>& gpuIdx,  //
    const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, provider, gpuIdx, inputShapes)
{
}

ObjectDetectionOrtSessionHandler::~ObjectDetectionOrtSessionHandler()
{
}

}  // namespace Ort
