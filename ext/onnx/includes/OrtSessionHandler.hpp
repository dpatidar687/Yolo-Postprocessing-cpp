#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Ort
{
class OrtSessionHandler
{
 public:
    // DataOutputType->(pointer to output data, shape of output data)
    using DataOutputType = std::pair<float*, std::vector<int64_t>>;

    OrtSessionHandler(const std::string& modelPath,  //
                      const std::string& provider,   //
                      const std::optional<size_t>& gpuIdx = std::nullopt,
                      const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);
    ~OrtSessionHandler();

    // multiple inputs, multiple outputs
    std::vector<DataOutputType> operator()(const std::vector<float*>& inputImgData);
    std::vector<DataOutputType> operator()(const std::vector<float*>& inputImgData, std::vector<std::pair<std::string, std::vector<int64_t>>> output_map);

 private:
    class OrtSessionHandlerIml;
    std::unique_ptr<OrtSessionHandlerIml> m_piml;
};
}  // namespace Ort
