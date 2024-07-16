
#include "Utility.hpp"
#include "ort_utility.hpp"

#include <cuda_provider_options.h>
#include <openvino_provider_factory.h>
#include <tensorrt_provider_options.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unordered_map>

namespace
{
std::string toString(const ONNXTensorElementDataType dataType)
{
    switch (dataType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            return "float";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
            return "uint8_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            return "int8_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
            return "uint16_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
            return "int16_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            return "int32_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            return "int64_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
            return "string";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            return "bool";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            return "float16";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            return "double";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
            return "uint32_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
            return "uint64_t";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: {
            return "complex with float32 real and imaginary components";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: {
            return "complex with float64 real and imaginary components";
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
            return "complex with float64 real and imaginary components";
        }
        default:
            return "undefined";
    }
}
}  // namespace

namespace Ort
{
//-----------------------------------------------------------------------------//
// OrtSessionHandlerIml Definition
//-----------------------------------------------------------------------------//

class OrtSessionHandler::OrtSessionHandlerIml
{
 public:
    OrtSessionHandlerIml(const std::string& modelPath,         //
                         const std::string& provider,          //
                         const std::optional<size_t>& gpuIdx,  //
                         const std::optional<std::vector<std::vector<int64_t>>>& inputShapes);
    ~OrtSessionHandlerIml();

    std::vector<DataOutputType> operator()(const std::vector<float*>& inputData, std::vector<std::pair<std::string, std::vector<int64_t>>> output_map);
    std::vector<DataOutputType> operator()(const std::vector<float*>& inputData);

 private:
    void initSession(const std::string& provider);
    void initModelInfo();

 private:
    std::string m_modelPath;

    Ort::Session m_session;
    Ort::Env m_env;
    Ort::AllocatorWithDefaultOptions m_ortAllocator;

    std::optional<size_t> m_gpuIdx;

    std::vector<std::vector<int64_t>> m_inputShapes;
    std::vector<std::vector<int64_t>> m_outputShapes;

    std::vector<int64_t> m_inputTensorSizes;
    std::vector<int64_t> m_outputTensorSizes;

    uint8_t m_numInputs;
    uint8_t m_numOutputs;

    std::vector<char*> m_inputNodeNames;
    std::vector<AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<char*> m_outputNodeNames;
    std::vector<AllocatedStringPtr> outputNodeNameAllocatedStrings;

    bool m_inputShapesProvided = false;
};

//-----------------------------------------------------------------------------//
// OrtSessionHandler
//-----------------------------------------------------------------------------//

OrtSessionHandler::OrtSessionHandler(const std::string& modelPath,         //
                                     const std::string& provider,  //
                                     const std::optional<size_t>& gpuIdx,  //
                                     const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : m_piml(std::make_unique<OrtSessionHandlerIml>(modelPath,  //
                                                    provider,     //
                                                    gpuIdx,     //
                                                    inputShapes))
{
}

OrtSessionHandler::~OrtSessionHandler() = default;

std::vector<OrtSessionHandler::DataOutputType> OrtSessionHandler::operator()(const std::vector<float*>& inputImgData)
{
    return this->m_piml->operator()(inputImgData);
}

std::vector<OrtSessionHandler::DataOutputType> OrtSessionHandler::operator()(const std::vector<float*>& inputImgData, std::vector<std::pair<std::string, std::vector<int64_t>>> output_map ) // Overloading the operator for output layer mapping correction
{
    return this->m_piml->operator()(inputImgData, output_map);
}

//-----------------------------------------------------------------------------//
// piml class implementation
//-----------------------------------------------------------------------------//

OrtSessionHandler::OrtSessionHandlerIml::OrtSessionHandlerIml(
    const std::string& modelPath,         //
    const std::string& provider,          //
    const std::optional<size_t>& gpuIdx,  //
    const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : m_modelPath(modelPath)
    , m_session(nullptr)
    , m_env(nullptr)
    , m_ortAllocator()
    , m_gpuIdx(gpuIdx)
    , m_inputShapes()
    , m_outputShapes()
    , m_numInputs(0)
    , m_numOutputs(0)
    , m_inputNodeNames()
    , m_outputNodeNames()
{
    this->initSession(provider);

    if (inputShapes.has_value()) {
        m_inputShapesProvided = true;
        m_inputShapes = inputShapes.value();
    }

    this->initModelInfo();
}

OrtSessionHandler::OrtSessionHandlerIml::~OrtSessionHandlerIml()
{

    for (auto& elem : this->m_inputNodeNames) {
        free(elem);
        elem = nullptr;
    }
    this->m_inputNodeNames.clear();

    for (auto& elem : this->m_outputNodeNames) {
        free(elem);
        elem = nullptr;
    }
    this->m_outputNodeNames.clear();


}

void OrtSessionHandler::OrtSessionHandlerIml::initSession(const std::string& old_provider)
{
    m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions sessionOptions;

    // TODO: need to take care of the following line as it is related to CPU
    // consumption using openmp
    // sessionOptions.SetIntraOpNumThreads(1);

    // Mapping of old provider to new providers from UI
    const std::unordered_map<std::string, std::string> provider_mapping = {
      {"gstreamer-gpu", "EBDS"},
      {"onnx-tensorrt", "EBGPU"},
      {"onnx-cpu", "EBCPU"},
      {"onnx-gpu", "EBGPU"},
      {"onnx-openvino-cpu", "EBVINOCPU"},
      {"onnx-openvino-gpu", "EBVINOIGPU"},

    };
    std::cout<<"-------------------------> OLD PROVIDER ---"<<old_provider<<std::endl;
    const std::string provider = provider_mapping.at(old_provider);

    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == "EBGPU") {
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, m_gpuIdx.value()));
    } else if (provider.find("EBTRT") == 0 || provider.find("EBDS") == 0) {
      // Set Tensorrt options
      OrtTensorRTProviderOptions trt_options{};
      trt_options.trt_max_workspace_size = 1073741824; // 1GB
      trt_options.trt_max_partition_iterations = 10000;
      trt_options.trt_min_subgraph_size = 1;
      trt_options.trt_engine_cache_enable = 1;
      trt_options.trt_engine_cache_path = "./docker/models_cache/";

      // For now use only FP16
      trt_options.trt_fp16_enable = 1;

      trt_options.device_id = m_gpuIdx.value();
      sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
    } else if (provider.find("EBVINO") == 0) {

      sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      // Set OpenVino options
      OrtOpenVINOProviderOptions vino_options{};
      vino_options.device_id = "";
      vino_options.num_of_threads = 8;


      if (provider.find("gpu") != std::string::npos)
        vino_options.device_type = "GPU_FP16";
      else
        vino_options.device_type = "CPU_FP16";


      sessionOptions.AppendExecutionProvider_OpenVINO(vino_options);
    } else if (provider != "EBCPU") {
      std::cerr << "Provider should be one of: EBCPU, EBGPU, EBTRT, EBVINOCPU or EBVINOIGPU. Got "
                << provider << "." << std::endl;
      return;
    }

    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    m_session = Ort::Session(m_env, m_modelPath.c_str(), sessionOptions);
    m_numInputs = m_session.GetInputCount();
    DEBUG_LOG("Model number of inputs: %d\n", m_numInputs);

    m_inputNodeNames.reserve(m_numInputs);
    inputNodeNameAllocatedStrings.reserve(m_numInputs);
    m_inputTensorSizes.reserve(m_numInputs);

    m_numOutputs = m_session.GetOutputCount();
    DEBUG_LOG("Model number of outputs: %d\n", m_numOutputs);

    m_outputNodeNames.reserve(m_numOutputs);
    m_outputTensorSizes.reserve(m_numOutputs);
    outputNodeNameAllocatedStrings.reserve(m_numOutputs);
}

void OrtSessionHandler::OrtSessionHandlerIml::initModelInfo()
{
    for (int i = 0; i < m_numInputs; i++) {
        if (!m_inputShapesProvided) {
            Ort::TypeInfo typeInfo = m_session.GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

            m_inputShapes.emplace_back(tensorInfo.GetShape());
        }

        const auto& curInputShape = m_inputShapes[i];

        m_inputTensorSizes.emplace_back(
            std::accumulate(std::begin(curInputShape), std::end(curInputShape), 1, std::multiplies<int64_t>()));

        //auto inputName = m_session.GetInputNameAllocated(i, m_ortAllocator);
        inputNodeNameAllocatedStrings.push_back(m_session.GetInputNameAllocated(i, m_ortAllocator));
        m_inputNodeNames.emplace_back(inputNodeNameAllocatedStrings.back().get());
    }

    {
#if ENABLE_DEBUG
        std::stringstream ssInputs;
        ssInputs << "Model input shapes: ";
        ssInputs << m_inputShapes << std::endl;
        ssInputs << "Model input node names: ";
        ssInputs << m_inputNodeNames << std::endl;
        DEBUG_LOG("%s\n", ssInputs.str().c_str());
#endif
    }

    for (int i = 0; i < m_numOutputs; ++i) {
        Ort::TypeInfo typeInfo = m_session.GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        std::vector<int64_t>  shape = tensorInfo.GetShape();
        shape.erase(shape.begin()); // removing the batch dimension

        m_outputShapes.emplace_back(shape);

        //auto outputName = m_session.GetOutputNameAllocated(i, m_ortAllocator);
        outputNodeNameAllocatedStrings.push_back(m_session.GetOutputNameAllocated(i, m_ortAllocator));
        m_outputNodeNames.emplace_back(outputNodeNameAllocatedStrings.back().get());
    }

    {
#if ENABLE_DEBUG
        std::stringstream ssOutputs;
        ssOutputs << "Model output shapes: ";
        ssOutputs << m_outputShapes << std::endl;
        ssOutputs << "Model output node names: ";
        ssOutputs << m_outputNodeNames << std::endl;
        DEBUG_LOG("%s\n", ssOutputs.str().c_str());
#endif
    }
}

std::vector<OrtSessionHandler::DataOutputType> OrtSessionHandler::OrtSessionHandlerIml::
operator()(const std::vector<float*>& inputData)
{
    if (m_numInputs != inputData.size()) {
        throw std::runtime_error("Mismatch size of input data\n");
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(m_numInputs);

    for (int i = 0; i < m_numInputs; ++i) {
        inputTensors.emplace_back(std::move(
            Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(inputData[i]), m_inputTensorSizes[i],
                                            m_inputShapes[i].data(), m_inputShapes[i].size())));
    }

    auto outputTensors = m_session.Run(Ort::RunOptions{nullptr}, m_inputNodeNames.data(), inputTensors.data(),
                                       m_numInputs, m_outputNodeNames.data(), m_numOutputs);

    assert(outputTensors.size() == m_numOutputs);
    std::vector<DataOutputType> outputData;
    outputData.reserve(m_numOutputs);

    int count = 1;
    for (auto& elem : outputTensors) {
        DEBUG_LOG("type of input %d: %s", count++, toString(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str());
        outputData.emplace_back(
            std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));
    }

    return outputData;
}

std::vector<OrtSessionHandler::DataOutputType> OrtSessionHandler::OrtSessionHandlerIml::
operator()(const std::vector<float*>& inputData, std::vector<std::pair<std::string, std::vector<int64_t>>> output_map)
{
    if (m_numInputs != inputData.size()) {
        throw std::runtime_error("Mismatch size of input data\n");
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(m_numInputs);

    for (int i = 0; i < m_numInputs; ++i) {
        inputTensors.emplace_back(std::move(
            Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(inputData[i]), m_inputTensorSizes[i],
                                            m_inputShapes[i].data(), m_inputShapes[i].size())));
    }

    auto outputTensors = m_session.Run(Ort::RunOptions{nullptr}, m_inputNodeNames.data(), inputTensors.data(),
                                       m_numInputs, m_outputNodeNames.data(), m_numOutputs);

    assert(outputTensors.size() == m_numOutputs);
    std::vector<DataOutputType> outputData;
    outputData.reserve(m_numOutputs);

    std::vector<u_int64_t> indices;

    bool flag {true};

    // Check if names are in correct order. If not, correct the ordering of output
    for (auto i : output_map){
        std::string name = i.first.c_str();
        auto index =std::find(m_outputNodeNames.begin(), m_outputNodeNames.end(), name);
        if (index==m_outputNodeNames.end()){
            flag = false;
            continue;
        }
        else {
            indices.push_back(index-m_outputNodeNames.begin());
            }
    }

    if (!flag){
    // If output names are not correct in spec.json, use output shapes. Check if shapes are in correct order. If not, correct the ordering of output
        indices.clear();
        for (auto i : output_map){
            std::vector<int64_t> shape = i.second;
            auto index = std::find(m_outputShapes.begin(), m_outputShapes.end(), shape);
            if (index==m_outputShapes.end()){
                flag = false;
                continue;
            }
            else {
                indices.push_back(index-m_outputShapes.begin());
                flag = true;
                }
        }
    }

    int count = 1;

    if (flag) {
        for (auto i : indices) {
            auto& elem = outputTensors[i];
            DEBUG_LOG("Correcting order of output according to the spec json")
            DEBUG_LOG("type of input %d: %s", count++, toString(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str());
            outputData.emplace_back(
                std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));
        }
    }
    else{
        // If both name and shape are not correctly defined in spec,, use order outputed by onnx
        for (auto& elem : outputTensors) {
            DEBUG_LOG("Using original onnx output order")
            DEBUG_LOG("type of input %d: %s", count++, toString(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str());
            outputData.emplace_back(
                std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));
        }
    }

    return outputData;
}
}  // namespace Otr
