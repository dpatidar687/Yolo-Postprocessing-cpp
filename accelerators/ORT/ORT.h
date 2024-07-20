#pragma once
#include "../models/InferenceEngine.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <cuda_provider_options.h>
#include <cuda_runtime.h>
#include "../models/types.h"
// #include "ext/onnx/includes/Utility.hpp"
// #include "ext/onnx/includes/ort_utility.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unordered_map>

typedef struct ort_tensor_details
{
    std::string name;
    std::vector<int64_t> shape;
    int64_t element_count = 1;
} ort_tensor_details;

namespace mtx
{
    
    class ORTInferenceEngine : public InferenceEngine
    {
    public:
        ORTInferenceEngine(const std::string &model_path, const std::string &provider,
                           const uint8_t &gpuIdx,
                           const std::vector<std::vector<long int>> &inputShapes,
                           const std::vector<std::vector<long int>> &outputShape, pre_processing preprocess = pre_processing());

        ~ORTInferenceEngine();

        void enqueue(float *data) override;
        std::vector<tensor_details> execute_network() override;
        std::vector<tensor_details> execute_async_network() override;
        std::unordered_map<std::string, std::vector<int64_t>> get_output_shape() override;
        std::unordered_map<std::string, std::vector<int64_t>> get_input_shape();

        static void async_callback_function(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr);


    private:
        bool infer_done;
        std::unordered_map<std::string, tensor_details> output_tensors;

        std::unordered_map<std::string, ort_tensor_details> input_tensors_details;
        std::unordered_map<std::string, ort_tensor_details> output_tensors_details;

        Ort::Session session;
        Ort::SessionOptions sessionOptions;
        Ort::Env env;

        std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
        std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;

        std::vector<std::string> input_names;
        std::vector<std::string> output_names;

        std::vector<char *> output_node_names;
        std::vector<char *> input_node_names;

        

        std::vector<Ort::Value >input_tensor;
        std::vector<Ort::Value> _output_tensor;

        Ort::RunOptions runOptions;
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memoryInfo;


        void getModelDetails(Ort::Session &session);
    };
}
