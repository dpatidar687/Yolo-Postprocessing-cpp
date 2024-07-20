#include "ORT.h"

namespace mtx
{

    ORTInferenceEngine::ORTInferenceEngine(const std::string &model_path, const std::string &provider,
                                           const uint8_t &gpuIdx,
                                           const std::vector<std::vector<int64_t>> &inputShapes,
                                           const std::vector<std::vector<long int>> &outputShape, pre_processing preprocess) :

                                            session(nullptr),
                                            env(nullptr),
                                            input_tensor(std::vector<Ort::Value>()),
                                            _output_tensor(std::vector<Ort::Value>()),
                                            runOptions(nullptr),
                                            memoryInfo(nullptr),
                                            InferenceEngine(model_path, provider, inputShapes, outputShape, preprocess)
                                            
    {

        std::cout << "inside the constructor of ort .cpp file" << std::endl;
        std::string _name = "onnx_runtime" + std::to_string(rand());
        this->env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, _name.c_str());

        this->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

        this->sessionOptions.DisableMemPattern();
        this->sessionOptions.DisableCpuMemArena();
        this->sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        if (provider == "onnx-cpu")
        {
            this->sessionOptions = Ort::SessionOptions();
            // this->sessionOptions.SetIntraOpNumThreads(1);
        }
        else if (provider == "onnx-gpu")
        {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = gpuIdx;
            // cuda_options.arena_extend_strategy = 0;
            // cuda_options.do_copy_in_default_stream = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            // cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

            this->sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            this->sessionOptions.SetIntraOpNumThreads(1);
            this->sessionOptions.SetInterOpNumThreads(6);
        }

        this->session = Ort::Session(env, model_path.c_str(), sessionOptions);

        // print GPU information using ORT

        this->memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        getModelDetails(session);

        int nDevices;
        cudaGetDeviceCount(&nDevices);
        cudaDeviceProp prop;

        for (int i = 0; i < nDevices; i++)
        {
            cudaGetDeviceProperties(&prop, i);
            std::cout << "Device Number: " << i << std::endl;
            std::cout << "  Device name: " << prop.name << std::endl;
            std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
            std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
            std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
        }

        // release session memory
        Ort::detail::OrtRelease(runOptions);

        for(auto &tensor : input_tensors_details)
        {
            this->input_node_names.push_back(const_cast<char *>(tensor.first.c_str()));
        }

        for(auto &tensor : output_tensors_details)
        {
            this->output_node_names.push_back(const_cast<char *>(tensor.first.c_str()));
        }

        // getting input shapes information
    }

    void ORTInferenceEngine::getModelDetails(Ort::Session &session)
    {

        auto numInputs = session.GetInputCount();
        auto numOutputs = session.GetOutputCount();

        std::cout << "Number of inputs: " << numInputs << std::endl;

        this->input_node_names.reserve(numInputs);
        this->output_node_names.reserve(numOutputs);

        for (int i = 0; i < numInputs; i++)
        {
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_shape = tensor_info.GetShape();
            inputNodeNameAllocatedStrings.push_back(session.GetInputNameAllocated(i, this->allocator));
            std::string input_name = inputNodeNameAllocatedStrings.back().get();
            this->input_names.push_back(input_name);
            // this->input_node_names.push_back(const_cast<char *>(input_names.back().c_str()));

            input_shape[0] = this->BATCH_SIZE;
            input_tensors_details[input_name].name = input_name;
            std::cout << "\tInput name: " << input_name << std::endl;
            std::cout << "\tInput Shape: ";

            std::vector<long int> input_vector_shape;
            for (auto shape : input_shape)
            {
                if (shape < 0)
                {
                    input_vector_shape.clear();
                    std::cout << "Still Dynamic shape" << std::endl;
                    input_vector_shape = this->input_shapes[i];
                    for (auto &s : input_vector_shape)
                    {
                        input_tensors_details[input_name].element_count *= s;
                        std::cout << s << " ";
                    }
                    break;
                }
                input_vector_shape.push_back(shape);

                input_tensors_details[input_name].element_count *= shape;
                std::cout << shape << " ";
            }

            input_tensors_details[input_name].shape = input_vector_shape;
            std::cout << std::endl;
        }

        std::cout << "Number of outputs: " << numOutputs << std::endl;

        for (int i = 0; i < numOutputs; i++){
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_shape = tensor_info.GetShape();
            outputNodeNameAllocatedStrings.push_back(session.GetOutputNameAllocated(i, this->allocator));
            std::string output_name = outputNodeNameAllocatedStrings.back().get();

            //  = output_shape;
            output_shape[0] = this->BATCH_SIZE;
            output_tensors_details[output_name].name = output_name;

            std::cout << "\tOutput Name: " << output_name << std::endl;
            this->output_names.push_back(output_name);
            // this->output_node_names.push_back(const_cast<char *>(output_names.back().c_str()));

            std::vector<long int> output_vector_shape;
            for (auto shape : output_shape){
                if (shape < 0){
                    output_vector_shape.clear();
                    std::cout << "\tStill Dynamic shape" << std::endl;
                    output_vector_shape = this->output_shapes[i];
                    output_vector_shape[0] = this->BATCH_SIZE;
                    for (auto &s : output_vector_shape)
                    {
                        output_tensors_details[output_name].element_count *= s;
                        // std::cout << s << " ";
                    }
                    break;
                }
                output_vector_shape.push_back(shape);
                output_tensors_details[output_name].element_count *= shape;
                // std::cout << shape << " ";
            }

            std::cout << "\tOutput Shape: " ;

            for(auto &s : output_vector_shape)
            {
                std::cout << s << " ";
            }
            std::cout << "\n" <<std::endl;

            output_tensors_details[output_name].shape = output_vector_shape;
            tensor_details ot;
            ot.shape = output_vector_shape;
            ot.name = output_name;
            ot.element_count = output_tensors_details[output_name].element_count;
            ot.data = NULL;
            // memset(ot.data, 0, ot.element_count * sizeof(float));

            output_tensors[output_name] = ot;
        }

        std::cout << std::endl;
    }

    ORTInferenceEngine::~ORTInferenceEngine()
    {
        for (auto &tensor : output_tensors)
        {
            delete[] tensor.second.data;
        }
        this->session.release();
    }

    void ORTInferenceEngine::enqueue(float *data)
    {
        for (auto &vect : this->input_tensor)
        {
            // release Ort value
            vect.release();
        }

        for (auto &[name, details] : this->input_tensors_details)
        {
            std::cout << "Enqueuing input tensor: " << name << std::endl;
            std::cout << "\t" << details.element_count << std::endl;
            this->input_tensor.push_back(
                Ort::Value::CreateTensor<float>(
                    this->memoryInfo,
                    const_cast<float *>(data),
                    details.element_count,
                    details.shape.data(),
                    details.shape.size()));
        }

    }

    std::vector<tensor_details> ORTInferenceEngine::execute_network(){
        std::vector<tensor_details> returndata;
        auto _output_tensors_ = this->session.Run(this->runOptions,
                        this->input_node_names.data(),
                        const_cast<Ort::Value *>(this->input_tensor.data()),
                        input_tensors_details.size(),
                        this->output_node_names.data(),
                        output_tensors_details.size()
                    );

        // std::cout << "Inference completed" << std::endl;
        for (int i = 0; i < _output_tensors_.size(); i++){
            float *data = _output_tensors_[i].GetTensorMutableData<float>();
            int64_t num_elements = _output_tensors_[i].GetTensorTypeAndShapeInfo().GetElementCount();

            // std::cout << "Expected: " << output_tensors_details[output_node_names[i]].element_count << " Got: " << num_elements << std::endl;

            if (num_elements != output_tensors_details[output_node_names[i]].element_count){
                std::cout << "Number of elements in output tensor does not match the expected number of elements" << std::endl;
                std::cout << "Expected: " << output_tensors_details[output_node_names[i]].element_count << " Got: " << num_elements << std::endl;
                std::runtime_error("Number of elements in output tensor does not match the expected number of elements");
            }

            // memcpy(this->output_tensors[output_node_names[i]].data, data,output_tensors_details[output_node_names[i]].element_count * sizeof(float));
            this->output_tensors[output_node_names[i]].data = data;
            returndata.push_back(output_tensors[output_node_names[i]]);
        }

        this->input_tensor.clear();

        return returndata;
    }

    std::vector<tensor_details> ORTInferenceEngine::execute_async_network()
    {
        return execute_network();
    }

    std::unordered_map<std::string, std::vector<int64_t>> ORTInferenceEngine::get_output_shape()
    {
        std::unordered_map<std::string, std::vector<int64_t>> _output_shapes;
        for (auto &tensor : this->output_tensors_details)
        {
            _output_shapes[tensor.first] = tensor.second.shape;
        }
        return _output_shapes;
    }

    std::unordered_map<std::string, std::vector<int64_t>> ORTInferenceEngine::get_input_shape()
    {
        std::unordered_map<std::string, std::vector<int64_t>> _input_shapes;
        for (auto &tensor : this->input_tensors_details)
        {
            _input_shapes[tensor.first] = tensor.second.shape;
        }
        return _input_shapes;
    }

}
