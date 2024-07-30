#include "TensorRT.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <cassert>

namespace mtx
{

    void TRTInferenceEngine::showAvailableDevices()
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);

        std::cout << std::endl;
        std::cout << "  Available GPU devices:";
        for (int device = 0; device < numGPUs; device++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            std::cout << "  Device Number: " << device << std::endl;
            std::cout << "  Device name: " << prop.name << std::endl;
            std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
            std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
            std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;

            this->deviceNames.push_back(std::string(prop.name));
        }
        std::cout << std::endl;
    }

    TRTInferenceEngine::TRTInferenceEngine(const std::string &model_path, const std::string provider, const std::vector<std::vector<long int>> &inputShapes, const std::vector<std::vector<long int>> &outputShapes, pre_processing preprocess, bool isSecondary) : isSecondary(isSecondary), InferenceEngine(model_path, provider, inputShapes, outputShapes, preprocess)
    {

        // Flag for secondary model inference
        needCpuOutput = true;
        if (isSecondary)
        {
            this->trt_options.maxBatchSize = std::min(8, int(4 * this->BATCH_SIZE));
            this->trt_options.optBatchSize = std::min(8, int(4 * this->BATCH_SIZE));
        }
        else
        {
            this->trt_options.maxBatchSize = 16; // std::min(32, (int)this->BATCH_SIZE);
            this->trt_options.optBatchSize = std::min(16, (int)this->BATCH_SIZE);
        }

        std::cout << "Converting onnx to trt engine..." << std::endl;
        bool overwrite = false;

    REBUILD:
        int buildStatus = buildTRT(model_path, overwrite);
        if (!buildStatus)
        {
            throw std::runtime_error("Unable to build TRT engine.");
        }
        int loadStatus = loadEngine();
        if (!loadStatus)
        {
            if (buildStatus == 2)
            {
                std::cout << "Could not load the trt engine found at disk" << std::endl;
                std::cout << "rebuilding the TRT engine...." << std::endl;
                overwrite = true;
                goto REBUILD;
            }
            throw std::runtime_error("Unable to load TRT engine.");
        }
        if (loadStatus == 2)
        {
            std::cout << "Found trt engine with lower max Batch" << std::endl;
            std::cout << "rebuilding the TRT engine...." << std::endl;
            overwrite = true;
            goto REBUILD;
        }
        bool status = allocateResources();
        if (!status)
            throw std::runtime_error("Could not allocate resources");
    }

    TRTInferenceEngine::~TRTInferenceEngine()
    {

        CHECK_CUDA_ERROR(cudaStreamSynchronize(*PreprocessCudaStream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(*InferCudaStream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(*PostprocessCudaStream));

        PreprocessCudaStream.reset();
        InferCudaStream.reset();
        PostprocessCudaStream.reset();

        for (auto &[name, ip_tensor] : ip_tensors)
        {
            std::cout << "Freeing tensor: " << name << std::endl;
            std::cout << "Size: " << ip_tensor.size << std::endl;
            CHECK_CUDA_ERROR(cudaFree(ip_tensor.buffer));
        }

        for (auto &[name, ip_tensor] : out_tensors)
        {
            CHECK_CUDA_ERROR(cudaFree(ip_tensor.buffer));
        }
    }

    int TRTInferenceEngine::buildTRT(std::string onnxModelPath, bool overwrite)
    {

        // Only regenerating the engine file if one with same options is not available
        const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
        this->trt_engine_path = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

        // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
        showAvailableDevices();

        if (static_cast<size_t>(this->trt_options.deviceIndex) >= deviceNames.size())
        {
            throw std::runtime_error("Error, provided device index is out of range!");
        }

        auto deviceName = deviceNames[this->trt_options.deviceIndex];
        // Remove spaces from the device name
        deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

        this->trt_engine_path += "." + deviceName;

        // Serialize the specified options into the filename
        if (this->trt_options.precision == Precision::FP16)
        {
            this->trt_engine_path += ".fp16";
        }
        else if (this->trt_options.precision == Precision::FP32)
        {
            this->trt_engine_path += ".fp32";
        }
        else
        {
            this->trt_engine_path += ".int8";
        }

        this->trt_engine_path = onnxModelPath.substr(0, filenamePos) + this->trt_engine_path;

        std::cout << "Searching for engine file with name: " << this->trt_engine_path << std::endl;

        if (doesFileExist(this->trt_engine_path) && !overwrite)
        {
            std::cout << "Engine found, not regenerating..." << std::endl;
            return 2;
        }

        if (!doesFileExist(onnxModelPath))
        {
            throw std::runtime_error("Could not find model at path: " + onnxModelPath);
        }

        // Was not able to find the engine file, generate...
        std::cout << "Engine not found, Generating TRT Engine..." << std::endl;

        // Create our engine builder.
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
        if (!builder)
        {
            return 0;
        }

        // Setting explicit batch flag as implicit batch trt model creation is deprecated
        auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            return 0;
        }

        // Create a parser for reading the onnx file.
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
        if (!parser)
        {
            return 0;
        }

        std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
        {
            throw std::runtime_error("Unable to read engine file");
        }

        // Parse the buffer we read into memory.
        auto parsed = parser->parse(buffer.data(), buffer.size());
        if (!parsed)
        {
            return 0;
        }

        // Ensure that all the inputs have the same batch size
        const auto numInputs = network->getNbInputs();
        if (numInputs < 1)
        {
            throw std::runtime_error("Error, model needs at least 1 input!");
        }
        const auto input0Batch = network->getInput(0)->getDimensions().d[0];
        for (int32_t i = 1; i < numInputs; ++i)
        {
            if (network->getInput(i)->getDimensions().d[0] != input0Batch)
            {
                throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
            }
        }

        // Check to see if the model supports dynamic batch size or not
        if (input0Batch == -1)
        {
            std::cout << "Model supports dynamic batch size" << std::endl;
        }
        else if (input0Batch == 1)
        {
            std::cout << "Model only supports fixed batch size of 1" << std::endl;
            // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
            if (this->trt_options.optBatchSize != input0Batch || this->trt_options.maxBatchSize != input0Batch)
            {
                throw std::runtime_error("Error, model only supports a fixed batch size of 1. Must set Options.optBatchSize and Options.maxBatchSize to 1");
            }
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            return 0;
        }

        nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();

        for (int32_t i = 0; i < numInputs; ++i)
        {
            // Must specify dimensions for all the inputs the model expects.
            const auto input = network->getInput(i);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
            int32_t inputC = inputDims.d[1];
            int32_t inputH = inputDims.d[2];
            int32_t inputW = inputDims.d[3];

            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(trt_options.optBatchSize, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(trt_options.maxBatchSize, inputC, inputH, inputW));
        }

        config->addOptimizationProfile(optProfile);

        // Set the precision level
        if (trt_options.precision == Precision::FP16)
        {
            // Ensure the GPU supports FP16 inference
            if (!builder->platformHasFastFp16())
            {
                throw std::runtime_error("Error: GPU does not support FP16 precision");
            }
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        // CUDA stream used for profiling by the builder.
        cudaStream_t profileStream;
        CHECK_CUDA_ERROR(cudaStreamCreate(&profileStream));
        config->setProfileStream(profileStream);

        // Building the engine
        std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            return 0;
        }

        // Write the engine to disk
        std::ofstream outfile(this->trt_engine_path, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

        std::cout << "Success, saved engine to " << this->trt_engine_path << std::endl;

        CHECK_CUDA_ERROR(cudaStreamDestroy(profileStream));
        return 1;
    }

    int TRTInferenceEngine::loadEngine()
    {
        // Read the serialized model from disk
        std::ifstream file(this->trt_engine_path, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
        {
            throw std::runtime_error("Unable to read engine file");
        }

        // Create a runtime to deserialize the engine file.
        this->m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
        if (!m_runtime)
        {
            return 0;
        }

        // Set the device index
        auto ret = cudaSetDevice(trt_options.deviceIndex);
        if (ret != 0)
        {
            int numGPUs;
            cudaGetDeviceCount(&numGPUs);
            auto errMsg = "Unable to set GPU device index to: " + std::to_string(trt_options.deviceIndex) +
                          ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
            throw std::runtime_error(errMsg);
        }

        // Create an engine, a representation of the optimized model.
        this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));

        if (!m_engine)
        {
            return 0;
        }

        // The execution context contains all of the state associated with a particular invocation
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

        if (!m_context)
        {
            return 0;
        }

        return 1;
    }
    bool TRTInferenceEngine::allocateResources()
    {

        // Preprocess stream used for data transfer and preprocessing
        this->PreprocessCudaStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);

        if (!this->PreprocessCudaStream || !this->PreprocessCudaStream->ptr())
            throw std::runtime_error("Unable to create preprocess cuda stream");

        // Infer stream
        this->InferCudaStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);
        // this->InferCudaStream_async = std::make_unique<CudaStream>(cudaStreamNonBlocking);

        // post process stream
        this->PostprocessCudaStream = std::make_unique<CudaStream>(cudaStreamNonBlocking);

        this->InputConsumedEvent = std::make_shared<CudaEvent>(cudaEventDisableTiming);

        this->InferCompleteEvent = std::make_shared<CudaEvent>(cudaEventDisableTiming | cudaEventBlockingSync);

        m_context->setInputConsumedEvent(*InputConsumedEvent);

        CHECK_CUDA_ERROR(cudaStreamAddCallback(*InferCudaStream, TRTInferenceEngine::callback, this, 0));

        // CHECK_CUDA_ERROR(cudaMallocAsync(&raw_data, BATCH_SIZE * IMG_CHANNEL*IMG_WIDTH*IMG_HEIGHT* sizeof(float), *PreprocessCudaStream));
        CHECK_CUDA_ERROR(cudaHostRegister(raw_data, BATCH_SIZE * IMG_CHANNEL * IMG_WIDTH * IMG_HEIGHT * sizeof(float), cudaHostRegisterMapped));

        // Allocate GPU memory for input and output buffers
        m_outputLengthsFloat.clear();
        for (int i = 0; i < m_engine->getNbIOTensors(); ++i)
        {
            // uint32_t layerSize = 1;
            infer_tensor_details ip_tensor;
            infer_tensor_details out_tensor;

            const auto tensorName = m_engine->getIOTensorName(i);
            const auto tensorType = m_engine->getTensorIOMode(tensorName);
            const auto tensorShape = m_engine->getTensorShape(tensorName);

            if (tensorType == nvinfer1::TensorIOMode::kINPUT)
            {

                std::stringstream s;
                ip_tensor.name = tensorName;
                ip_tensor.isConsumed = InputConsumedEvent;

                for (int j = 1; j < tensorShape.nbDims; ++j)
                {
                    s << tensorShape.d[j] << "x";
                    ip_tensor.size *= tensorShape.d[j];
                }
                ip_tensor.buffer = raw_data;

                // No need to alloc ip_tensor buffer as mapping is done above!!
                // CHECK_CUDA_ERROR(cudaMallocAsync(&ip_tensor.buffer, trt_options.maxBatchSize * ip_tensor.size * sizeof(float), *PreprocessCudaStream));

                // Store the input dims for later use
                m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
                nvinfer1::Dims4 inputDims = {trt_options.optBatchSize, tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]};
                m_context->setBindingDimensions(0, inputDims);

                ip_tensors[tensorName] = ip_tensor;
                std::cout << "\tIP layer name:: " << tensorName << std::endl;
                std::cout << "\tIP layer Shape:: " << s.str() << std::endl;
            }
            else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::stringstream s;
                // The binding is an output
                out_tensor.name = tensorName;
                out_tensor.isConsumed = InputConsumedEvent;
                // m_outputDims.push_back(tensorShape);

                for (int j = 1; j < tensorShape.nbDims; ++j)
                {
                    s << tensorShape.d[j] << "x";
                    out_tensor.size *= tensorShape.d[j];
                }

                CHECK_CUDA_ERROR(cudaMallocAsync(&out_tensor.buffer, trt_options.optBatchSize * out_tensor.size * sizeof(float), *PreprocessCudaStream));

                std::cout << "\tOUT layer name:: " << tensorName << std::endl;
                std::cout << "\tOUT layer Shape:: " << s.str() << std::endl;

                out_tensors[tensorName] = out_tensor;
                if (needCpuOutput)
                {
                    tensor_details out_temp_tensor;

                    out_temp_tensor.name = tensorName;
                    out_temp_tensor.data = new float[trt_options.optBatchSize * out_tensor.size];
                    out_temp_tensor.element_count = out_tensors[tensorName].size * trt_options.optBatchSize;
                    out_temp_tensor.is_consumed = false;
                    out_temp_tensor.shape = {trt_options.optBatchSize, tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]};

                    CHECK_CUDA_ERROR(cudaHostRegister(out_temp_tensor.data, trt_options.optBatchSize * out_tensor.size * sizeof(float), cudaHostRegisterMapped));
                    // memset(out_cpu_tensors[tensorName], 0, trt_options.maxBatchSize*out_tensor.size);
                    out_cpu_tensors.push_back(out_temp_tensor);
                }
            }
            else
            {
                throw std::runtime_error("Error, IO Tensor is neither an input or output!");
            }
        }

        // Set the address of the input and output buffers
        for (auto &[name, ip_tensor] : ip_tensors)
        {
            bool status = m_context->setTensorAddress(name.c_str(), ip_tensor.buffer);
            if (!status)
            {
                return false;
            }
        }

        for (auto &[name, op_tensor] : out_tensors)
        {
            bool status = m_context->setTensorAddress(name.c_str(), op_tensor.buffer);
            if (!status)
            {
                return false;
            }
        }

        // Synchronize and destroy the cuda stream
        CHECK_CUDA_ERROR(cudaStreamSynchronize(*PreprocessCudaStream));

        return true;
    }

    // Preprocessing on cuda for speedup
    // void TRTInferenceEngine::preprocess(cv::Mat img, int i){

    //     float* out = this->raw_data + i * ip_tensors.begin()->second.size;

    //     // TODO: Fix the ConvertBGRtoRGB kernel for high res images
    //     // ConvertBGRtoRGB(out, img.data, IMG_WIDTH, IMG_WIDTH, img.cols, img.rows, img.step, 1/SCALE_FACTOR[0], *PreprocessCudaStream);

    // }

    // float* TRTInferenceEngine::preprocess_batch(const std::vector<cv::Mat>& batch){

    //     assert(batch.size()>0);

    //     std::shared_ptr<CudaEvent> waitEvent = InputConsumedEvent;
    //     CHECK_CUDA_ERROR(cudaStreamWaitEvent(*InferCudaStream, *waitEvent, 0));

    //     int input_size = IMG_CHANNEL*IMG_WIDTH*IMG_HEIGHT;

    //     int i = 0;
    //     // for (int i=0; i<batch.size(); i++){
    //     std::vector<std::future<void>> futures;
    //     for (auto& img: batch){
    //         preprocess(img, i);
    //         futures.emplace_back(std::async(std::launch::async, &TRTInferenceEngine::preprocess, this, img, i));
    //         i+=1;
    //     }
    //     for(auto &f:futures)
    //         f.wait();
    //     futures.clear();

    //     // cudaDeviceSynchronize();

    //     return (float *)this->raw_data;
    // }

    void TRTInferenceEngine::enqueue(float *src)
    {

        assert(CURR_BATCH_SIZE > 0);

        // auto start0 = std::chrono::high_resolution_clock::now();
        int64_t curr_pos = 0;
        for (auto &[name, ip_tensor] : ip_tensors)
        {

            std::shared_ptr<CudaEvent> waitEvent = ip_tensor.isConsumed;
            cudaPointerAttributes attr;
            CHECK_CUDA_ERROR(cudaPointerGetAttributes(&attr, (void *)src));

            // Tensor on CPU here!!
            if (attr.device == -2 && attr.hostPointer == NULL)
            {
                // This means the ptr is on Host or CPU
                // Wait for the GPU input buffer to get consumed to allocate new data
                // Takes much longer and is blocking (Need to check why)
                // Takes around 3-4 ms for batch of 5 streams of 3x640x480
                // std::cout << "Tensor on cpu!!" << std::endl;
                if (ip_tensor.isConsumed)
                {

                    // Removing this waitEvent. Not any impact on results seen yet.
                    // Need more testing
                    // CHECK_CUDA_ERROR(cudaStreamWaitEvent(*InferCudaStream, *waitEvent, 0));

                    // std::cout << "Transfering in the cpu" << std::endl;
                    // std::cout << "current batch size in tensorrt .cpp " << CURR_BATCH_SIZE << std::endl;
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(ip_tensor.buffer, src + curr_pos,
                                                     CURR_BATCH_SIZE * ip_tensor.size * sizeof(float),
                                                     cudaMemcpyHostToDevice, *PreprocessCudaStream));
                    // Print values for debugging purposes
                }
            }
            // Tensor on GPU here!!
            if (attr.device == 0 && attr.hostPointer != NULL)
            {
                // This means the ptr is on DEVICE or GPU
                // If used internal impl, no need for waiting as the data is already allocated to the memory
                // Adding wait and copy if src and ip_tensor.buffer are not same
                // Event if there is any copy is taking place, it is extremely fast (~10 microseconds for batch of 5 frames of 3x640x480 data)
                // std::cout << "Tensor on GPU!! and transfering in the cpu" << std::endl;
                // std::cout << "current batch size in tensorrt .cpp " << CURR_BATCH_SIZE << std::endl;
                // std::cout << CURR_BATCH_SIZE * ip_tensor.size << std::endl;
                if (ip_tensor.buffer != src + curr_pos)
                {
                    if (ip_tensor.isConsumed)
                    {
                        // CHECK_CUDA_ERROR(cudaStreamWaitEvent(*InferCudaStream, *waitEvent, 0));
                        CHECK_CUDA_ERROR(cudaMemcpyAsync(ip_tensor.buffer, src + curr_pos,
                                                         CURR_BATCH_SIZE * ip_tensor.size * sizeof(float),
                                                         cudaMemcpyDeviceToDevice, *PreprocessCudaStream));
                    }
                }
            }
            curr_pos += ip_tensor.size;
        }

        if (!m_context->allInputDimensionsSpecified())
        {
            throw std::runtime_error("Error, not all required dimensions specified.");
        }
        // std::cout << "After copy" << std::endl;
        // auto start = std::chrono::high_resolution_clock::now();
        execute_network_trt(*InferCudaStream);
        // auto stop4 = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop4 - start);
        // std::cout<<"execute in: "<<duration.count()<<std::endl;
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(start - start0);
        // std::cout<<"copy input in: "<<duration.count()<<std::endl;
        // _init+=1;
    }

    std::vector<tensor_details> TRTInferenceEngine::execute_network()
    {
        // std::cout << out_cpu_tensors.size() << std::endl;

        // for (auto &a : out_cpu_tensors)
        // {
        //     std::cout << a.name << " : " << a.element_count << std::endl;
        //     // std::cout << "out_cpu_tensor: " << a.get_element_count() << std::endl;
        // }

        return out_cpu_tensors;
    }
    std::vector<tensor_details> TRTInferenceEngine::execute_async_network()
    {

        // std::cout << out_cpu_tensors.size() << std::endl;

        // for (auto &a : out_cpu_tensors)
        // {
        //     std::cout << a.name << " : " << a.element_count << std::endl;
        //     // std::cout << "out_cpu_tensor: " << a.get_element_count() << std::endl;
        // }
        return out_cpu_tensors;
    }

    void TRTInferenceEngine::copyOutputToCPU(cudaStream_t stream)
    {

        // CHECK_CUDA_ERROR(cudaEventQuery(*InferCompleteEvent));
        // CHECK_CUDA_ERROR(cudaEventSynchronize(*InferCompleteEvent));
        // CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, *InferCompleteEvent, 0 ));

        if (needCpuOutput)
        {
            int index = 0;
            for (auto &[name, out_tensor] : out_tensors)
            {
                CHECK_CUDA_ERROR(cudaMemcpyAsync(out_cpu_tensors[index].data, out_tensor.buffer, trt_options.optBatchSize * out_tensor.size * sizeof(float), cudaMemcpyDeviceToHost, *PostprocessCudaStream));
                index++;
            }
        }
        // CHECK_CUDA_ERROR(cudaEventSynchronize(*InferCompleteEvent));
        // CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    void CUDART_CB TRTInferenceEngine::callback(cudaStream_t stream, cudaError_t status, void *user_data)
    {

        std::cout << "In callback " << std::endl;
        TRTInferenceEngine *engine = (TRTInferenceEngine *)user_data;
        engine->copyOutputToCPU(stream);
        std::cout << "In callback Done copying" << std::endl;
        // CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    void TRTInferenceEngine::execute_network_trt(CudaStream &InferStream)
    {

        assert(m_context);
        assert(CURR_BATCH_SIZE > 0);
        assert(InferStream.ptr());

        // m_context->setInputConsumedEvent(*InputConsumedEvent);
        // auto start = std::chrono::high_resolution_clock::now();
        bool status = m_context->enqueueV3(InferStream);
        // auto stop4 = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop4 - start);
        // std::cout<<"enqueue in: "<<duration.count()<<std::endl;

        if (!status)
        {
            std::cerr << "Could not enqueue batch for inference" << std::endl;
        }

        // Event record and wait on inferstream to avoid any lose of work on the kernels
        CHECK_CUDA_ERROR(cudaEventRecord(*InferCompleteEvent, InferStream));
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(InferStream, *InferCompleteEvent, 0));

        // auto stop5 = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop5 - stop4);
        // std::cout<<"record and wait in: "<<duration.count()<<std::endl;
        if (needCpuOutput)
        {
            int index = 0;

            for (auto &[name, out_tensor] : out_tensors)
            {
                // std::cout << "Before copy" << trt_options.optBatchSize * out_tensor.size << std::endl;
                // std::cout << name << " : " << out_tensor.size << std::endl;
                // std::cout << trt_options.optBatchSize << std::endl;

                CHECK_CUDA_ERROR(cudaMemcpyAsync(out_cpu_tensors[index].data, out_tensor.buffer, trt_options.optBatchSize * out_tensor.size * sizeof(float), cudaMemcpyDeviceToHost, *PostprocessCudaStream));
                index++;
            }
            // std::cout << "After copy in the infer function " << std::endl;
            // std::cout << out_cpu_tensors.size() << std::endl;
            // for (auto a : out_cpu_tensors)
            // {
            //     std::cout << a.name << " : " << a.element_count << std::endl;
            // }
        }
        // auto stop6 = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop6 - stop5);
        // std::cout<<"copy in: "<<duration.count()<<std::endl ;

        // Really heavy sync call!!
        // No effect seen on the output data even though no sync is used. Need more thorough testing
        // CHECK_CUDA_ERROR(cudaStreamSynchronize(InferStream));
        // auto stop7 = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop7 - stop6);
    }

}
