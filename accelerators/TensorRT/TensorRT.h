#ifndef _TRT_H
#define _TRT_H

#include <fstream>
#include <unordered_map>
#include "NvInfer.h"
#ifdef ENABLE_TRT
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"
#endif

#include <iostream>
#include <map>
#include <memory>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../models/InferenceEngine.h"
// #include "preprocessor.h"
// #include "core/queue.h"


#define DISABLE_CLASS_COPY(NoCopyClass)       \
    NoCopyClass(const NoCopyClass&) = delete; \
    void operator=(const NoCopyClass&) = delete

#define SIMPLE_MOVE_COPY(Cls)    \
    Cls& operator=(Cls&& o) {    \
        move_copy(std::move(o)); \
        return *this;            \
    }                            \
    Cls(Cls&& o) { move_copy(std::move(o)); }


inline bool doesFileExist(const std::string& filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

#define CHECK_CUDA_ERROR(val) checkCudaErrorCode((val), #val, __FILE__, __LINE__)
inline void checkCudaErrorCode(cudaError_t code, const char* const func, const char* const file, const int line) {

    if (code != cudaSuccess) {

        std::string errMsg = "CUDA Runtime Error at: " + (std::string)file + " line : " + std::to_string(line) + " Error code: "+ std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);

        std::cerr<< "CUDA Runtime Error at: " << file<< " : "<< line << std::endl;
        std::cerr<< "Error Code: " << std::to_string(code) <<  std::endl;
        std::cerr<< "Error Message: " << cudaGetErrorString(code) <<  std::endl;

        throw std::runtime_error(errMsg);
    }
}


/**
 * Helper class for managing Cuda Streams.
 * DS Implemetation
 */
class CudaStream
{
public:
        explicit CudaStream(uint flag = cudaStreamDefault, int priority = 0){
            CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&m_Stream, flag, priority));
        };
        ~CudaStream(){
            if (m_Stream != nullptr)
                CHECK_CUDA_ERROR(cudaStreamDestroy(m_Stream));
        };
    operator cudaStream_t() { return m_Stream; }
    cudaStream_t& ptr() { return m_Stream; }
    SIMPLE_MOVE_COPY(CudaStream)

private:
    void move_copy(CudaStream&& o)
    {
        m_Stream = o.m_Stream;
        o.m_Stream = nullptr;
    }
    DISABLE_CLASS_COPY(CudaStream);

    cudaStream_t m_Stream = nullptr;
};

/**
 * Helper class for managing Cuda events.
 * DS Implemetation
 */
class CudaEvent
{
public:
        explicit CudaEvent(uint flag = cudaEventDefault){
            CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&m_Event, flag));
        };
        ~CudaEvent(){
            CHECK_CUDA_ERROR(cudaEventDestroy(m_Event));
        };
    operator cudaEvent_t() { return m_Event; }
    cudaEvent_t& ptr() { return m_Event; }
    SIMPLE_MOVE_COPY(CudaEvent)

private:
    void move_copy(CudaEvent&& o)
    {
        m_Event = o.m_Event;
        o.m_Event = nullptr;
    }
    DISABLE_CLASS_COPY(CudaEvent);

    cudaEvent_t m_Event = nullptr;
};

typedef struct {

    std::string name;
    int64_t size = 1;
    void* buffer;
    std::shared_ptr<CudaEvent> isConsumed;

} infer_tensor_details;

// Precision used for GPU inference
enum class Precision {
    // Full precision floating point value
    FP32,
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

struct Options {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8 inference.
    // Should be set to as large a batch number as your GPU will support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }

    }
};


namespace mtx{

    class TRTInferenceEngine : public InferenceEngine{

	    std::string trt_engine_path;
            std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
            std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
            std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

            bool isSecondary;
            bool needCpuOutput;

            std::vector<std::string> deviceNames;

            // Holds pointers to the input and output GPU buffers
            std::map<std::string, infer_tensor_details> ip_tensors;
            std::map<std::string, infer_tensor_details> out_tensors;
            std::vector<tensor_details> out_cpu_tensors;

            std::vector<std::string> m_IOTensorNames;
            std::vector<uint32_t> m_outputLengthsFloat{};
            std::vector<nvinfer1::Dims3> m_inputDims;
            // std::vector<nvinfer1::Dims> m_outputDims;


            // Status codes: 0 - Failed, 1 - Created new engine, 2- engine already present
            int buildTRT(std::string onnxModelPath, bool overwrite=false);
            static void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void* );
            void showAvailableDevices();
            int loadEngine();
            bool allocateResources();
            void copyOutputToCPU(cudaStream_t);
            void execute_network_trt(CudaStream&);

            // Create the cuda stream that will be used for preprocess and inference
            // std::thread process_thread;
            // std::thread infer_thread;

            // Queues for processing multiple batches together in parallel
            // mtx::Queue<cv::Mat> input_queue;
            // mtx::Queue<cv::Mat> process_queue;

            std::unique_ptr<CudaStream> PreprocessCudaStream;
            std::unique_ptr<CudaStream> InferCudaStream;
            std::unique_ptr<CudaStream> PostprocessCudaStream;

            std::shared_ptr<CudaEvent> InputConsumedEvent;
            std::shared_ptr<CudaEvent> InferCompleteEvent;

            Options trt_options;
            TRTLogger m_logger;

        public:
            TRTInferenceEngine(const std::string &model_path, const std::string provider, const std::vector<std::vector<long int>> &inputShapes, const std::vector<std::vector<long int>>&outputShapes, pre_processing preprocess, bool isSecondary);
	    virtual ~TRTInferenceEngine();

            // void preprocess_batch_cuda(const std::vector<cv::Mat>& img_batch);
            // void postprocess();
            // void postprocess_batch();

            std::unordered_map<std::string, std::vector<int64_t>> get_output_shape()  {
                std::unordered_map<std::string, std::vector<int64_t>> temp_shape;
                for (auto tensor : out_cpu_tensors){
                    temp_shape[tensor.name] =  tensor.shape;
                }
                return temp_shape;
            };

            void enqueue(float* src);
            // void preprocess(cv::Mat img,int i);
            std::vector<tensor_details> execute_async_network();
            std::vector<tensor_details> execute_network();
            // float* preprocess_batch(const std::vector<cv::Mat>& batch);
    };

}

// #endif
#endif
