#ifndef _OPENVINO_H
#define _OPENVINO_H

#ifdef ENABLE_VINO 
#include "../../models/InferenceEngine.h"
#include "openvino/openvino.hpp"
#include <malloc.h>

typedef struct ov_tensor_details{

    std::string name;
    ov::Shape shape;
    int64_t element_count = 1;

} ov_tensor_details;

namespace mtx{

    class hey{

        public:
            std::string name="deepak";

    };

    class OpenVINOInferenceEngine : public InferenceEngine{
	    std::string xml_path;
	    std::shared_ptr<ov::Model> model;
	    ov::CompiledModel ov_model;
        ov::Core core;
	    std::vector<ov::Tensor> input_tensor;
	    ov::Tensor output_tensor;
	    std::unordered_map<std::string, ov_tensor_details> ov_input_tensors;
	    std::unordered_map<std::string, ov_tensor_details> ov_output_tensors;
        void getModelDetails(ov::CompiledModel &network);
	    void showAvailableDevices();
	    std::vector<ov::InferRequest> async_infer_pool;
        std::vector<ov::InferRequest> infer_pool;
	    inline void convert_input_data_to_ov_tensor(float *src, int index=0);
	    std::unordered_map<std::string, tensor_details> tensor_raw_data;
        std::vector<bool> _in_queue;
        std::vector<bool> _async_in_queue ;
        public:
            OpenVINOInferenceEngine(const std::string &model_path, const std::string &provider,
                                    const std::vector<std::vector<int64_t>> &inputShapes,
                                    const std::vector<std::vector<long int>> &outputShape, pre_processing preprocess = pre_processing());
	        virtual ~OpenVINOInferenceEngine();
            // ~OpenVINOInferenceEngine() = default;
            //void preprocess_batch();
            //void postprocess();
            //void postprocess_batch();
            virtual void enqueue(float *);
            std::vector<tensor_details> execute_network() override;
            std::vector<tensor_details> execute_async_network() override;
            std::unordered_map<std::string, std::vector<int64_t>> get_output_shape() override;
            void change_batch_size(int64_t batch_size);
    };

}

#endif
#endif
