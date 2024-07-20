using namespace std;
#include <vector>
#include <chrono>
#include <ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <random>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <stdio.h>
#include <iostream>
// #include <../models/model_config.h>

#include <core/providers/cuda/cuda_provider_options.h>
#include <core/providers/cpu/cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/tensorrt/tensorrt_provider_factory.h>
#include <onnxruntime_c_api.h>

class Yolov7
{
    int IMG_WIDTH;
    int IMG_HEIGHT; 
    int BATCH_SIZE = 1;
    int IMG_CHANNEL;
    float *dst;
    bool use_letterbox = true;
    std::vector<float> letter_box_color = {0, 0, 0};

    std::vector<std::vector<float> > ANCHORS;
    std::vector<int64_t> NUM_ANCHORS;
    int number_of_classes;

    Ort::Env env_;
    Ort::Session *session_;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::RunOptions runOptions;

    std::string model_path_;
    std::string provider = "cpu";
    std::string input_name;
    size_t input_count;
    size_t output_count;
    size_t inputTensorSize;
    std::vector<int64_t> inputShape;
    std::vector<std::vector<int64_t>> outputShape;

    std::vector<std::string> output_names;
    std::vector<std::string> input_names;
    float confidence;
    float nms_threshold;
    std::string model;


    const char *const *names_of_outputs_cstr;
    std::vector<const char *> names_of_outputs_ptr;

    const char *const *names_of_inputs_cstr;
    std::vector<const char *> names_of_inputs_ptr;

public:
    std::vector<std::vector<float> > inference_output;
    Yolov7(int number_of_classes, std::vector<std::vector<float> > anchors, const std::string &model_path,
           int batch_size, std::string provider, bool letter_box , std::vector<float> letter_box_color);

    float sigmoid(float x) const;

    py::array preprocess_batch(py::list &batch);

    inline void preprocess(const unsigned char *src, const int64_t b);
    cv::Mat create_letterbox(const cv::Mat &frame) const;
    cv::Mat numpyArrayToMat(py::array_t<uchar> arr);

    py::list detect(py::array &input_tensor_array);

    py::tuple postprocess(py::list &infered,
                          const float confidenceThresh, const float nms_threshold, const uint16_t num_classes,
                          const int64_t input_image_height, const int64_t input_image_width,
                          const int64_t batch_ind);

    void post_process_feature_map(const float *out_feature_map, const float confidenceThresh,
                                  const int num_classes, const int64_t input_image_height,
                                  const int64_t input_image_width, const int factor,
                                  const std::vector<float> &anchors, const int64_t &num_anchors,
                                  std::vector<std::array<float, 4> > &bboxes,
                                  std::vector<float> &scores,
                                  std::vector<uint64_t> &classIndices, const int b);

    
    void post_process_new(const float *out_feature_map, const float confidenceThresh,
                              const int num_classes, const int64_t input_image_height,
                              const int64_t input_image_width,
                              std::vector<std::array<float, 4> > &bboxes,
                              std::vector<float> &scores,
                              std::vector<uint64_t> &classIndices, const int b, const int64_t num_boxes) ;

    std::array<float, 4> post_process_box(const float &xt, const float &yt, const float &width,
                                          const float &height,
                                          const int64_t &input_image_height,
                                          const int64_t &input_image_width) const;

    std::vector<uint64_t> nms(const std::vector<std::array<float, 4> > &bboxes,
                              const std::vector<float> &scores,
                              const float overlapThresh = 0.45,
                              uint64_t topK = std::numeric_limits<uint64_t>::max());

    py::list postprocess_batch(py::list &infered,
                               const float confidenceThresh, const float nms_threshold,
                               const int64_t input_image_height, const int64_t input_image_width);

    ~Yolov7()
    {
        delete session_;
        delete[] this->dst;
    };
};

// class yolobase{

//     public:
//         yolobase() = default;
//         yolobase(const mtx::ModelConfig &config){
//              std::cout << "\t Model Path: " << config.get_confidence_threshold() << std::endl;
//              std::cout << "done with the exposeing of yolobase"<< std::endl;
//         }
//         ~yolobase() = default;
        
// };