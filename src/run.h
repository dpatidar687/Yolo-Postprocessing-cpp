
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <yolov3.h>
#include <yolov7.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class YoloDetectorv3
{
public:

    YoloDetectorv3() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}

    

    ~YoloDetectorv3()
    {
        delete session_;
    }

    // YoloDetectorv3();
    // ~YoloDetectorv3();

    void initialize(const std::string &model_path, int height, int width, int channels);

    std::vector<float>  preprocess(std::string img_path, int height, int width , int channel , size_t batch_index);

    // std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> detect(
    //     std::vector<float> input_tensor, int input_image_height, int input_image_width);

    std::vector<std::vector<float>> detect(std::vector<float> input_tensor);
    size_t vectorProduct(const std::vector<int64_t> &vector);

private:
    Ort::Env env_;
    Ort::Session *session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string model_path_;
    char const *input_name_;
    char const *output_name1_;
    char const *output_name2_;
    int height;
    int width;
    int channels;
    int number_of_classes;
    int batch_size;
    float confidence;
    float nms_threshold;
    std::vector<std::vector<float>> anchors;
    std::string model;


};

class YoloDetectorv7
{
public:
    YoloDetectorv7() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}

    
    ~YoloDetectorv7()
    {
        delete session_;
    }

    void initialize(const std::string &model_path, int height, int width, int channels);

    std::vector<std::vector<float>> detect(
        std::vector<float> input_tensor);

    size_t vectorProduct(const std::vector<int64_t> &vector);

private:
    Ort::Env env_;
    Ort::Session *session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string model_path_;
    char const *input_name_;
    char const *output_name1_;
    char const *output_name2_;
    char const *output_name3_;
    int height;
    int width;
    int channels;
    int number_of_classes;
    int batch_size;
    float confidence;
    float nms_threshold;
    std::string model;
    std::vector<std::vector<float>> anchors;
};

