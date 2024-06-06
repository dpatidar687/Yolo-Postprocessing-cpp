#include <iostream>
using namespace std;
#include <iostream>
#include <vector>


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <core/providers/cpu/cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <stdio.h>
#include <iostream>


class Yolov7
{

    int IMG_WIDTH;
    int IMG_HEIGHT;
    int BATCH_SIZE = 1;
    int IMG_CHANNEL;
    float *dst;
    std::vector<std::vector<float>> ANCHORS;
    std::vector<int64_t> NUM_ANCHORS;
    bool use_letterbox = false; // whether to use letterbox while preprocessing and postproceesing

    Ort::Env env_;
    Ort::Session *session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string model_path_;
    char const *input_name_;
    char const *output_name1_;
    char const *output_name2_;
    char const *output_name3_;

    int number_of_classes;
    float confidence;
    float nms_threshold;
    std::string model;

public:
    Yolov7() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}
    Yolov7(int batch_size, int image_size, std::vector<std::vector<float>> anchors);

    float sigmoid(float x) const;

    std::vector<float> preprocess(py::array_t<uchar> image_arr, size_t batch_index);

    void initialize(const std::string &model_path, int height, int width,
                    int channels, int batch_size);

    size_t vectorProduct(const std::vector<int64_t> &vector);

    std::vector<std::vector<float>> detect(std::vector<float> input_tensor);

    std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> postprocess(const std::vector<std::vector<float>> &inferenceOutput,
                                                                                                         const float confidenceThresh, const float nms_threshold, const uint16_t num_classes,
                                                                                                         const int64_t input_image_height, const int64_t input_image_width,
                                                                                                         const int64_t batch_ind);

    void post_process_feature_map(const float *out_feature_map, const float confidenceThresh,
                                  const int num_classes, const int64_t input_image_height,
                                  const int64_t input_image_width, const int factor,
                                  const std::vector<float> &anchors, const int64_t &num_anchors,
                                  std::vector<std::array<float, 4>> &bboxes,
                                  std::vector<float> &scores,
                                  std::vector<uint64_t> &classIndices, const int b);

    std::array<float, 4> post_process_box(const float &xt, const float &yt, const float &width,
                                          const float &height,
                                          const int64_t &input_image_height,
                                          const int64_t &input_image_width) const;

    std::vector<uint64_t> nms(const std::vector<std::array<float, 4>> &bboxes,
                              const std::vector<float> &scores,
                              const float overlapThresh = 0.45,
                              uint64_t topK = std::numeric_limits<uint64_t>::max());

    ~Yolov7()
    {
        delete session_;
    };
    cv::Mat numpyArrayToMat(py::array_t<uchar> arr) ;
};  