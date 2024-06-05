#include <iostream>
using namespace std;
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
// #include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
// #include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"
// #include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <stdio.h>
#include <iostream>

template <class T>
class ptr_wrapper
{
public:
    ptr_wrapper() : ptr(nullptr) {}
    ptr_wrapper(T *ptr) : ptr(ptr) {}
    ptr_wrapper(const ptr_wrapper &other) : ptr(other.ptr) {}
    T &operator*() const { return *ptr; }
    T *operator->() const { return ptr; }
    T *get() const { return ptr; }
    void destroy() { delete ptr; }
    T &operator[](std::size_t idx) const { return ptr[idx]; }

private:
    T *ptr;
    size_t size;
};

class Yolov3
{

private:
    int IMG_WIDTH;
    int IMG_HEIGHT;
    int BATCH_SIZE = 1;
    int IMG_CHANNEL;
    float *dst;
    std::vector<std::vector<float> > ANCHORS;
    std::vector<int64_t> NUM_ANCHORS;
    bool use_letterbox = false;

    Ort::Env env_;
    Ort::Session *session_;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string model_path_;
    char const *input_name_;
    char const *output_name1_;
    char const *output_name2_;
    std::vector<int64_t> inputShape;

    std::vector<std::vector<float> > inference_output;

    int number_of_classes;
    float confidence;
    float nms_threshold;
    std::string model;
public:
    Yolov3() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}
    Yolov3(int number_of_classes, std::vector<std::vector<float> > anchors,
           const std::string &model_path, int height, int width,
           int channels, int batch_size);

    void preprocess(py::array_t<uchar> image_arr, size_t batch_index);

    // size_t vectorProduct(const std::vector<int64_t> &vector);

    float sigmoid(float x) const;

    void detect(ptr_wrapper<float> input_tensor_ptr);

    std::tuple<std::vector<std::array<float, 4> >, std::vector<uint64_t>, std::vector<float> >
    postprocess(const ptr_wrapper<std::vector<std::vector<float> > > &infered,
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

    std::array<float, 4> post_process_box(const float &xt, const float &yt, const float &width,
                                          const float &height,
                                          const int64_t &input_image_height,
                                          const int64_t &input_image_width) const;

    std::vector<uint64_t> nms(const std::vector<std::array<float, 4> > &bboxes,
                              const std::vector<float> &scores,
                              const float overlapThresh = 0.45,
                              uint64_t topK = std::numeric_limits<uint64_t>::max());
    ~Yolov3()
    {
        delete session_;
    };
    cv::Mat numpyArrayToMat(py::array_t<uchar> arr);

    ptr_wrapper<float> get_raw_data(void) { return this->dst; }
    ptr_wrapper<std::vector<std::vector<float> > > get_inference_output(void) { return &this->inference_output; }
};