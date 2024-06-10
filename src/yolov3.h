#include <iostream>
using namespace std;
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <stdio.h>
#include <iostream>

#include <core/providers/cuda/cuda_provider_options.h>
#include <core/providers/cpu/cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/tensorrt/tensorrt_provider_factory.h>




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

    Ort::Env env_;
    Ort::Session *session_;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


    std::string model_path_;
    std::string provider = "cpu";

    std::string output_name1;
    std::string output_name2;
    std::string input_name ;
    size_t input_count;
    size_t output_count;
    size_t inputTensorSize;
    std::vector<int64_t> inputShape;
    // const char* names_of_input;
    // const char* names_of_output;

    

    int number_of_classes;
    float confidence;
    float nms_threshold;
    std::string model;
public:
    std::vector<std::vector<float>> inference_output;
    // Yolov3() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}
    Yolov3(int number_of_classes, std::vector<std::vector<float> > anchors,const std::string &model_path, int batch_size, std::string provider);

    // void preprocess(py::array_t<uchar> image_arr, size_t batch_index);


    float* preprocess_batch( py::list &batch) ;

    inline void preprocess(const unsigned char *src, const int64_t b)  ;

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

 std::vector<std::tuple<std::vector<std::array<float, 4> >, std::vector<uint64_t>, std::vector<float>>>
    postprocess_batch(const ptr_wrapper<std::vector<std::vector<float>>> &infered,
        const float confidenceThresh, const float nms_threshold, const uint16_t num_classes,
        const int64_t input_image_height, const int64_t input_image_width);
               
    float* get_raw_img()  {
        return dst;
    }

    size_t get_size_img()  {
        return IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL * BATCH_SIZE;
    }

    std::vector<std::vector<float>> get_raw_inference_output() {
        return inference_output;
    }
    size_t get_size_inference_output() {
        return inference_output.size();
    }



    ~Yolov3()
    {
        delete session_;
        delete[] this->dst;
    };
    cv::Mat numpyArrayToMat(py::array_t<uchar> arr);
    ptr_wrapper<float> get_img_ptr(void) { return this->dst; }
    ptr_wrapper<std::vector<std::vector<float>>> get_inference_output_ptr(void) { return &this->inference_output; }
   
};