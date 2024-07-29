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

class Base_classifier {
public:
    Base_classifier(const std::string &model_path, int batch_size, std::string provider);
    ~Base_classifier();

   
    py::list infer_cpp(py::array &input_array);

private:
    private:
    int IMG_WIDTH;
    int IMG_HEIGHT;
    int BATCH_SIZE = 1;
    int IMG_CHANNEL;
    float *dst;

    Ort::Env env_;
    Ort::Session *session_;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::RunOptions runOptions;

    std::string model_path_;
    std::string provider = "cpu";

    size_t input_count;
    size_t output_count;
    size_t inputTensorSize;
    std::vector<int64_t> inputShape;
    std::vector<std::string> output_names;
    std::vector<std::string> input_names;
    
    const char *const *names_of_outputs_cstr;
    std::vector<const char *> names_of_outputs_ptr;

    const char *const *names_of_inputs_cstr;
    std::vector<const char *> names_of_inputs_ptr;
};

