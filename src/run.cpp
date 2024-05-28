#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <yolov3.cpp>
#include <array>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Include this header for automatic conversion of STL containers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

using namespace std;
using namespace cv; 
class YoloDetector {
public:
    YoloDetector() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}

    void initialize(const std::string& model_path) {
        model_path_ = model_path;
        session_ = new Ort::Session(env_, model_path_.c_str(), Ort::SessionOptions());
        input_name_ = session_->GetInputName(0, allocator_);
        output_name1_ = session_->GetOutputName(0, allocator_);
        output_name2_ = session_->GetOutputName(1, allocator_);
        std::cout << "Created the session " << std::endl;
    }

    ~YoloDetector() {
        delete session_;
    }

    std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> detect(std::string image_path);
    size_t vectorProduct(const std::vector<int64_t> &vector);


    // Other member functions...

private:
    Ort::Env env_;
    Ort::Session* session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string model_path_;
    char const* input_name_;
    char const* output_name1_;
    char const* output_name2_;
};



size_t YoloDetector::vectorProduct(const std::vector<int64_t> &vector)
{
  if (vector.empty())
    return 0;

size_t product = 1;
for (const auto &element : vector)
    product *= element;

return product;
}

std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> YoloDetector::detect(std::string image_path){

    cv::Mat image = cv::imread(image_path);
    const int batch_size = 1;
    const int batch_index = 0;
    const int channels = 3;
    const int height =416;
    const int width = 416;

    float nms_threshold = 0.45;
    int number_of_classes = 1;
    float confidence = 0.6;
    std ::string folder_path = "./";
    std ::string save_path = folder_path+"/output/";

    int input_image_width = image.cols;
    int input_image_height = image.rows;

    Yolov3 y(number_of_classes, width);
    float *blob = new float[channels * height * width];
    std::vector<int64_t> inputTensorShape{1, channels, height, width};

    y.preprocessing(blob, image, batch_index);

    size_t inputTensorSize = vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    auto inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape1 = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape2 = session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();


    std ::vector<float> inputValues = inputTensorValues;
    inputShape[0] = 1;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto inputOnnxTensor = Ort::Value::CreateTensor<float>(memoryInfo,
     inputValues.data(), inputValues.size(),
     inputShape.data(), inputShape.size());

    std ::vector<std::string> inputNames = {input_name_};
    std ::vector<std::string> outputNames1 = {output_name1_};
    std ::vector<std::string> outputNames2 = {output_name2_};

    static const char *output_names[] = {output_name1_, output_name2_};
    // cout << sizeof(output_names) << "      " << sizeof(output_names[0]) << endl;
    static const size_t NUM_OUTPUTS = sizeof(output_names) / sizeof(output_names[0]);

    OrtValue *p_output_tensors[NUM_OUTPUTS] = {nullptr};

    char const *const *ii = &input_name_;
    char const *const *oo = output_names;
    std ::vector<float> outputTensor;

    auto outputValues1 = session_->Run(
        Ort::RunOptions{nullptr},
        ii,
        &inputOnnxTensor, inputNames.size(), oo, 2);


    auto *rawOutput1 = outputValues1[0].GetTensorMutableData<float>();
    std::vector<int64_t> out1 = outputValues1[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count1 = outputValues1[0].GetTensorTypeAndShapeInfo().GetElementCount();

    auto *rawOutput2 = outputValues1[1].GetTensorData<float>();
    std::vector<int64_t> out2 = outputValues1[1].GetTensorTypeAndShapeInfo().GetShape();
    size_t count2 = outputValues1[1].GetTensorTypeAndShapeInfo().GetElementCount();

    int arrSize1 = count1 * sizeof(rawOutput1[0]);
    std::vector<float> vec1(rawOutput1, rawOutput1 + count1);
    int arrSize2 = sizeof(rawOutput2) / sizeof(rawOutput2[0]);
    std::vector<float> vec2(rawOutput2, rawOutput2 + count2);

    std::vector<std::vector<float>> vectorOfVectors;
    vectorOfVectors.push_back(vec1);
    vectorOfVectors.push_back(vec2);

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> class_indices;

    auto processed_result = y.postprocess(vectorOfVectors, confidence, number_of_classes, input_image_height, input_image_width, batch_index);
    std::tie(bboxes, scores, class_indices) = processed_result;

    // std::cout << "size of boxes " << bboxes.size() << std::endl;


    std::vector<uint64_t> after_nms_indices;

    after_nms_indices = y.nms(bboxes, scores, nms_threshold);

    std::vector<std::array<float, 4>> after_nms_bboxes;
    std::vector<uint64_t> after_nms_class_indices;
    std::vector<float> after_nms_scores;

    after_nms_bboxes.reserve(after_nms_indices.size());
    after_nms_class_indices.reserve(after_nms_indices.size());
    after_nms_scores.reserve(after_nms_indices.size());

    for (const auto idx : after_nms_indices)
    {
      after_nms_bboxes.emplace_back(bboxes[idx]);
      after_nms_class_indices.emplace_back(class_indices[idx]);
      after_nms_scores.emplace_back(scores[idx]);
  }
    // cout<< "-----------------------------------------------------------------------------" << endl;
    // cout << after_nms_bboxes.size() << "           " << after_nms_scores.size() << "               " << after_nms_class_indices.size() << endl;
    // cout << save_path + filename << endl;
    // cout << imagePath << endl;
  for (int i = 0; i < after_nms_bboxes.size(); ++i)
  {
      const std::array<float, 4> &bbox = after_nms_bboxes[i];

      float x1 = bbox[0];
      float y1 = bbox[1];
      float x2 = bbox[2];
      float y2 = bbox[3];
      // std::cout << "Bounding Box " << i << ": (" << int(x1) << ", " << int(y1) << ", " << int(x2) << ", " << int(y2) << " ,"
      //           << after_nms_scores[i] << " ," << after_nms_class_indices[i] << ")" << std::endl;
  }
    // y.show_boxes(image, after_nms_bboxes, after_nms_class_indices, after_nms_scores, save_path + filename);
    // y.draw_and_save(image, after_nms_bboxes, after_nms_class_indices, after_nms_scores, "./1_re.jpg");

  image.release();
    // std::cout << "adsd" << std::endl;
  delete[] blob;
    // delete[] rawOutput1;
    // delete[] rawOutput2;
    // delete inputOnnxTensor;

  return std::make_tuple(after_nms_bboxes,after_nms_class_indices,after_nms_scores);
}


PYBIND11_MODULE(run_yolo_onnx, m) {
    py::class_<YoloDetector>(m, "YoloDetector")
    .def(py::init<>())
    .def("initialize", &YoloDetector::initialize)
    .def("detect", &YoloDetector::detect);
}


// int main() {
//     std::string model_path = "/home/manish/Documents/older/alexandria/face_detection.tiny_yolov3/v3/onnx/yolo_tiny_25_07.onnx";
//     YoloDetector detector;
//     detector.initialize(model_path);
//     std::vector<std::array<float, 4>> after_nms_bboxes;
//     std::vector<uint64_t> after_nms_class_indices;
//     std::vector<float> after_nms_scores;

//     std::tie(after_nms_bboxes, after_nms_class_indices, after_nms_scores) = detector.detect("/home/manish/tt/2.jpg");


//     // Now the detector is initialized with the specified ONNX model
//     // You can call other member functions of YoloDetector as needed
//     return 0;
// }