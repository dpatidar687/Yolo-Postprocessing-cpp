#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <yolov3.cpp>
#include <yolov7.cpp>

#include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Include this header for automatic conversion of STL containers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

using namespace std;
using namespace cv; 
class YoloDetectorv3 {
public:
    YoloDetectorv3() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}


void initialize(const std::string& model_path, int height, int width, int channels, int number_of_classes, int batch_size, float confidence, float nms_threshold, 
    std::vector<std::vector<float>> anchors) {
    this->model_path_ = model_path;
    this->height = height;
    this->width = width;
    this->channels = channels;
    this->number_of_classes = number_of_classes;
    this->batch_size = batch_size;
    this->confidence = confidence;
    this->nms_threshold = nms_threshold;
    this->anchors = anchors;
    
    std::cout << model_path << std::endl;
    session_ = new Ort::Session(env_, this->model_path_.c_str(), Ort::SessionOptions());
    input_name_ = session_->GetInputName(0, allocator_);
    output_name1_ = session_->GetOutputName(0, allocator_);
    output_name2_ = session_->GetOutputName(1, allocator_);
    std::cout << "Created the session " << std::endl;
    }

    ~YoloDetectorv3() {
        delete session_;
    }

    std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> detect( 
     std::vector<float>  input_tensor,
     int input_image_height,  int input_image_width);

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
    int height;
    int width;
    int channels;
    int number_of_classes;
    int batch_size;
    float confidence;
    float nms_threshold;
    std::vector<std::vector<float>> anchors;
};



size_t YoloDetectorv3::vectorProduct(const std::vector<int64_t> &vector)
{
  if (vector.empty())
    return 0;

size_t product = 1;
for (const auto &element : vector)
    product *= element;

return product;
}

std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> YoloDetectorv3::detect(   
   std::vector<float>  input_tensor, int input_image_height, int input_image_width){

    // cv::Mat image = cv::imread(image_path);
    const int batch_index = 0;
    // cout << "calling the detect fucntion "<< endl;

    Yolov3 y(number_of_classes, width, anchors);
    float *blob = new float[channels * height * width];

    
    std::vector<int64_t> inputTensorShape{1, channels, height, width};
    std::copy(input_tensor.begin(), input_tensor.end(), blob);

    size_t inputTensorSize = vectorProduct(inputTensorShape);
    cout<< inputTensorSize << endl;
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
        
    // std::cout<< "size of input tensor " << inputTensorValues.size() << endl;

    auto inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape1 = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape2 = session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();


    std ::vector<float> inputValues = inputTensorValues;
    inputShape[0] = 1;

    cout << inputValues.size() << endl;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto inputOnnxTensor = Ort::Value::CreateTensor<float>(memoryInfo,
     inputValues.data(), inputValues.size(),
     inputShape.data(), inputShape.size());

    std ::vector<std::string> inputNames = {input_name_};
    std ::vector<std::string> outputNames1 = {output_name1_};
    std ::vector<std::string> outputNames2 = {output_name2_};

    static const char *output_names[] = {output_name1_, output_name2_};

    cout << sizeof(output_names) << "      " << sizeof(output_names[0]) << endl;

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

    std::cout << vec1.size() << "    " << vec2.size() << endl;

    auto processed_result = y.postprocess(vectorOfVectors, confidence, number_of_classes, input_image_height, input_image_width, batch_index);
    std::tie(bboxes, scores, class_indices) = processed_result;

    std::cout << "size of boxes " << bboxes.size() << std::endl;


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
  
  delete[] blob;

  return std::make_tuple(after_nms_bboxes,after_nms_class_indices,after_nms_scores);
}




// ==========================================================================================================
class YoloDetectorv7 {
public:
    YoloDetectorv7() : env_(ORT_LOGGING_LEVEL_WARNING, "test"), allocator_(Ort::AllocatorWithDefaultOptions()), session_(nullptr) {}

    
void initialize(const std::string& model_path, int height, int width, int channels, int number_of_classes, int batch_size, 
    float confidence, float nms_threshold, std::vector<std::vector<float>> anchors) {
    this->model_path_ = model_path;
    this->height = height;
    this->width = width;
    this->channels = channels;
    this->number_of_classes = number_of_classes;
    this->batch_size = batch_size;
    this->confidence = confidence;
    this->nms_threshold = nms_threshold;
    this->anchors = anchors;

    std::cout << model_path << std::endl;


    session_ = new Ort::Session(env_, this->model_path_.c_str(), Ort::SessionOptions());
    input_name_ = session_->GetInputName(0, allocator_);
    std::cout<< input_name_ << std::endl;

   
    output_name1_ = session_->GetOutputName(0, allocator_);
    output_name2_ = session_->GetOutputName(1, allocator_);
    output_name3_ = session_->GetOutputName(2, allocator_);
    std::cout<< output_name1_ << std::endl;
    std::cout<< output_name2_ << std::endl;
    std::cout<< output_name3_ << std::endl;


    auto outputShape1 = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output Shape: ";
    for (const auto& dim : outputShape1) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    auto outputShape2 = session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output Shape: ";
    for (const auto& dim : outputShape2) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    auto outputShape3 = session_->GetOutputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output Shape: ";
    for (const auto& dim : outputShape3) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;


    std::cout << "Created the session " << std::endl;
    }

    ~YoloDetectorv7() {
        delete session_;
    }

    std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> detect(   
   std::vector<float>  input_tensor,int input_image_height, int input_image_width);
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
    char const* output_name3_;
    int height;
    int width;
    int channels;
    int number_of_classes;
    int batch_size;
    float confidence;
    float nms_threshold;
    std::vector<std::vector<float>> anchors;
};



size_t YoloDetectorv7::vectorProduct(const std::vector<int64_t> &vector)
{
  if (vector.empty())
    return 0;

size_t product = 1;
for (const auto &element : vector)
    product *= element;

return product;
}

std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> YoloDetectorv7::detect(   
   std::vector<float>  input_tensor,int input_image_height, int input_image_width){
    
    int batch_index = 0;
    

    Yolov7 v7object(number_of_classes, width, anchors);
    float *blob = new float[channels * height * width];
    std::vector<int64_t> inputTensorShape{1, channels, height, width};

   
    std::copy(input_tensor.begin(), input_tensor.end(), blob);

    size_t inputTensorSize = vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    auto inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape1 = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape2 = session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape3 = session_->GetOutputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();


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
    std ::vector<std::string> outputNames3 = {output_name3_};

    
    static const char *output_names[] = {output_name1_, output_name2_, output_name3_};
    std::cout << sizeof(output_names) << "      " << sizeof(output_names[0]) << sizeof(output_names[1]) << sizeof(output_names[2]) << std::endl;


    static const size_t NUM_OUTPUTS = sizeof(output_names) / sizeof(output_names[0]);

    std::cout << "Number of outputs: " << NUM_OUTPUTS << std::endl;

    std::cout<<"Input shape: "<<inputShape[0]<<" "<<inputShape[1]<<" "<<inputShape[2]<<" "<<inputShape[3]<<endl;


    OrtValue *p_output_tensors[NUM_OUTPUTS] = {nullptr};

    char const *const *ii = &input_name_;
    char const *const *oo = output_names;
    std ::vector<float> outputTensor;

    auto outputValues1 = session_->Run(
        Ort::RunOptions{nullptr},
        ii,
        &inputOnnxTensor, inputNames.size(), oo, 3);

    auto numInputs = session_->GetInputCount();
    auto numOutputs = session_->GetOutputCount();
    cout<< numInputs << " " << numOutputs << endl;

    

    auto *rawOutput1 = outputValues1[0].GetTensorMutableData<float>();
    std::vector<int64_t> out1 = outputValues1[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count1 = outputValues1[0].GetTensorTypeAndShapeInfo().GetElementCount();

    auto *rawOutput2 = outputValues1[1].GetTensorData<float>();
    std::vector<int64_t> out2 = outputValues1[1].GetTensorTypeAndShapeInfo().GetShape();
    size_t count2 = outputValues1[1].GetTensorTypeAndShapeInfo().GetElementCount();

    auto *rawOutput3 = outputValues1[2].GetTensorData<float>();
    std::vector<int64_t> out3 = outputValues1[2].GetTensorTypeAndShapeInfo().GetShape();
    size_t count3 = outputValues1[2].GetTensorTypeAndShapeInfo().GetElementCount();


    
    int arrSize1 = count1 * sizeof(rawOutput1[0]);
    std::vector<float> vec1(rawOutput1, rawOutput1 + count1);

    int arrSize2 = sizeof(rawOutput2) / sizeof(rawOutput2[0]);
    std::vector<float> vec2(rawOutput2, rawOutput2 + count2);

    int arrSize3 = sizeof(rawOutput3) / sizeof(rawOutput3[0]);
    std::vector<float> vec3(rawOutput3, rawOutput3 + count3);

    // std::cout << arrSize1 << " " << arrSize2 << " " << arrSize3 << std::endl;

    std::vector<std::vector<float>> vectorOfVectors;
    vectorOfVectors.push_back(vec3);
    vectorOfVectors.push_back(vec2);
    vectorOfVectors.push_back(vec1);

    std::cout << vec1.size() << " " << vec2.size() << " " << vec3.size() << std::endl;
    std::cout << "size of vectorOfVectors " << vectorOfVectors.size() << std::endl;


    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> class_indices;

      std::cout<<number_of_classes<< " number of classes "<< std::endl;

    auto processed_result = v7object.postprocess(vectorOfVectors, confidence, number_of_classes, input_image_height, input_image_width, batch_index);
    std::tie(bboxes, scores, class_indices) = processed_result;

    std::cout << "size of boxes " << bboxes.size() << std::endl;


    std::vector<uint64_t> after_nms_indices;

    after_nms_indices = v7object.nms(bboxes, scores, nms_threshold);

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
    cout << after_nms_bboxes.size() << "           " << after_nms_scores.size() << "               " << after_nms_class_indices.size() << endl;
    // cout << save_path + filename << endl;
    // cout << imagePath << endl;
  // for (int i = 0; i < after_nms_bboxes.size(); ++i)
  // {
  //     const std::array<float, 4> &bbox = after_nms_bboxes[i];

  //     float x1 = bbox[0];
  //     float y1 = bbox[1];
  //     float x2 = bbox[2];
  //     float y2 = bbox[3];
      // std::cout << "Bounding Box " << i << ": (" << int(x1) << ", " << int(y1) << ", " << int(x2) << ", " << int(y2) << " ,"
      //           << after_nms_scores[i] << " ," << after_nms_class_indices[i] << ")" << std::endl;
  // }
    // y.show_boxes(image, after_nms_bboxes, after_nms_class_indices, after_nms_scores, save_path + filename);
    // v7object.draw_and_save(image, after_nms_bboxes, after_nms_class_indices, after_nms_scores, "/workspace/yolo_onnx_release/image/result.jpg");

  // image.release();
    // std::cout << "adsd" << std::endl;
  delete[] blob;
    

  return std::make_tuple(after_nms_bboxes,after_nms_class_indices,after_nms_scores);
}

PYBIND11_MODULE(run_yolo_onnx, m) {
    py::class_<YoloDetectorv3>(m, "YoloDetectorv3")
    .def(py::init<>())
    .def("initialize", &YoloDetectorv3::initialize)
    .def("detect", &YoloDetectorv3::detect);

    py::class_<YoloDetectorv7>(m, "YoloDetectorv7")
    .def(py::init<>())
    .def("initialize", &YoloDetectorv7::initialize)
    .def("detect", &YoloDetectorv7::detect);

}











// int main() {
//     std::string model_path = "/home/manish/Documents/older/alexandria/face_detection.tiny_yolov3/v3/onnx/yolo_tiny_25_07.onnx";
//     YoloDetectorv3 detector;
//     detector.initialize(model_path);
//     std::vector<std::array<float, 4>> after_nms_bboxes;
//     std::vector<uint64_t> after_nms_class_indices;
//     std::vector<float> after_nms_scores;

//     std::tie(after_nms_bboxes, after_nms_class_indices, after_nms_scores) = detector.detect("/home/manish/tt/2.jpg");


//     // Now the detector is initialized with the specified ONNX model
//     // You can call other member functions of YoloDetectorv3 as needed
//     return 0;
// }