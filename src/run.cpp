#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
// #include <yolov3.h>
// #include <yolov7.h>
#include <run.h>
#include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Include this header for automatic conversion of STL containers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


void YoloDetectorv3::initialize(const std::string &model_path, int height, int width, int channels, int number_of_classes, int batch_size, float confidence, float nms_threshold,
                    std::vector<std::vector<float>> anchors, std::string model)
    {
        this->model_path_ = model_path;
        this->height = height;
        this->width = width;
        this->channels = channels;
        this->number_of_classes = number_of_classes;
        this->batch_size = batch_size;
        this->confidence = confidence;
        this->nms_threshold = nms_threshold;
        this->anchors = anchors;
        this->model = model;

        std::cout << model_path << std::endl;
        session_ = new Ort::Session(env_, this->model_path_.c_str(), Ort::SessionOptions());
        input_name_ = session_->GetInputName(0, allocator_);
        output_name1_ = session_->GetOutputName(0, allocator_);
        output_name2_ = session_->GetOutputName(1, allocator_);
        std::cout << "Created the session " << std::endl;
    }
size_t YoloDetectorv3::vectorProduct(const std::vector<int64_t> &vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto &element : vector)
        product *= element;

    return product;
}
std::vector<float>  YoloDetectorv3::preprocess(std::string img_path, int height, int width , int channel , size_t batch_index)
  {
    const cv::Mat &img = cv::imread(img_path);

    int IMG_WIDTH = width;
    int IMG_HEIGHT = height;
    int IMG_CHANNEL = channel;
    cv::Mat img_resized;
    // float *src = new float[channels * height * width];
        // std::cout << "3333 RUN The ALGO "<< std::endl;
        std::vector<float> flat_list(IMG_CHANNEL * IMG_HEIGHT * IMG_WIDTH);

    cv::resize(img, img_resized, cv::Size(IMG_WIDTH, IMG_HEIGHT));

    cv::Mat img_normalized;
    img_normalized = img_resized;
    cv::Mat img_normalized_rgb;
    cv::cvtColor(img_normalized, img_normalized_rgb, cv::COLOR_BGR2RGB);

    const unsigned char *dst = img_normalized_rgb.data;

    for (int i = 0; i < IMG_HEIGHT; ++i)
    {
      for (int j = 0; j < IMG_WIDTH; ++j)
      {
        for (int c = 0; c < IMG_CHANNEL; ++c)
        {
          flat_list[batch_index * IMG_CHANNEL * IMG_HEIGHT * IMG_WIDTH +
              c * IMG_HEIGHT * IMG_WIDTH + i * IMG_WIDTH + j] =
              ((dst[i * IMG_WIDTH * IMG_CHANNEL + j * IMG_CHANNEL + c] / 255.0f));
        }
      }
    }
    return flat_list;
}
std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>> YoloDetectorv3::detect(
    std::vector<float> input_tensor, int input_image_height, int input_image_width)
// std::vector<std::array<float>> YoloDetectorv3::detect(
//     std::vector<float> input_tensor, int input_image_height, int input_image_width
// )
{

    // cv::Mat image = cv::imread(image_path);
    const int batch_index = 0;
    // cout << "calling the detect fucntion "<< endl;

    Yolov3 v3object(number_of_classes, width, anchors);
    float *blob = new float[channels * height * width];

    std::vector<int64_t> inputTensorShape{1, channels, height, width};
    std::copy(input_tensor.begin(), input_tensor.end(), blob);

    size_t inputTensorSize = vectorProduct(inputTensorShape);
    cout << inputTensorSize << endl;
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::cout<< "size of input tensor " << inputTensorValues.size() << endl;

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
    std::vector<uint64_t> class_indices;
    std::vector<float> scores;

    std::cout << vec1.size() << "    " << vec2.size() << endl;

    auto processed_result = v3object.postprocess(vectorOfVectors, confidence, number_of_classes, input_image_height, input_image_width, batch_index);
    std::tie(bboxes, scores, class_indices) = processed_result;

    std::cout << "size of boxes " << bboxes.size() << std::endl;

    std::vector<uint64_t> after_nms_indices;

    after_nms_indices = v3object.nms(bboxes, scores, nms_threshold);

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

    return std::make_tuple(after_nms_bboxes, after_nms_class_indices, after_nms_scores);
    // return vectorOfVectors;
}







// // =====================================================================================================================



void YoloDetectorv7::initialize(const std::string &model_path, int height, int width, int channels, int number_of_classes, int batch_size,
                    float confidence, float nms_threshold, std::vector<std::vector<float>> anchors, std::string model)
    {
        this->model_path_ = model_path;
        this->height = height;
        this->width = width;
        this->channels = channels;
        this->number_of_classes = number_of_classes;
        this->batch_size = batch_size;
        this->confidence = confidence;
        this->nms_threshold = nms_threshold;
        this->anchors = anchors;
        this->model = model;
        std::cout << model_path << std::endl;

        session_ = new Ort::Session(env_, this->model_path_.c_str(), Ort::SessionOptions());
        input_name_ = session_->GetInputName(0, allocator_);
        std::cout << input_name_ << std::endl;

        int numInputs = session_->GetInputCount();
        int numOutputs = session_->GetOutputCount();
        cout << numInputs << " " << numOutputs << endl;

        for (int i = 0; i < numOutputs; i++)
        {
            auto outputShape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "Output Shape: ";
            for (const auto &dim : outputShape)
            {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        output_name1_ = session_->GetOutputName(0, allocator_);
        output_name2_ = session_->GetOutputName(1, allocator_);
        output_name3_ = session_->GetOutputName(2, allocator_);
        std::cout << output_name1_ << std::endl;
        std::cout << output_name2_ << std::endl;
        std::cout << output_name3_ << std::endl;

        std::cout << "Created the session " << std::endl;
    }

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
    std::vector<float> input_tensor, int input_image_height, int input_image_width)
{

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

    std::cout << "Input shape: " << inputShape[0] << " " << inputShape[1] << " " << inputShape[2] << " " << inputShape[3] << endl;

    OrtValue *p_output_tensors[NUM_OUTPUTS] = {nullptr};

    char const *const *ii = &input_name_;
    char const *const *oo = output_names;
    std ::vector<float> outputTensor;

    auto outputValues1 = session_->Run(
        Ort::RunOptions{nullptr},
        ii,
        &inputOnnxTensor, inputNames.size(), oo, 3);

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

    std::cout << number_of_classes << " number of classes " << std::endl;

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
   
    delete[] blob;

    return std::make_tuple(after_nms_bboxes, after_nms_class_indices, after_nms_scores);
}

PYBIND11_MODULE(run_yolo_onnx, m)
{
    py::class_<YoloDetectorv3>(m, "YoloDetectorv3")
        .def(py::init<>())
        .def("initialize", &YoloDetectorv3::initialize)
        .def("detect", &YoloDetectorv3::detect);

    py::class_<Yolov3>(m, "Yolov3")
        .def(py::init<int, int, std::vector<std::vector<float>>>())
        .def("preprocess", &Yolov3::preprocess);



    py::class_<YoloDetectorv7>(m, "YoloDetectorv7")
        .def(py::init<>())
        .def("initialize", &YoloDetectorv7::initialize)
        .def("detect", &YoloDetectorv7::detect);

    py::class_<Yolov7>(m, "Yolov7")
        .def(py::init<int, int, std::vector<std::vector<float>>>())
        .def("preprocess", &Yolov7::preprocess);

}
