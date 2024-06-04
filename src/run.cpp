#include <run.h>
namespace py = pybind11;

void YoloDetectorv3::initialize(const std::string &model_path, int height, int width, 
int channels, int batch_size)
    {
        this->model_path_ = model_path;
        this->height = height;
        this->width = width;
        this->channels = channels;
        this->batch_size = batch_size;
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
std::vector<std::vector<float>> YoloDetectorv3::detect(
    std::vector<float> input_tensor)
{
    float *blob = new float[channels * height * width];

    std::vector<int64_t> inputTensorShape{batch_size, channels, height, width};
    std::copy(input_tensor.begin(), input_tensor.end(), blob);

    size_t inputTensorSize = vectorProduct(inputTensorShape);
    // cout << inputTensorSize << endl;
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    // std::cout<< "size of input tensor " << inputTensorValues.size() << endl;

    auto inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape1 = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto outputShape2 = session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();

    std ::vector<float> inputValues = inputTensorValues;
    inputShape[0] = 1;

    // cout << inputValues.size() << endl;

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
    char const *const *names_of_outputs = output_names;
    std ::vector<float> outputTensor;

    auto outputValues1 = session_->Run(
        Ort::RunOptions{nullptr},
        ii,
        &inputOnnxTensor, inputNames.size(), names_of_outputs, 2);

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

    // std::cout << vec1.size() << "    " << vec2.size() << endl;
    return vectorOfVectors;
}


// // =====================================================================================================================



void YoloDetectorv7::initialize(const std::string &model_path, int height, int width, 
int channels, int batch_size)
    {
        this->model_path_ = model_path;
        this->height = height;
        this->width = width;
        this->channels = channels;
        this->batch_size = batch_size;

        session_ = new Ort::Session(env_, this->model_path_.c_str(), Ort::SessionOptions());
        input_name_ = session_->GetInputName(0, allocator_);
        // std::cout << input_name_ << std::endl;

        int numInputs = session_->GetInputCount();
        int numOutputs = session_->GetOutputCount();
       

        output_name1_ = session_->GetOutputName(0, allocator_);
        output_name2_ = session_->GetOutputName(1, allocator_);
        output_name3_ = session_->GetOutputName(2, allocator_);
    
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

std::vector<std::vector<float>> YoloDetectorv7::detect(std::vector<float> input_tensor)
{

    // int batch_index = 0;

    // Yolov7 v7object(number_of_classes, width, anchors);
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
    // std::cout << sizeof(output_names) << "      " << sizeof(output_names[0]) << sizeof(output_names[1]) << sizeof(output_names[2]) << std::endl;

    static const size_t NUM_OUTPUTS = sizeof(output_names) / sizeof(output_names[0]);

    OrtValue *p_output_tensors[NUM_OUTPUTS] = {nullptr};

    char const *const *ii = &input_name_;
    char const *const *names_of_outputs = output_names;
    std ::vector<float> outputTensor;

    auto outputValues1 = session_->Run(
        Ort::RunOptions{nullptr},
        ii,
        &inputOnnxTensor, inputNames.size(), names_of_outputs, 3);

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
    return vectorOfVectors;

}

PYBIND11_MODULE(run_yolo_onnx, m)
{
    py::class_<YoloDetectorv3>(m, "YoloDetectorv3")
        .def(py::init<>())
        .def("initialize", &YoloDetectorv3::initialize)
        .def("detect", &YoloDetectorv3::detect);

    py::class_<Yolov3>(m, "Yolov3")
        .def(py::init<int, int, std::vector<std::vector<float>>>())
        .def("preprocess", &Yolov3::preprocess)
        .def("postprocess", &Yolov3::postprocess);




    py::class_<YoloDetectorv7>(m, "YoloDetectorv7")
        .def(py::init<>())
        .def("initialize", &YoloDetectorv7::initialize)
        .def("detect", &YoloDetectorv7::detect);

    py::class_<Yolov7>(m, "Yolov7")
        .def(py::init<int, int, std::vector<std::vector<float>>>())
        .def("preprocess", &Yolov7::preprocess)
        .def("postprocess", &Yolov7::postprocess);

}
