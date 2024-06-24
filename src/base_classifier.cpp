#include "base_classifier.h"

Base_classifier::Base_classifier(const std::string &model_path, int batch_size, std::string provider)
{
    this->BATCH_SIZE = batch_size;
    this->model_path_ = model_path;
    this->provider = provider;

    std::cout << model_path << std::endl;
    std::string _name = "onnx_runtime" + std::to_string(rand());
    this->env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, _name.c_str());
    this->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    this->sessionOptions.DisableMemPattern();
    this->sessionOptions.DisableCpuMemArena();
    this->sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    if (provider == "cpu")
    {
        std::cout << "Using CPU" << std::endl;
        this->sessionOptions = Ort::SessionOptions();
    }
    else if (provider == "gpu")
    {
        int gpu_id = 0;
        std::cout << "Using GPU with ID: " << gpu_id << "" << std::endl;
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = gpu_id;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        this->sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        this->sessionOptions.SetIntraOpNumThreads(1);
        this->sessionOptions.SetInterOpNumThreads(6);
    }

    this->session_ = new Ort::Session(env_, this->model_path_.c_str(), sessionOptions);
    this->input_count = session_->GetInputCount();
    this->output_count = session_->GetOutputCount();

    this->inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Input shape: " << inputShape[0] << " " << inputShape[1] << " " << inputShape[2] << " " << inputShape[3] << std::endl;
    for (int i = 0; i < this->input_count; ++i)
    {
        input_names.push_back(session_->GetInputNameAllocated(i, allocator_).get());
        std::cout << "Input name " << i << ": " << input_names[i] << std::endl;
    }

    for (int i = 0; i < this->output_count; ++i)
    {
        output_names.push_back(session_->GetOutputNameAllocated(i, allocator_).get());
        std::cout << "Output name " << i << ": " << output_names[i] << std::endl;
    }
    std::cout << "Created the session " << std::endl;

    this->IMG_HEIGHT = inputShape[2];
    this->IMG_WIDTH = inputShape[3];
    this->IMG_CHANNEL = inputShape[1];
    this->inputShape[0] = this->BATCH_SIZE;
    this->inputTensorSize = this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_HEIGHT * this->IMG_WIDTH;

    this->dst = new float[this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_WIDTH *
                          this->IMG_HEIGHT];

    for (const auto &str : input_names)
    {
        names_of_inputs_ptr.push_back(str.c_str());
    }
    this->names_of_inputs_cstr = names_of_inputs_ptr.data();

    for (const auto &str : output_names)
    {
        names_of_outputs_ptr.push_back(str.c_str());
    }

    this->names_of_outputs_cstr = names_of_outputs_ptr.data();
}

Base_classifier::~Base_classifier()
{
    // Destructor cleanup if necessary
}

py::list Base_classifier::infer(py::array &input_array)
{
    py::buffer_info buf = input_array.request();

    float *ptr = static_cast<float *>(buf.ptr);
    float *const_ptr = const_cast<float *>(ptr);
    auto inputOnnxTensor = Ort::Value::CreateTensor<float>(this->info,
                                                           const_ptr, this->inputTensorSize,
                                                           this->inputShape.data(), this->inputShape.size());
    auto outputValues = session_->Run(this->runOptions,
                                      names_of_inputs_cstr,
                                      &inputOnnxTensor, this->input_count,
                                      names_of_outputs_cstr, this->output_count);

    py::list pylist = py::list();
    for (int i = outputValues.size() - 1; i >= 0; i--)
    {
        float *a = outputValues[i].GetTensorMutableData<float>();

        auto capsule = py::capsule(a, [](void *a)
                                   { delete reinterpret_cast<float *>(a); });
        auto py_arr = py::array(outputValues[i].GetTensorTypeAndShapeInfo().GetElementCount(), a, capsule);

        pylist.attr("append")(py_arr);
        py_arr.release();
        capsule.release();
    }

    return pylist;
}
