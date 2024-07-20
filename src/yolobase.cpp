#include <yolobase.h>

Yolobase::Yolobase(const mtx::ModelConfig &config)
{

  this->PREPROCESS = config.get_preprocess_struct();
  this->BATCH_SIZE = config.get_input_batch_size();
  this->input_shape_info = config.get_input_shape_info();
  this->IMG_WIDTH = this->input_shape_info["width"];
  this->IMG_HEIGHT = this->input_shape_info["height"];
  this->IMG_CHANNEL = this->input_shape_info["channels"];

  this->MEAN_VALUE = this->PREPROCESS.MEAN_VALUE;
  this->SCALE_FACTOR = this->PREPROCESS.SCALE_FACTOR;
  this->USE_LETTERBOX = this->PREPROCESS.USE_LETTERBOX;
  this->PADDING_COLOR = this->PREPROCESS.PADDING_COLOR;
  this->NORM = this->PREPROCESS.NORM;

  this->IMG_MODE = PREPROCESS.IMG_MODE;
  this->IMG_ORDER = PREPROCESS.IMG_ORDER;
  this->dst = new float[this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_WIDTH *
                        this->IMG_HEIGHT];
  //  ###############################################################################################
  this->model_path = config.get_model_path();
  this->gpu_idx = 0; // hardcode GPU index as it always starts from 0 at runtime
  this->provider = config.get_provider();
  this->classes = config.get_labels();
  this->conf = config.get_confidence_threshold();
  this->softnms, this->classwise_nms = config.get_nms_info();
  this->classwise_nms_threshold = config.get_classwise_nms_threshold();
  this->nms_threshold = config.get_nms_threshold();
  this->num_classes = this->classes.size();
  auto shape_info = config.get_input_shape_info();
  this->output_shape_info = config.get_output_shape_info(true);
  this->model_input_shape = {shape_info["height"], shape_info["width"], shape_info["channels"]};
  this->async = true;
  this->PREPROCESS_INFO = config.get_preprocess_struct();
  std::vector<std::vector<int64_t>> output_shape;
  for (auto &vv : this->output_shape_info)
  {
    output_shape.push_back(vv.second);
  }
  this->batch_size = config.get_input_batch_size();
  this->draw_blobs_on_frames = config.get_draw_blobs_on_frames();
  this->named_anchor_boxes = config.get_named_anchors();
  uint8_t gpuIdx = 0;

  std::cout << "\t Provider: " << this->provider << std::endl;

  if (this->provider == "onnx-openvino-cpu" || this->provider == "onnx-openvino-gpu")
  {
    std::cout << "Using OpenVINO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

    this->yolo_infer = std::make_unique<mtx::OpenVINOInferenceEngine>(this->model_path, this->provider,
                                                                      std::vector<std::vector<long int>>{{this->batch_size, this->model_input_shape[2], 
                                                                          this->model_input_shape[0], this->model_input_shape[1]}},
                                                                      output_shape, this->PREPROCESS_INFO);
  }
  else if (this->provider == "onnx-gpu" || this->provider == "onnx-cpu")
  {
    std::cout << "Using ONNX Runtime !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    this->yolo_infer = std::make_unique<mtx::ORTInferenceEngine>(this->model_path, this->provider, gpuIdx,
                                                                 std::vector<std::vector<long int>>{{this->batch_size, this->model_input_shape[2], 
                                                                      this->model_input_shape[0], this->model_input_shape[1]}},
                                                                 output_shape, this->PREPROCESS_INFO);
    this->async = false;
  }
  else if (this->provider == "onnx-tensorrt")
  {
    std::cout << "Using TensorRT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    this->yolo_infer = std::make_unique<mtx::TRTInferenceEngine>(this->model_path, this->provider,
                                                                 std::vector<std::vector<long int>>{{this->batch_size, this->model_input_shape[2],
                                                                   this->model_input_shape[0], this->model_input_shape[1]}}, 
                                                                   output_shape, this->PREPROCESS_INFO, false);
  }
  else
  {
    throw std::runtime_error("Unsupported provider");
  }
}

py::list Yolobase::detect_ov(py::array &input_array)
{

  std::cout << "------------------------------------------------------------------" << std::endl;

  std::cout << "Input_array shape " << input_array.shape(0) << std::endl;
  py::buffer_info buf = input_array.request();

  float *ptr = static_cast<float *>(buf.ptr);
  // float *const_ptr = const_cast<float *>(ptr);

  std::cout << "before enqueue" << std::endl;
  this->yolo_infer->enqueue(ptr);

  std::cout << "enqueue done" << std::endl;

  const auto &outputValues = this->yolo_infer->execute_network();

  std::cout << "called the function detect_ov" << std::endl;
  std::cout << "outputValues.size() " << outputValues.size() << std::endl;

  std::cout << "------------------------------------------------------------------" << std::endl;

  py::list pylist = py::list();
  for (int i = 0; i < outputValues.size(); i++)
  {
    float *a = outputValues[i].data;
    std::cout << "output shape " << outputValues[i].element_count << std::endl;
    auto capsule = py::capsule(a, [](void *a)
                               { delete reinterpret_cast<float *>(a); });

    auto py_arr = py::array(outputValues[i].element_count, a, capsule);

    pylist.attr("append")(py_arr);
    py_arr.release();
    capsule.release();
  }

  return pylist;
}


cv::Mat Yolobase::numpyArrayToMat(py::array_t<uchar> arr)
{
  auto buffer = arr.request();
  int height = buffer.shape[0];
  int width = buffer.shape[1];
  int channels = buffer.shape[2];

  cv::Mat mat(height, width, CV_8UC(channels), buffer.ptr);
  return mat;
}

float Yolobase::sigmoid(float x) const
{
  return 1.0 / (1.0 + std::exp(-x));
}
cv::Mat Yolobase::create_letterbox(const cv::Mat &frame) const
{
  std::cout << "using letterboxing be aware this is in yolobase" << std::endl;
  int origW = frame.cols, origH = frame.rows;
  std::vector<float> originImageSize{static_cast<float>(origH), static_cast<float>(origW)};
  float scale = std::min<float>(1.0 * this->IMG_WIDTH / origW, 1.0 * this->IMG_HEIGHT / origH);
  cv::Mat scaled_image;
  cv::resize(frame, scaled_image, cv::Size(), scale, scale, cv::INTER_CUBIC);

  cv::Mat processed_image(this->IMG_HEIGHT, this->IMG_WIDTH, CV_8UC3, cv::Scalar(this->PADDING_COLOR[0], this->PADDING_COLOR[1], this->PADDING_COLOR[2]));

  // std::cout << this->PADDING_COLOR[0] << std::endl;
  scaled_image.copyTo(processed_image(cv::Rect((this->IMG_WIDTH - scaled_image.cols) / 2,
                                               (this->IMG_HEIGHT - scaled_image.rows) / 2,
                                               scaled_image.cols, scaled_image.rows)));
  return processed_image;
}
py::array Yolobase::preprocess_batch(py::list &batch)
{
  for (int64_t b = 0; b < batch.size(); ++b)
  {
    py::array_t<uchar> np_array = batch[b].cast<py::array_t<uchar>>();
    cv::Mat img = numpyArrayToMat(np_array);

    if (this->USE_LETTERBOX)
    {
      std::cout << "using letterboxing be aware " << std::endl;
      const unsigned char *src = this->create_letterbox(img).data;
      this->preprocess(src, b);
    }
    else
    {
      cv::Mat temp;
      cv::resize(img, temp, cv::Size(this->IMG_WIDTH, this->IMG_HEIGHT), 0, 0, cv::INTER_LINEAR);
      cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);
      this->preprocess(temp.data, b);
    }
  }

  auto capsule = py::capsule(dst, [](void *dst)
                             { delete reinterpret_cast<float *>(dst); });

  py::array img_array = py::array(this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_WIDTH * this->IMG_HEIGHT, dst, capsule);
  capsule.release();

  return img_array;
}

inline void Yolobase::preprocess(const unsigned char *src, const int64_t b)
{

  if (this->PREPROCESS.IMG_ORDER == NETWORK_INPUT_ORDER::NCHW)
  {
#pragma omp parallel for
    for (int64_t c = 0; c < this->IMG_CHANNEL; ++c)
    {
      for (int64_t i = 0; i < this->IMG_HEIGHT; ++i)
      {
        for (int64_t j = 0; j < this->IMG_WIDTH; ++j)
        {
          this->dst[b * this->IMG_CHANNEL * this->IMG_WIDTH * this->IMG_HEIGHT +
                    c * this->IMG_HEIGHT * this->IMG_WIDTH + i * this->IMG_WIDTH + j] =
              (float)(src[i * this->IMG_WIDTH * this->IMG_CHANNEL + j * this->IMG_CHANNEL + c] - this->PREPROCESS.MEAN_VALUE[c]) / this->PREPROCESS.SCALE_FACTOR[c];
        }
      }
    }
  }
  else
  {
#pragma omp parallel for
    for (int64_t i = 0; i < this->IMG_HEIGHT; ++i)
    {
      for (int64_t j = 0; j < this->IMG_WIDTH; ++j)
      {
        for (int64_t c = 0; c < this->IMG_CHANNEL; ++c)
        {
          this->dst[b * this->IMG_CHANNEL * this->IMG_WIDTH * this->IMG_HEIGHT +
                    i * this->IMG_WIDTH * this->IMG_CHANNEL + j * this->IMG_CHANNEL + c] =
              (float)(src[i * this->IMG_WIDTH * this->IMG_CHANNEL + j * this->IMG_CHANNEL + c] - this->PREPROCESS.MEAN_VALUE[c]) / this->PREPROCESS.SCALE_FACTOR[c];
        }
      }
    }
  }
}
