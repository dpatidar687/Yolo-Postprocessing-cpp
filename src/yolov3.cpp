
#include <yolov3.h>
#include <opencv2/core.hpp>
namespace
{
  template <typename T>
  std::deque<size_t> sortIndexes(const std::vector<T> &v)
  {
    std::deque<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(std::begin(indices), std::end(indices), [&v](size_t i1, size_t i2)
                     { return v[i1] < v[i2]; });

    return indices;
  }
}

Yolov3::Yolov3(int number_of_classes, std::vector<std::vector<float>> anchors, const std::string &model_path, int batch_size, std::string provider)
{
  this->ANCHORS = anchors;
  this->BATCH_SIZE = batch_size;
  this->model_path_ = model_path;
  this->provider = provider;
  this->number_of_classes = number_of_classes;

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

  // this->input_name = session_->GetInputNameAllocated(0, allocator_).get();
  this->inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  std::cout << "Input shape: " << inputShape[0] << " " << inputShape[1] << " " << inputShape[2] << " " << inputShape[3] << std::endl;
  this->IMG_HEIGHT = inputShape[2];
  this->IMG_WIDTH = inputShape[3];
  this->IMG_CHANNEL = inputShape[1];
  this->BATCH_SIZE = batch_size;
  this->inputShape[0] = batch_size;
  this->inputTensorSize = this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_HEIGHT * this->IMG_WIDTH;

  // std::cout << "Input name: " << input_name << std::endl;

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

  for (auto &anchor : this->ANCHORS)
    this->NUM_ANCHORS.push_back(anchor.size() / 2);

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

float Yolov3::sigmoid(float x) const
{
  return 1.0 / (1.0 + std::exp(-x));
}

cv::Mat Yolov3::numpyArrayToMat(py::array_t<uchar> arr)
{
  auto buffer = arr.request();
  int height = buffer.shape[0];
  int width = buffer.shape[1];
  int channels = buffer.shape[2];

  cv::Mat mat(height, width, CV_8UC(channels), buffer.ptr);

  return mat;
}
py::array Yolov3::preprocess_batch(py::list &batch)
{
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << &batch << endl;

  auto start2 = Yolov3::get_current_ns();
  for (int64_t b = 0; b < batch.size(); ++b)
  {
    py::array_t<uchar> np_array = batch[b].cast<py::array_t<uchar>>();
    cv::Mat img = numpyArrayToMat(np_array);
    cv::Mat temp;
    cv::resize(img, temp, cv::Size(this->IMG_WIDTH, this->IMG_HEIGHT), 0, 0,
               cv::INTER_LINEAR);
    cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);

    this->preprocess(temp.data, b);
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = finish - start;
  std::cout << "Elapsed Time in preprocessing : " << elapsed.count() << " mili seconds" << std::endl;


  auto finish2 = Yolov3::get_current_ns();
  std::cout<< "Elapsed Time in preprocessing cpp : " << (finish2 - start2) << " nano seconds" << std::endl;

  // auto start3 = Yolov3::get_current_ns();
  // this->dst = (float *)cv::dnn::blobFromImages(processed_images, 1.0, cv::Size(this->IMG_WIDTH, this->IMG_HEIGHT), cv::Scalar(), swap_channels, false).data;
  // auto finish3 = Yolov3::get_current_ns();
  // std::cout<< "Elapsed Time in preprocessing cv::dnn::blobFromImages : " << (finish3 - start3) << " nano seconds" << std::endl;

  auto capsule = py::capsule(dst, [](void *dst)
                             { delete reinterpret_cast<float *>(dst); });
                             
  
  py::array img_array = py::array(this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_WIDTH * this->IMG_HEIGHT, dst, capsule);
  capsule.release();
  return img_array;
}

inline void Yolov3::preprocess(const unsigned char *src, const int64_t b)
{
  // auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < this->IMG_HEIGHT; ++i)
  {
    for (int64_t j = 0; j < this->IMG_WIDTH; ++j)
    {
      for (int64_t c = 0; c < this->IMG_CHANNEL; ++c)
      {
        this->dst[b * this->IMG_CHANNEL * this->IMG_WIDTH * this->IMG_HEIGHT +
                  c * this->IMG_HEIGHT * this->IMG_WIDTH + i * this->IMG_WIDTH + j] =
            src[i * this->IMG_WIDTH * this->IMG_CHANNEL + j * this->IMG_CHANNEL + c] / 255.0;
      }
    }
  }
  // auto finish = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = finish - start;
  // std::cout << "Elapsed Time in loop of pre only : " << elapsed.count() << "mili seconds" << std::endl;
}

py::list Yolov3::detect(py::array &input_array)
{
  // auto start = std::chrono::high_resolution_clock::now();

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

  for (size_t i = 0; i < outputValues.size(); ++i)
  {

    float *a = outputValues[i].GetTensorMutableData<float>();

    auto capsule = py::capsule(a, [](void *a)
                               { delete reinterpret_cast<float *>(a); });
    auto py_arr = py::array(outputValues[i].GetTensorTypeAndShapeInfo().GetElementCount(), a, capsule);

    pylist.attr("append")(py_arr);
    py_arr.release();
    capsule.release();
  }

  // auto finish = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = finish - start;
  // std::cout << "Elapsed Time in inference : " << elapsed.count() << " seconds" << std::endl;
  // std::cout << &pylist << std::endl;
  return pylist;
}

py::list Yolov3::postprocess_batch(py::list &infered,
                                   const float confidenceThresh, const float nms_threshold,
                                   const int64_t input_image_height, const int64_t input_image_width)
{
  // auto start = std::chrono::high_resolution_clock::now();

  int batch = this->BATCH_SIZE;
  const int64_t num_classes = this->number_of_classes;
  // std::vector<std::tuple<std::vector<std::array<float, 4> >, std::vector<uint64_t>, std::vector<float>>> processed_result_vector;
  py::list processed_result_vector;

  for (int batch_ind = 0; batch_ind < batch; batch_ind++)
  {
    // std::tuple<std::vector<std::array<float, 4> >, std::vector<uint64_t>, std::vector<float>> processed_result  =
    py::tuple processed_result = this->postprocess(infered, confidenceThresh, nms_threshold,
                                                   num_classes, input_image_height, input_image_width, batch_ind);

    processed_result_vector.append(processed_result);
  }

  // auto finish = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = finish - start;
  // std::cout << "Elapsed Time in postprocessing : " << elapsed.count() << " seconds" << std::endl;
  return processed_result_vector;
}

py::tuple Yolov3::postprocess(py::list &infered,
                              const float confidenceThresh, const float nms_threshold, const uint16_t num_classes,
                              const int64_t input_image_height, const int64_t input_image_width,
                              const int64_t batch_ind)
{

  std::vector<std::array<float, 4>> bboxes;
  std::vector<float> scores;
  std::vector<uint64_t> classIndices;

  for (int i = 0; i < len(infered); i++)
  {

    py::array_t<float> array = infered[i].cast<py::array_t<float>>();
    py::buffer_info buf = array.request();
    std::vector<float> vec(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.size);

    this->post_process_feature_map(vec.data(), confidenceThresh, num_classes,
                                   input_image_height, input_image_width, 32 / pow(2, i),
                                   this->ANCHORS[i], this->NUM_ANCHORS[i], bboxes, scores,
                                   classIndices, batch_ind);
  }

  std::vector<uint64_t> after_nms_indices;
  after_nms_indices = nms(bboxes, scores, nms_threshold);

  // std::vector<std::array<float, 4> > after_nms_bboxes;
  // std::vector<uint64_t> after_nms_class_indices;
  // std::vector<float> after_nms_scores;

  py::list after_nms_bboxes;
  py::list after_nms_class_indices;
  py::list after_nms_scores;
  // py::list boxes ;
  // after_nms_bboxes.reserve(after_nms_indices.size());
  // after_nms_class_indices.reserve(after_nms_indices.size());
  // after_nms_scores.reserve(after_nms_indices.size());

  for (const auto idx : after_nms_indices)
  {
    after_nms_bboxes.append(bboxes[idx]);
    after_nms_class_indices.append(classIndices[idx]);
    after_nms_scores.append(scores[idx]);
    // boxes.append([bboxes[idx], classIndices[idx], scores[idx]]);
  }
  return py::make_tuple(after_nms_bboxes, after_nms_class_indices, after_nms_scores);
}

/*
py::list Yolov3::postprocess_batch(const ptr_wrapper<std::vector<std::vector<float>>> &infered,
                                   const float confidenceThresh, const float nms_threshold,
                                   const int64_t input_image_height, const int64_t input_image_width)
{
  // auto start = std::chrono::high_resolution_clock::now();

  int batch = this->BATCH_SIZE;
  const int64_t num_classes = this->number_of_classes;
  // std::vector<std::tuple<std::vector<std::array<float, 4> >, std::vector<uint64_t>, std::vector<float>>> processed_result_vector;
  py::list processed_result_vector;

  for (int batch_ind = 0; batch_ind < batch; batch_ind++)
  {
    // std::tuple<std::vector<std::array<float, 4> >, std::vector<uint64_t>, std::vector<float>> processed_result  =
    py::tuple processed_result = this->postprocess(infered, confidenceThresh, nms_threshold,
                                                   num_classes, input_image_height, input_image_width, batch_ind);

    processed_result_vector.append(processed_result);
  }

  // auto finish = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = finish - start;
  // std::cout << "Elapsed Time in postprocessing : " << elapsed.count() << " seconds" << std::endl;
  return processed_result_vector;
}

// std::tuple<std::vector<std::array<float, 4> >, std::vector<uint64_t>, std::vector<float>>
py::tuple Yolov3::postprocess(const ptr_wrapper<std::vector<std::vector<float> > > &infered,
                              const float confidenceThresh, const float nms_threshold, const uint16_t num_classes,
                              const int64_t input_image_height, const int64_t input_image_width,
                              const int64_t batch_ind)
{

  std::vector<std::array<float, 4> > bboxes;
  std::vector<float> scores;
  std::vector<uint64_t> classIndices;

  for (int i = 0; i < infered.get()->size(); i++)
  {
    this->post_process_feature_map(infered.get()->at(i).data(), confidenceThresh, num_classes,
                                   input_image_height, input_image_width, 32 / pow(2, i),
                                   this->ANCHORS[i], this->NUM_ANCHORS[i], bboxes, scores,
                                   classIndices, batch_ind);
  }

  std::vector<uint64_t> after_nms_indices;

  after_nms_indices = nms(bboxes, scores, nms_threshold);

  std::vector<std::array<float, 4> > after_nms_bboxes;
  std::vector<uint64_t> after_nms_class_indices;
  std::vector<float> after_nms_scores;

  after_nms_bboxes.reserve(after_nms_indices.size());
  after_nms_class_indices.reserve(after_nms_indices.size());
  after_nms_scores.reserve(after_nms_indices.size());

  for (const auto idx : after_nms_indices)
  {
    after_nms_bboxes.emplace_back(bboxes[idx]);
    after_nms_class_indices.emplace_back(classIndices[idx]);
    after_nms_scores.emplace_back(scores[idx]);
  }
  return py::make_tuple(after_nms_bboxes, after_nms_class_indices, after_nms_scores);
}
'''*/
void Yolov3::post_process_feature_map(const float *out_feature_map, const float confidenceThresh,
                                      const int num_classes, const int64_t input_image_height,
                                      const int64_t input_image_width, const int factor,
                                      const std::vector<float> &anchors, const int64_t &num_anchors,
                                      std::vector<std::array<float, 4>> &bboxes,
                                      std::vector<float> &scores,
                                      std::vector<uint64_t> &classIndices, const int b)
{

  const int64_t feature_map_height = this->IMG_HEIGHT / factor;
  const int64_t feature_map_width = this->IMG_WIDTH / factor;
  const int64_t feature_map_size = feature_map_width * feature_map_height;
  const int64_t num_filters = (num_classes + 5) * num_anchors;
  const int64_t num_boxes = feature_map_size * num_filters;
  float tmpScores[num_classes];

  std::vector<float> outputData(out_feature_map + b * num_boxes,
                                out_feature_map + (b + 1) * num_boxes);

  for (uint64_t i = 0; i < feature_map_size; ++i)
  {
    for (uint64_t j = 0; j < num_anchors; ++j)
    {
      for (uint64_t k = 0; k < num_classes; ++k)
      {
        tmpScores[k] = outputData[i + feature_map_size * ((num_classes + 5) * j + k + 5)];
      }
      uint64_t maxIdx =
          std::distance(tmpScores, std::max_element(tmpScores, tmpScores + num_classes));
      const float &class_confidence = sigmoid(tmpScores[maxIdx]);
      const float &box_confidence =
          sigmoid(outputData[i + feature_map_size * ((num_classes + 5) * j + 4)]);

      const float &probability = class_confidence * box_confidence;
      if (box_confidence < confidenceThresh) // TODO check if giving correct result when Nan
        continue;

      float xcenter = outputData[i + feature_map_size * ((num_classes + 5) * j)];
      float ycenter = outputData[i + feature_map_size * ((num_classes + 5) * j + 1)];
      float width = outputData[i + feature_map_size * ((num_classes + 5) * j + 2)];
      float height = outputData[i + feature_map_size * ((num_classes + 5) * j + 3)];

      xcenter = (sigmoid(xcenter) + (i % feature_map_width)) / feature_map_width;   // [0-1]
      ycenter = (sigmoid(ycenter) + (i / feature_map_height)) / feature_map_height; // [0-1]

      float w = expf(width);
      width = expf(width) * anchors[2 * j] / this->IMG_WIDTH;        // [0-1]? TODO verify this
      height = expf(height) * anchors[2 * j + 1] / this->IMG_HEIGHT; // [0-1]? TODO verify this
      float xt = xcenter - width / 2;
      float yt = ycenter - height / 2;

      std::array<float, 4> out_box =
          post_process_box(xt, yt, width, height, input_image_height, input_image_width);
      bboxes.emplace_back(out_box);
      scores.emplace_back(probability);
      classIndices.emplace_back(maxIdx);
    }
  }
}

std::array<float, 4> Yolov3::post_process_box(const float &xt, const float &yt, const float &width,
                                              const float &height,
                                              const int64_t &input_image_height,
                                              const int64_t &input_image_width) const
{
  float xmin, xmax, ymin, ymax;

  xmin = xt * input_image_width;
  ymin = yt * input_image_height;
  xmax = xmin + width * input_image_width;
  ymax = ymin + height * input_image_height;

  xmin = std::max<float>(xmin, 0.0);
  ymin = std::max<float>(ymin, 0.0);
  xmax = std::min<float>(xmax, input_image_width - 1);
  ymax = std::min<float>(ymax, input_image_height - 1);

  return std::array<float, 4>{xmin, ymin, xmax, ymax};
}

std::vector<uint64_t> Yolov3::nms(const std::vector<std::array<float, 4>> &bboxes,
                                  const std::vector<float> &scores,
                                  float overlapThresh,
                                  uint64_t topK)
{
  if (bboxes.size() == 0)
  {
    return std::vector<uint64_t>();
  }
  uint64_t boxesLength = bboxes.size();
  const uint64_t realK = std::max(std::min(boxesLength, topK), static_cast<uint64_t>(1));

  std::vector<uint64_t> keepIndices;
  keepIndices.reserve(realK);

  std::deque<uint64_t> sortedIndices = ::sortIndexes(scores);
  // keep only topk bboxes for (uint64_t i = 0; i < boxesLength - realK; ++i) { sortedIndices.pop_front(); }

  std::vector<float> areas;
  areas.reserve(boxesLength);
  std::transform(std::begin(bboxes), std::end(bboxes), std::back_inserter(areas),
                 [](const auto &elem)
                 { return (elem[2] - elem[0]) * (elem[3] - elem[1]); });

  while (!sortedIndices.empty())
  {
    uint64_t currentIdx = sortedIndices.back();
    keepIndices.emplace_back(currentIdx);

    if (sortedIndices.size() == 1)
    {
      break;
    }

    sortedIndices.pop_back();
    std::vector<float> ious;
    ious.reserve(sortedIndices.size());

    const auto &curBbox = bboxes[currentIdx];
    const float curArea = areas[currentIdx];

    std::deque<uint64_t> newSortedIndices;

    for (const uint64_t elem : sortedIndices)
    {
      const auto &bbox = bboxes[elem];
      float tmpXmin = std::max(curBbox[0], bbox[0]);
      float tmpYmin = std::max(curBbox[1], bbox[1]);
      float tmpXmax = std::min(curBbox[2], bbox[2]);
      float tmpYmax = std::min(curBbox[3], bbox[3]);

      float tmpW = std::max<float>(tmpXmax - tmpXmin, 0.0);
      float tmpH = std::max<float>(tmpYmax - tmpYmin, 0.0);

      const float intersection = tmpW * tmpH;
      const float tmpArea = areas[elem];
      const float unionArea = tmpArea + curArea - intersection;
      const float iou = intersection / unionArea;

      if (iou <= overlapThresh)
      {
        newSortedIndices.emplace_back(elem);
      }
    }

    sortedIndices = newSortedIndices;
  }
  return keepIndices;
}