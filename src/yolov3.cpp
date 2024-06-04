
#include <yolov3.h>

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

Yolov3::Yolov3(int batch_size, int image_size, std::vector<std::vector<float>> anchors)
{
  this->IMG_HEIGHT = image_size;
  this->IMG_WIDTH = image_size;
  this->IMG_CHANNEL = 3;
  this->ANCHORS = anchors;
  this->BATCH_SIZE = batch_size;
  for (auto &anchor : this->ANCHORS)

    this->NUM_ANCHORS.push_back(anchor.size() / 2); // divide by 2 as it has width and height scales

  this->dst = new float[this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_WIDTH *
                        this->IMG_HEIGHT]; // avoid allocating heap memory at every iteration
}

float Yolov3::sigmoid(float x) const
{
  return 1.0 / (1.0 + std::exp(-x));
}

cv::Mat Yolov3::numpyArrayToMat(py::array_t<uchar> arr) {
    auto buffer = arr.request();
    int height = buffer.shape[0];
    int width = buffer.shape[1];
    int channels = buffer.shape[2];

    cv::Mat mat(height, width, CV_8UC(channels), buffer.ptr);

    return mat;
}
std::vector<float> Yolov3::preprocess(py::array_t<uchar> image_arr, size_t batch_index)
{
  // const cv::Mat &img = cv::imread(img_path);
  cv::Mat img = numpyArrayToMat(image_arr);

  cv::Mat img_resized;

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
  std::cout << "using a preprocess of yolov3 class" << std::endl;
  return flat_list;
}
void Yolov3::initialize(const std::string &model_path, int height, int width,
                        int channels, int batch_size)
{
  this->model_path_ = model_path;
  this->IMG_HEIGHT = height;
  this->IMG_WIDTH = width;
  this->IMG_CHANNEL = channels;
  this->BATCH_SIZE = batch_size;

  std::cout << model_path << std::endl;
  session_ = new Ort::Session(env_, this->model_path_.c_str(), Ort::SessionOptions());
  input_name_ = session_->GetInputName(0, allocator_);
  output_name1_ = session_->GetOutputName(0, allocator_);
  output_name2_ = session_->GetOutputName(1, allocator_);
  std::cout << "Created the session " << std::endl;
}
size_t Yolov3::vectorProduct(const std::vector<int64_t> &vector)
{
  if (vector.empty())
    return 0;

  size_t product = 1;
  for (const auto &element : vector)
    product *= element;

  return product;
}

std::vector<std::vector<float>> Yolov3::detect(
    std::vector<float> input_tensor)
{
  float *blob = new float[this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_WIDTH * this->IMG_HEIGHT];
  std::vector<int64_t> inputTensorShape{this->BATCH_SIZE, this->IMG_CHANNEL, this->IMG_HEIGHT, this->IMG_WIDTH};
  std::copy(input_tensor.begin(), input_tensor.end(), blob);

  size_t inputTensorSize = vectorProduct(inputTensorShape);
  std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

  auto inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto outputShape1 = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto outputShape2 = session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();

  std ::vector<float> inputValues = inputTensorValues;
  inputShape[0] = this->BATCH_SIZE;

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

std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>>
Yolov3::postprocess(const std::vector<std::vector<float>> &inferenceOutput,
                    const float confidenceThresh, const float nms_threshold, const uint16_t num_classes,
                    const int64_t input_image_height, const int64_t input_image_width,
                    const int64_t batch_ind)
{
  std::vector<std::array<float, 4>> bboxes;
  std::vector<float> scores;
  std::vector<uint64_t> classIndices;
  // cout << num_classes << endl;
  for (int i = 0; i < inferenceOutput.size(); i++)
  {
    // cout << NUM_ANCHORS[i] << endl;
    // cout << inferenceOutput[i].size() << endl;
    this->post_process_feature_map(inferenceOutput[i].data(), confidenceThresh, num_classes,
                                   input_image_height, input_image_width, 32 / pow(2, i),
                                   this->ANCHORS[i], this->NUM_ANCHORS[i], bboxes, scores,
                                   classIndices, batch_ind);
  }

  std::vector<uint64_t> after_nms_indices;

  after_nms_indices = nms(bboxes, scores, nms_threshold);

  std::vector<std::array<float, 4>> after_nms_bboxes;
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

  return std::make_tuple(after_nms_bboxes, after_nms_class_indices, after_nms_scores);
}

void Yolov3::post_process_feature_map(const float *out_feature_map, const float confidenceThresh,
                                      const int num_classes, const int64_t input_image_height,
                                      const int64_t input_image_width, const int factor,
                                      const std::vector<float> &anchors, const int64_t &num_anchors,
                                      std::vector<std::array<float, 4>> &bboxes,
                                      std::vector<float> &scores,
                                      std::vector<uint64_t> &classIndices, const int b)
{

  // TODO: use precomputed values from onnx model
  const int64_t feature_map_height = this->IMG_HEIGHT / factor;
  const int64_t feature_map_width = this->IMG_WIDTH / factor;
  const int64_t feature_map_size = feature_map_width * feature_map_height;
  const int64_t num_filters = (num_classes + 5) * num_anchors;
  const int64_t num_boxes = feature_map_size * num_filters;
  float tmpScores[num_classes];

  // cout << "num_boxes is " << num_boxes << endl;
  // cout << feature_map_height << feature_map_width << endl;

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
      if (probability < confidenceThresh) // TODO check if giving correct result when Nan
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
  if (this->use_letterbox)
  {
    int64_t old_h = input_image_height, old_w = input_image_width;
    int64_t offset_h = 0, offset_w = 0;

    // TODO: This if-block can be precomputed as it is same for each inference
    if (((float)input_image_width / this->IMG_WIDTH) >=
        ((float)input_image_height / this->IMG_HEIGHT))
    {
      old_h = (float)this->IMG_HEIGHT * input_image_width / this->IMG_WIDTH;
      offset_h = (old_h - input_image_height) / 2;
    }
    else
    {
      old_w = (float)this->IMG_WIDTH * input_image_height / this->IMG_HEIGHT;
      offset_w = (old_w - input_image_width) / 2;
    }

    xmin = xt * old_w;
    ymin = yt * old_h;
    xmax = width * old_w;
    ymax = height * old_h;

    xmin -= offset_w;
    ymin -= offset_h;
    xmax += xmin;
    ymax += ymin;

    // Convert coordinates wrt model inputs (needed by create_blobs_from_detections())
    xmin = ((float)xmin / input_image_width) * this->IMG_WIDTH;
    ymin = ((float)ymin / input_image_height) * this->IMG_HEIGHT;
    xmax = ((float)xmax / input_image_width) * this->IMG_WIDTH;
    ymax = ((float)ymax / input_image_height) * this->IMG_HEIGHT;
  }
  else
  {
    xmin = xt * input_image_width;
    ymin = yt * input_image_height;
    xmax = xmin + width * input_image_width;
    ymax = ymin + height * input_image_height;
  }

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
  // assert(bboxes.size() > 0);
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