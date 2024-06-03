#include <iostream>
using namespace std;
#include <iostream>
#include <vector>

// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
// #include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <yolov7.h>
// #include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"
// #include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include <stdio.h>
#include <iostream>
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

Yolov7::Yolov7(int numClasses, int image_size, std::vector<std::vector<float>> anchors)
{
  this->IMG_HEIGHT = image_size;
  this->IMG_WIDTH = image_size;
  this->IMG_CHANNEL = 3;
  this->ANCHORS = anchors;

  for (auto &anchor : this->ANCHORS)

    this->NUM_ANCHORS.push_back(anchor.size() / 2); // divide by 2 as it has width and height scales

  this->dst = new float[this->BATCH_SIZE * this->IMG_CHANNEL * this->IMG_WIDTH *
                        this->IMG_HEIGHT]; // avoid allocating heap memory at every iteration
}
std::vector<float>  Yolov7::preprocess(std::string img_path, size_t batch_index)
  {
    const cv::Mat &img = cv::imread(img_path);
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
  std::cout << "using a preprocess of yolov7 class" << std::endl;
  return flat_list;
}
float Yolov7::sigmoid(float x) const
{
  return 1.0 / (1.0 + std::exp(-x));
}

std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>>Yolov7::postprocess(const std::vector<std::vector<float>> &inferenceOutput,
    const float confidenceThresh, const float nms_threshold, const uint16_t num_classes,
    const int64_t input_image_height, const int64_t input_image_width,
    const int64_t batch_ind)
{

  std::cout << "using a postprocess of yolov7 class" << std::endl;          
  std::vector<std::array<float, 4>> bboxes;
  std::vector<float> scores;
  std::vector<uint64_t> classIndices;

  if ((inferenceOutput.size() > 1))
  {

    for (int i = 0; i < inferenceOutput.size(); i++)
    {
      
      this->post_process_feature_map(inferenceOutput[i].data(), confidenceThresh, num_classes,
                                     input_image_height, input_image_width, 32 / pow(2, i),
                                     this->ANCHORS[i], this->NUM_ANCHORS[i], bboxes, scores,
                                     classIndices, batch_ind);
                      
    }
  }
  else
  {
    std::cout << "Output is not in three scales " << std::endl;
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

void Yolov7::post_process_feature_map(const float *out_feature_map, const float confidenceThresh,
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

      if (probability < confidenceThresh) // TODO check if giving correct result when Nan
        continue;

      float xcenter = outputData[i + feature_map_size * ((num_classes + 5) * j)];
      float ycenter = outputData[i + feature_map_size * ((num_classes + 5) * j + 1)];
      float width = outputData[i + feature_map_size * ((num_classes + 5) * j + 2)];
      float height = outputData[i + feature_map_size * ((num_classes + 5) * j + 3)];

      xcenter = (sigmoid(xcenter) * 2 - 0.5 + (i % feature_map_width)) / feature_map_width;   // [0-1]
      ycenter = (sigmoid(ycenter) * 2 - 0.5 + (i / feature_map_height)) / feature_map_height; // [0-1]

      width = pow(sigmoid(width) * 2, 2) * anchors[2 * j] / this->IMG_WIDTH;
      height = pow(sigmoid(height) * 2, 2) * anchors[2 * j + 1] / this->IMG_HEIGHT;

      float xt = xcenter - (width / 2);
      float yt = ycenter - (height / 2);

      std::array<float, 4> out_box =
          post_process_box(xt, yt, width, height, input_image_height, input_image_width);

      bboxes.emplace_back(out_box);
      scores.emplace_back(probability);
      classIndices.emplace_back(maxIdx);
    }
  }
}

std::array<float, 4> Yolov7::post_process_box(const float &xt, const float &yt, const float &width,
                                              const float &height,
                                              const int64_t &input_image_height,
                                              const int64_t &input_image_width) const
{

  float xmin, xmax, ymin, ymax;

  xmin = xt * input_image_width;
  ymin = yt * input_image_height;
  xmax = xmin + (width * input_image_width);
  ymax = ymin + (height * input_image_height);


  xmin = std::max<float>(xmin, 0.0);
  ymin = std::max<float>(ymin, 0.0);
  xmax = std::min<float>(xmax, input_image_width - 1);
  ymax = std::min<float>(ymax, input_image_height - 1);

  return std::array<float, 4>{xmin, ymin, xmax, ymax};
}

std::vector<uint64_t> Yolov7::nms(const std::vector<std::array<float, 4>> &bboxes,
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
