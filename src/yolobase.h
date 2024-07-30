using namespace std;
#include <vector>
#include <chrono>
#include <ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <random>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <stdio.h>
#include <iostream>

#include <../models/model_config.h>
#include <../accelerators/TensorRT/TensorRT.h>
#include <../accelerators/ORT/ORT.h>
#include <../accelerators/OpenVINO/OpenVINO.h>
#include <../models/InferenceEngine.h>

#include <core/providers/cuda/cuda_provider_options.h>
#include <core/providers/cpu/cpu_provider_factory.h>
#include "onnxruntime_cxx_api.h"
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/tensorrt/tensorrt_provider_factory.h>
#include <onnxruntime_c_api.h>

class Yolobase
{
private:
  int IMG_WIDTH;
  int IMG_HEIGHT;
  int BATCH_SIZE = 1;
  int IMG_CHANNEL;

  std::vector<float> MEAN_VALUE = {0.0, 0.0, 0.0};
  std::vector<float> SCALE_FACTOR = {255.0, 255.0, 255.0};
  bool USE_LETTERBOX = true;
  std::array<float, 3> PADDING_COLOR = {0.0, 0.0, 0.0};
  std::vector<float> NORM = {255.0, 255.0, 255.0};

  MODEL_COLOR_FORMAT IMG_MODE = MODEL_COLOR_FORMAT::RGB;
  NETWORK_INPUT_ORDER IMG_ORDER = NETWORK_INPUT_ORDER::NCHW;

  float *dst;
  pre_processing PREPROCESS;
  std::unordered_map<std::string, int64_t> input_shape_info;

  float conf;          /**< Minimum Confidence Threshold for Object Detection. */
  float nms_threshold; /**< Non-Max Suppression Threshold.*/
  int softnms, classwise_nms;
  std::unordered_map<std::string, float> classwise_nms_threshold;
  std::string provider;                   /**< Model execution mode, provided by the UI. */
  bool draw_blobs_on_frames;              /**< Whether or not draw all the blobs and maintain a annotated frame in blobs, provided by the UI. */
  std::optional<size_t> gpu_idx;          /**< GPU ID. */
  std::vector<std::string> classes;       /**< Vector of strings specifying the names of detection classes. */
  std::string model_path;                 /**< Path to Model files. */
  uint16_t num_classes;                   /**< Number of detection classes. */
  std::vector<int64_t> model_input_shape; /**< Input Shape Information. */
  std::vector<std::pair<std::string, std::vector<int64_t>>> output_shape_info;
  std::vector<std::vector<float>> input_anchors;                          /**< Vector of vectors of floats containing YOLOV3 anchor boxes. */
  int64_t batch_size;                                                     /**< Batch Size from config. */
  std::unordered_map<std::string, std::vector<float>> named_anchor_boxes; /**< Named Anchor Boxes. */
  std::unique_ptr<mtx::InferenceEngine> yolo_infer;                       /**< Inference Engine Object. */
  bool async;
  pre_processing PREPROCESS_INFO;
  bool isSecondary = false;

public:
  // Yolobase() = default;
  ~Yolobase()
  {
    std::cout << "destructor called of Yolobase class " << std::endl;
    // delete[] this->dst;
  }
  Yolobase(const mtx::ModelConfig &config);

  py::array preprocess_batch(py::list &batch);
  inline void preprocess(const unsigned char *src, const int64_t b);
  cv::Mat create_letterbox(const cv::Mat &frame) const;
  cv::Mat numpyArrayToMat(py::array_t<uchar> arr);
  float sigmoid(float x) const;

  py::dict infer_cpp(py::array &input_array);
  py::dict infer_trt_ort(py::array &input_array);
};
