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
// #include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"
// #include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include <stdio.h>
#include <iostream>

class Yolov3
{

private:
    int IMG_WIDTH;
    int IMG_HEIGHT;
    int BATCH_SIZE = 1;
    int IMG_CHANNEL;
    float *dst;
    std::vector<std::vector<float>> ANCHORS;
    std::vector<int64_t> NUM_ANCHORS;
    bool use_letterbox = false; // whether to use letterbox while preprocessing and postproceesing

public:
    Yolov3( int batch_size, int image_size, std::vector<std::vector<float>> anchors);

    //   void preprocess(cv::Mat& img, float* data);
    std::vector<float>  preprocess(std::string img_path, size_t batch_index);
    
    float sigmoid(float x) const;

std::tuple<std::vector<std::array<float, 4>>, std::vector<uint64_t>, std::vector<float>>
            postprocess(const std::vector<std::vector<float>> &inferenceOutput,
                const float confidenceThresh,const float nms_threshold, const uint16_t num_classes,
                const int64_t input_image_height, const int64_t input_image_width,
                const int64_t batch_ind);

    
    void post_process_feature_map(const float *out_feature_map, const float confidenceThresh,
                                const int num_classes, const int64_t input_image_height,
                                const int64_t input_image_width, const int factor,
                                const std::vector<float> &anchors, const int64_t &num_anchors,
                                std::vector<std::array<float, 4>> &bboxes,
                                std::vector<float> &scores,
                                std::vector<uint64_t> &classIndices, const int b);

    std::array<float, 4> post_process_box(const float &xt, const float &yt, const float &width,
                                          const float &height,
                                          const int64_t &input_image_height,
                                          const int64_t &input_image_width) const;

    std::vector<uint64_t> nms(const std::vector<std::array<float, 4>> &bboxes,           
                              const std::vector<float> &scores,                          
                              const float overlapThresh = 0.45,                          
                               uint64_t topK = std::numeric_limits<uint64_t>::max() );
};