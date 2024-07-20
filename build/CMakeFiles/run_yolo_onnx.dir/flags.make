# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# compile CXX with /usr/bin/c++
CXX_DEFINES = -DENABLE_CUDA -DENABLE_TRT -DENABLE_VINO -DIE_THREAD=IE_THREAD_TBB -DOV_THREAD=OV_THREAD_TBB -DTBB_PREVIEW_WAITING_FOR_WORKERS=1 -Drun_yolo_onnx_EXPORTS

CXX_INCLUDES = -I/usr/include/python3.8 -I/docker/deepak/Yolo-Postprocessing-cpp/src -I/docker/deepak/Yolo-Postprocessing-cpp/models -I/docker/deepak/Yolo-Postprocessing-cpp/accelerators/OpenVINO -I/docker/deepak/Yolo-Postprocessing-cpp/accelerators/TensorRT -I/docker/deepak/Yolo-Postprocessing-cpp/accelerators/ORT -I/docker/deepak/Yolo-Postprocessing-cpp/run_yolo_onnx -I/usr/local/include/onnxruntime -I/docker/deepak/Yolo-Postprocessing-cpp/ext/spdlog/include -isystem /usr/local/include/opencv4 -isystem /docker/deepak/Yolo-Postprocessing-cpp/PUBLIC -isystem /usr/local/include/onnxruntime/core/session -isystem /usr/local/include/onnxruntime/core/providers/cuda -isystem /opt/intel/openvino/runtime/include -isystem /usr/local/cuda/include -isystem /opt/intel/openvino/runtime/include/ie

CXX_FLAGS = -O3 -DNDEBUG -std=gnu++17 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects -Wno-error=deprecated-declarations

