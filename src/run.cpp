#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <opencv2/opencv.hpp>
#include <core/providers/cpu/cpu_provider_factory.h>
#include <core/providers/cuda/cuda_provider_options.h>
#include "onnxruntime_cxx_api.h"
#include <yolov3.h>
#include <yolov7.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;



PYBIND11_MODULE(run_yolo_onnx, m)
{
    py::class_<Yolov3>(m, "Yolov3")
        .def(py::init<int, std::vector<std::vector<float>>, const std::string&,  
                 int, std::string>(),
             py::arg("number_of_classes"),
             py::arg("anchors"),
             py::arg("model_path"),
             py::arg("batch_size"),
             py::arg("provider")
             )
        .def("preprocess", &Yolov3::preprocess)
        .def("detect", &Yolov3::detect)
        .def("postprocess", &Yolov3::postprocess)
        .def("get_raw_data", &Yolov3::get_raw_data)
        .def("get_inference_output", &Yolov3::get_inference_output);


    // py::class_<Yolov7>(m, "Yolov7")
        // .def(py::init<int, int, std::vector<std::vector<float>>>())
        // .def("preprocess", &Yolov7::preprocess, py::return_value_policy::reference)
        // .def("initialize", &Yolov7::initialize)
        // .def("detect", &Yolov7:: detect, py::return_value_policy::reference) 
        // .def("postprocess", &Yolov7::postprocess, py::return_value_policy::reference);


    py::class_<ptr_wrapper<float>>(m, "ptr_wrapper");

    py::class_<ptr_wrapper<std::vector<std::vector<float>>>>(m, "ptr_wrapper_inference_output");

}
