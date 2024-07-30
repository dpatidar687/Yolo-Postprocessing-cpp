#include <yolov3.h>
#include <yolov7.h>
#include <pybind11/pybind11.h>
#include <base_classifier.h>
#include <../models/model_config.h>
#include <yolobase.h>

PYBIND11_MODULE(Yolo_Infer_CPP, m)
{
    py::class_<Yolov3>(m, "Yolov3")
        .def(py::init<int, std::vector<std::vector<float> >, const std::string &,
                      int, std::string>(),
             py::arg("number_of_classes"),
             py::arg("anchors"),
             py::arg("model_path"),
             py::arg("batch_size"),
             py::arg("provider"))
        .def("preprocess_batch", &Yolov3::preprocess_batch, py::return_value_policy::reference)
        .def("detect", &Yolov3::detect, py::return_value_policy::reference)
        .def("postprocess_batch", &Yolov3::postprocess_batch, py::return_value_policy::reference);
  


    py::class_<Yolov7>(m, "Yolov7")
        .def(py::init<int, std::vector<std::vector<float>>, const std::string&,
                      int, const std::string&, bool, std::vector<float>>(),
             py::arg("number_of_classes"),
             py::arg("anchors"),
             py::arg("model_path"),
             py::arg("batch_size"),
             py::arg("provider"),
             py::arg("letter_box"),
             py::arg("letter_box_color"))
        .def("preprocess_batch", &Yolov7::preprocess_batch)
        .def("detect", &Yolov7::detect)
        .def("postprocess_batch", &Yolov7::postprocess_batch, py::return_value_policy::reference);


    
    py::class_<Base_classifier>(m, "Base_classifier")
    .def(py::init<const std::string &, int , const std::string >())
    .def("infer_cpp", &Base_classifier::infer_cpp), py::return_value_policy::reference;


    py::class_<mtx::ModelConfig>(m, "ModelConfig")
        .def(py::init<const std::string&, const std::string&, const int64_t&, const float&>());

    py::class_<Yolobase>(m, "Yolobase")
        .def(py::init<const mtx::ModelConfig&>(), py::arg("config"))
        .def("preprocess_batch", &Yolobase::preprocess_batch, py::return_value_policy::reference)
        .def("infer_cpp", &Yolobase::infer_cpp, py::return_value_policy::reference)
        .def("infer_trt_ort", &Yolobase::infer_trt_ort, py::return_value_policy::reference);
        

}
