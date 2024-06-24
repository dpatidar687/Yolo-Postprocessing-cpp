
#include <yolov3.h>
#include <yolov7.h>
#include <pybind11/pybind11.h>
#include <base_classifier.h>




PYBIND11_MODULE(run_yolo_onnx, m)
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
    // .def("get_numpy_array_img", &Yolov3::get_numpy_array_img);
    // .def("get_img_ptr", &Yolov3::get_img_ptr)




        py::class_<Yolov7>(m, "Yolov7")
            .def(py::init<int, std::vector<std::vector<float> >, const std::string &,
                          int, std::string>(),
                 py::arg("number_of_classes"),
                 py::arg("anchors"),
                 py::arg("model_path"),
                 py::arg("batch_size"),
                 py::arg("provider"))
            .def("preprocess_batch", &Yolov7::preprocess_batch)
            .def("detect", &Yolov7::detect)
            .def("postprocess_batch", &Yolov7::postprocess_batch, py::return_value_policy::reference);
    
    py::class_<Base_classifier>(m, "Base_classifier")
    .def(py::init<const std::string &, int , const std::string >())
    .def("infer", &Base_classifier::infer);
    


}
