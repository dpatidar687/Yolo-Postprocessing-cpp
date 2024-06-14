
#include <yolov3.h>
#include <yolov7.h>
#include <pybind11/pybind11.h>

class Check
{
public:
    std::string name;

    Check(const std::string &name)
    {
        this->name = name;
        std::cout << this->name << std::endl;
    }

    float *getString()
    {
        std::vector<float> v = {1.0, 2.0, 3.0};
        float *x = v.data();
        return x;
    }
};

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
            .def("postprocess_batch", &Yolov7::postprocess_batch, py::return_value_policy::reference)
            .def("get_img_ptr", &Yolov7::get_img_ptr)
            .def("get_inference_output_ptr", &Yolov7::get_inference_output_ptr)
            // .def("get_raw_img", &Yolov7::get_raw_img)

            // .def("get_numpy_array_img", [](Yolov7 &self)
            //      { return py::array_t<float>(
            //            {self.get_size_img()}, // shape
            //            {sizeof(float)},       // stride
            //            self.get_raw_img(),    // the pointer to the data
            //            py::cast(&self)        // reference to keep it alive
            //        ); })

            .def("get_numpy_array_inference_output", [](Yolov7 &self)
                 {
                     return self.inference_output; // reference to keep it alive
                 });

    py::class_<ptr_wrapper<float> >(m, "ptr_wrapper")
        .def("get", &ptr_wrapper<float>::get);

    py::class_<ptr_wrapper<std::vector<std::vector<float> > > >(m, "ptr_wrapper_inference_output");
    py::class_<v7_ptr_wrapper<float> >(m, "v7_ptr_wrapper");
    py::class_<v7_ptr_wrapper<std::vector<std::vector<float> > > >(m, "v7_ptr_wrapper_inference_output");

    py::class_<Check>(m, "Check")
        .def(py::init<std::string>())
        .def("getString", &Check::getString);
}
