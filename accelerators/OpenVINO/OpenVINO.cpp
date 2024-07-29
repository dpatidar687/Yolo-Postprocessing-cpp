#include "OpenVINO.h"

namespace mtx
{

	OpenVINOInferenceEngine::OpenVINOInferenceEngine(const std::string &model_path, const std::string &provider,
													 const std::vector<std::vector<int64_t>> &inputShapes,
													 const std::vector<std::vector<long int>> &outputShape, pre_processing preprocess) : InferenceEngine(model_path, provider, inputShapes, outputShape, preprocess)
	{

		// Target device : CPU/GPU
		std::string target_device = "CPU";
		std::cout << "TARGET DEVICE " << target_device << std::endl;

		if(provider == "onnx-openvino-gpu")
		{
			target_device = "GPU";
		}

		std::cout << "Converting onnx to xml..." << std::endl;
		std::string command = "ovc " + this->model_path + " --compress_to_fp16 True --output_model /tmp/"; // --input \"[4, 3, 416, 416]\"";
		std::cout << command << std::endl;
		system(command.c_str());
		std::string delimiter = "/";
		char *token;
		std::vector<std::string> creds;
		token = strtok(const_cast<char *>(this->model_path.c_str()), const_cast<char *>(delimiter.c_str()));
		while (token != NULL)
		{
			creds.push_back(token);
			token = strtok(NULL, const_cast<char *>(delimiter.c_str()));
		}
		this->xml_path = "/tmp/" + creds[creds.size() - 1].substr(0, creds[creds.size() - 1].size() - 4) + "xml";
		showAvailableDevices();
		std::cout << this->core.get_versions(target_device) << std::endl;

		/**************** FOR HIGH PERF IN CPU (100% CPU UTILIZATION) CONFIG ***************/
		if(target_device == "CPU"){
		   this->core.set_property(target_device, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
		   this->core.set_property(target_device, ov::hint::num_requests(this->BATCH_SIZE*2));
		}

		this->model = this->core.read_model(this->xml_path);
		this->model->get_parameters()[0]->set_layout("N...");
		// this->BATCH_SIZE /= 4;
		ov::set_batch(this->model, 1);
		std::cout << this->core.get_property(target_device, ov::device::full_name) << std::endl;
		this->ov_model = this->core.compile_model(this->model, target_device);
		getModelDetails(this->ov_model);
		int inf_threads = 0, inf_reqs = 0;
		inf_reqs = this->ov_model.get_property(ov::optimal_number_of_infer_requests);
		if (target_device == "CPU")
			inf_threads = this->ov_model.get_property(ov::inference_num_threads);
		std::cout << "Optimum number of inference requests: " << inf_reqs << " | Optimum number of inference threads: " << inf_threads << std::endl;

		// Creating N inference requests
		for (int i = 0; i < this->BATCH_SIZE; i++)
		{
			this->infer_pool.push_back(this->ov_model.create_infer_request());
			this->async_infer_pool.push_back(this->ov_model.create_infer_request());
			this->input_tensor.push_back(ov::Tensor());
			this->_in_queue.push_back(false);
			this->_async_in_queue.push_back(false);
		}
		// mtx::info("Loaded Model to device");
	}

	OpenVINOInferenceEngine::~OpenVINOInferenceEngine()
	{

		for (auto &output_data : this->tensor_raw_data)
			delete[] output_data.second.data;
		this->tensor_raw_data.clear();
	}

	std::unordered_map<std::string, std::vector<int64_t>> OpenVINOInferenceEngine::get_output_shape()
	{
		std::unordered_map<std::string, std::vector<int64_t>> model_shape;
		for (auto &tensor : this->ov_output_tensors)
		{
			// convert openvino shape to std::vector<float>
			std::vector<int64_t> shape;
			for (auto &dim : tensor.second.shape)
			{
				shape.push_back(dim);
			}
			model_shape[tensor.first] = shape;
		}
		return model_shape;
	}

	void OpenVINOInferenceEngine::showAvailableDevices()
	{
		std::vector<std::string> devices = this->core.get_available_devices();
		std::cout << std::endl;
		std::cout << "Available target devices:";
		for (const auto &device : devices)
		{
			std::cout << "  " << device;
		}
		std::cout << std::endl;
	}

	void OpenVINOInferenceEngine::change_batch_size(int64_t batch_size)
	{
		// this->BATCH_SIZE = batch_size;
		for (int i = this->BATCH_SIZE; i <= batch_size; i++)
		{
			this->infer_pool.push_back(this->ov_model.create_infer_request());
			this->async_infer_pool.push_back(this->ov_model.create_infer_request());
			this->input_tensor.push_back(ov::Tensor());
			this->_in_queue.push_back(false);
			this->_async_in_queue.push_back(false);
		}
		this->BATCH_SIZE = batch_size;
	}

	void OpenVINOInferenceEngine::getModelDetails(ov::CompiledModel &network)
	{
		// std::cout << "model name: " << network.get_friendly_name() << std::endl;

		int out_counter = 0;

		const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
		std::cout << "    inputs" << std::endl;
		for (const ov::Output<const ov::Node> &input : inputs)
		{

			tensor_details tensor;
			ov_tensor_details ov_tensor;
			const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
			std::cout << "        input name: " << name << std::endl;
			tensor.name = name;
			ov_tensor.name = name;

			const ov::element::Type type = input.get_element_type();
			std::cout << "        input type: " << type << std::endl;

			ov::Shape static_shape;
			ov::PartialShape partial_shape = input.get_partial_shape(); // get zero output partial shape
			if (partial_shape.is_dynamic())
			{
				static_shape.push_back(1); // this->BATCH_SIZE;
				std::cout << "Still dynamic input shape..." << std::endl;
				for (auto dim_ : this->input_shapes[out_counter])
				{
					static_shape.push_back(dim_);
				}
			}
			else
			{
				std::cout << "Converted to static shape..." << std::endl;
				static_shape = partial_shape.get_shape();
			}

			static_shape = partial_shape.get_shape();

			for (int i = 0; i < static_shape.size(); i++)
			{
				tensor.shape.push_back(static_shape[i]);
				tensor.element_count *= static_shape[i];
				ov_tensor.element_count *= static_shape[i];
			}
			std::cout << "        input shape: " << static_shape << std::endl;
			ov_tensor.shape = static_shape;
			this->input_tensors[name] = tensor;
			this->ov_input_tensors[name] = ov_tensor;
			out_counter++;
		}

		std::cout << std::endl;
		const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
		std::cout << "    outputs" << std::endl;

		out_counter = 0;

		for (const ov::Output<const ov::Node> &output : outputs)
		{
			tensor_details tensor;
			ov_tensor_details ov_tensor;
			const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
			std::cout << "        output name: " << name << std::endl;
			tensor.name = name;
			ov_tensor.name = name;

			const ov::element::Type type = output.get_element_type();
			std::cout << "        output type: " << type << std::endl;

			ov::Shape static_shape;
			ov::PartialShape partial_shape = output.get_partial_shape(); // get zero output partial shape
			if (partial_shape.is_dynamic())
			{
				std::cout << "Still dynamic output shape..." << std::endl;
				// static_shape.push_back(1); // this->BATCH_SIZE;
				for (auto dim_ : this->output_shapes[out_counter])
				{
					static_shape.push_back(dim_);
				}
			}
			else
			{
				std::cout << "Converted to static shape..." << std::endl;
				static_shape = partial_shape.get_shape();
			}

			for (int i = 0; i < static_shape.size(); i++)
			{
				tensor.shape.push_back(static_shape[i]);
				tensor.element_count *= static_shape[i];
				ov_tensor.element_count *= static_shape[i];
			}
			std::cout << "        output shape: " << static_shape << std::endl;
			ov_tensor.shape = static_shape;
			this->output_tensors[name] = tensor;
			this->ov_output_tensors[name] = ov_tensor;
			std::cout << std::endl;
			tensor_details tensor_data;
			tensor_data.name = name;
			tensor_data.data = new float[this->BATCH_SIZE * ov_tensor.element_count];
			tensor_data.shape = tensor.shape;
			tensor_data.element_count = tensor.element_count;
			this->tensor_raw_data[name] = tensor_data;
			memset(tensor_data.data, 0, this->BATCH_SIZE * ov_tensor.element_count);
			out_counter++;
		}
	}

	inline void OpenVINOInferenceEngine::convert_input_data_to_ov_tensor(float *src, int index)
	{

		this->input_tensor[index] = ov::Tensor(ov::element::f32, this->ov_input_tensors.begin()->second.shape, src);
	}

	void OpenVINOInferenceEngine::enqueue(float *src)
	{
		int64_t elem_count = this->ov_input_tensors.begin()->second.element_count;
		// #pragma omp parallel for

		// std::cout << "Batch size: " << this->BATCH_SIZE << std::endl;
		// std::cout << "Curr batch size: " << this->CURR_BATCH_SIZE << std::endl;

		for (int i = 0; i < this->BATCH_SIZE; i++)
		{
			if (i >= this->CURR_BATCH_SIZE)
			{
				this->_in_queue[i] = false;
				continue;
			}
			this->convert_input_data_to_ov_tensor(src + (i * elem_count), i);
			this->infer_pool[i].set_input_tensor(this->input_tensor[i]);
			this->infer_pool[i].start_async();
			this->_in_queue[i] = true;
		}

		// std::cout << "Length of infer_pool: " << this->infer_pool.size() << std::endl;
	}

	std::vector<tensor_details> OpenVINOInferenceEngine::execute_network()
	{
		std::vector<tensor_details> output_tensors;
		for (int i = 0; i < this->BATCH_SIZE; i++)
		{
			// if (_in_queue[i])
			// {
				this->infer_pool[i].wait();
				for (auto &tensor_name : this->ov_output_tensors)
				{
					// std::cout << "Output name: " << tensor_name.first << std::endl;
					ov::Tensor output = infer_pool[i].get_tensor(tensor_name.first);
					auto *raw_tensor = this->tensor_raw_data[tensor_name.first].data;
					auto *tensor_data = const_cast<float *>(output.data<const float>());
					memcpy(raw_tensor + i * tensor_name.second.element_count, tensor_data, tensor_name.second.element_count * sizeof(float));
					// output_tensors.push_back(this->tensor_raw_data[tensor_name.first]);
				}
			// 	_in_queue[i] = false;
			// }
		}

		for (auto &tensor_name : this->ov_output_tensors)
		{
			// std::cout << "Output name: " << tensor_name.first << std::endl;
			// std::cout << "batch size: " << this->BATCH_SIZE << std::endl;
			// std::cout << "current batch size: " << this->CURR_BATCH_SIZE << std::endl;

			auto opt_tensor = this->tensor_raw_data[tensor_name.first];
			opt_tensor.shape[0] = this->BATCH_SIZE;
			opt_tensor.element_count *= this->BATCH_SIZE;
			output_tensors.push_back(opt_tensor);
		}

// 		for (auto &tensor_name : this->ov_output_tensors)
// {
//     std::cout << "Output name: " << tensor_name.first << std::endl;
//     for (int i = 0; i < this->BATCH_SIZE; i++)
//     {
//         if (_in_queue[i])
//         {
//             this->infer_pool[i].wait();
//             ov::Tensor output = infer_pool[i].get_tensor(tensor_name.first);
//             auto *raw_tensor = this->tensor_raw_data[tensor_name.first].data;
//             auto *tensor_data = const_cast<float *>(output.data<const float>());
//             memcpy(raw_tensor + i * tensor_name.second.element_count, tensor_data, tensor_name.second.element_count * sizeof(float));
            
//             _in_queue[i] = false;
//         }
// 		output_tensors.push_back(this->tensor_raw_data[tensor_name.first]);
//     }
// }


		// malloc_trim(0);
		return output_tensors;
	}

	std::vector<tensor_details> OpenVINOInferenceEngine::execute_async_network()
	{
		std::vector<tensor_details> output_tensors;
		for (int i = 0; i < this->BATCH_SIZE; i++)
		{

			if (_async_in_queue[i])
			{

				this->async_infer_pool[i].wait();

				for (auto &tensor_name : this->ov_output_tensors)
				{
					ov::Tensor output = async_infer_pool[i].get_tensor(tensor_name.first);
					auto *raw_tensor = this->tensor_raw_data[tensor_name.first].data;
					auto *tensor_data = const_cast<float *>(output.data<const float>());

					memcpy(raw_tensor + i * tensor_name.second.element_count, tensor_data, tensor_name.second.element_count * sizeof(float));
					output_tensors.push_back(this->tensor_raw_data[tensor_name.first]);
				}
				if (this->_in_queue[i])
				{
					// move the busy inf request to async pool
					std::swap(async_infer_pool[i], infer_pool[i]);
					_async_in_queue[i] = true;

					_in_queue[i] = false;
				}
				else
					_async_in_queue[i] = false;
			}
			else
			{
				if (this->_in_queue[i])
				{
					// move the busy inf request to async pool
					std::swap(async_infer_pool[i], infer_pool[i]);
					_async_in_queue[i] = true;
					_in_queue[i] = false;
				}
				else
				{
					_async_in_queue[i] = false;
				}
			}
		}

		// malloc_trim
		return output_tensors;
	}

}
