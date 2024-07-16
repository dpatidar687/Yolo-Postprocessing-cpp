
import time
import os
import build.run_yolo_onnx
import json
import numpy as np
import cv2
import torch
import ctypes

def get_preprocessing( model_info):
    preprocessing_args = {}
    for input_data in model_info["input_data"]["layers"]:
        img_size = input_data["shape"]
        channel_order = "BGR"
        letterbox = False
        if "channel_order" in input_data["attribs"]:
            if input_data["channel_order"] == "RGB":
                channel_order = "RGB"
        if "letterbox" in input_data["attribs"]:
            if input_data["letterbox"]:
                letterbox = True
        mean = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
        norm = 255
        if "preprocesses" in input_data.keys():
            norm = np.array(input_data["preprocesses"]["norm"])
            mean = np.array(input_data["preprocesses"]["mean"])
            std = np.array(input_data["preprocesses"]["std"])
        
        preprocessing_args.update({input_data["name"]: {"channel_order": channel_order, "img_size": img_size, "norm": norm, "mean": mean, "std": std, "letterbox": letterbox}})
    return preprocessing_args
def get_postprocessing(model_info):
    post_processing_args = {}
    conf = model_info["model_config"]["conf"]
    labels = model_info["output_data"]["labels"]
    for i, layer in enumerate(model_info["output_data"]["layers"]):
        # stride = self.preprocessing_args[self.input_name[0]]["img_size"][2]/layer["shape"][2]
        scale = layer["shape"][2]
        # grid = np.array(np.meshgrid(np.arange(0, scale , 1), np.arange(0, scale , 1))).transpose(1, 2, 0).reshape(1, 1, scale, scale, 2)
        # grid = grid.astype(np.float32)
        anchor = np.array(layer["anchors"])

        nms = model_info["output_data"]["nms"]
        post_processing_args[layer["name"]] = {"anchors": anchor, "nms": nms, "conf": conf, "labels": labels}
        
    
    return post_processing_args



model_info = json.load(open("/docker/deepak/Yolo-Postprocessing-cpp/spec.json", "r"))
model_info.update({"model_config": {"conf": 0.46}})
preprocessing_args = get_preprocessing(model_info)
post_processing_args = get_postprocessing(model_info)

# print(preprocessing_args)
# print(post_processing_args)

yc = build.run_yolo_onnx.yolobase(
    build.run_yolo_onnx.ModelConfig('/docker/deepak/Yolo-Postprocessing-cpp/spec.json', 'onnx-gpu',
                                     2, 0.3, False,'no', {}  )
)
# mc = build.run_yolo_onnx.ModelConfig('/docker/deepak/Yolo-Postprocessing-cpp/spec.json', 'onnx-gpu',
#                                      2, 0.3, False,'no', {}  )


save_path = '/docker/deepak/yolo_onnx_release/image/'
# video_path = "/docker/deepak/PlatformEdgeCrossing.avi"
img_path = "/docker/deepak/image/person_standing.webp"


start_time =  time.time()
batch_size = 1



# anchors = {}
anchors = []
for i in reversed(post_processing_args):
    # anchors[i] =  list(post_processing_args[i]['anchors'])
    anchors.append(list(post_processing_args[i]['anchors']))
    nms = post_processing_args[i]['nms']
    number_of_classes = len(post_processing_args[i]['labels'])
    confidence = post_processing_args[i]['conf']

provider='cpu'

# anchors = [[116, 90, 156, 198, 373, 326],
#          [30, 61, 62, 45, 59, 119],
#         [10, 13, 16, 30, 33, 23]]

model_path = '/docker/models/'+ model_info["onnx_path"]
letter_box_color = [114, 114, 114]
letter_box = True
# std::unordered_map<int, float> classwise_nms_thresh;



v7_object = build.run_yolo_onnx.Yolov7(number_of_classes, anchors, model_path, batch_size, provider, letter_box, letter_box_color)

full_image = cv2.imread(img_path)


height = width = 416
def preprocess_py(batch_list):
    img_array = []
    preprocessed_img = np.empty(())
    for i in batch_list:
        img =i
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (height, width), interpolation = cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = img/255.0
        img_array.append(img)
        
    preprocessed_img = np.array(img_array)
    preprocessed_img = preprocessed_img.reshape(batch_size*height*width*3)
    return preprocessed_img



while True: 
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

    
    start_batch_time = time.time()
    preprocessed_img_cpp = v7_object.preprocess_batch(batch_list)  
    print("preprocess time in py file ",(time.time() - start_batch_time)*1000)

    
    detect_start_time = time.time()   
    inferenced_output = v7_object.detect(preprocessed_img_cpp)
    print("detection time in python file ", (time.time() - detect_start_time )* 1000)
    
        
    post_start_time = time.time()
    list_of_boxes = v7_object.postprocess_batch(inferenced_output, confidence , nms_threshold , full_image.shape[0] , full_image.shape[1])
    print("post time in py file",(time.time() - post_start_time)*1000)
    
    print(len(list_of_boxes))
    print(len(batch_list))
    
    for k in range(batch_size):
        full_image = batch_list[k]
        
        boxes = list_of_boxes[k][0]
        cls = list_of_boxes[k][1]
        score = list_of_boxes[k][2]
        # print(k, len(boxes))
        for i in range(len(boxes)) :
            x1 = boxes[i][0]
            y1 = boxes[i][1]
            x2 = boxes[i][2]
            y2 = boxes[i][3]
            print(x1, y1, x2, y2, cls[i], score[i])
            cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
        cv2.imwrite('/docker/deepak/image/v7_output'+str(k)+'.jpg', full_image)
        out.write(full_image)
     
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
    exit()


cap.release()
out.release()
