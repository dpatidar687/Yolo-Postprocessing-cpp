


import build.run_yolo_onnx
import time
import os
import numpy as np
import cv2
import torch


model_path = "/workspace/yolo_onnx_release/models/yolo_tiny_25_07.onnx"
save_path = '/workspace/yolo_onnx_release/image/'
image_path = "/workspace/yolo_onnx_release/image/61vMB3QmbWL._AC_UF894,1000_QL80_.jpg"



start_time =  time.time()
batch_size = 1
batch_index = 0
channels = 3
img_size = 416
height = 416
width = 416
nms_threshold = 0.45
number_of_classes = 1
confidence = 0.6


anchors = [[81, 82, 135, 169, 344, 319],
        [10, 14, 23, 27, 37, 58]]

model = 'yolov3'


v3_object = build.run_yolo_onnx.Yolov3(number_of_classes,img_size, anchors)
v3_object.initialize(model_path, height, width, channels, batch_size)
init_time = time.time()
print("time taken in initialization",start_time - init_time)

img = cv2.imread(image_path)

# preprocessing
before_prepocess = time.time()

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (width, height))
# img = img/255
# img = np.transpose(img, (2, 0, 1))
# flat_list = img.flatten().tolist()


v3_object.preprocess(img, batch_index)

flat_list = v3_object.get_raw_data()

# print(id(flat_list[0]))
# print(type(flat_list.data))


# print(type(flat_list))
after_prepocess = time.time()
print("overall preprocess time ", after_prepocess - before_prepocess)

# Inference
before_detect = time.time()
vectors_of_vectors = v3_object.detect(flat_list)

vectors_of_vectors = v3_object.get_inference_output()
after_detect = time.time()
print("overall inference time ", after_detect - before_detect)



full_image = cv2.imread(image_path)
input_image_height = full_image.shape[0]
input_image_width = full_image.shape[1]


# //post processing
before_postprocess = time.time()
list_of_boxes = v3_object.postprocess(vectors_of_vectors, confidence , nms_threshold , number_of_classes, input_image_height , input_image_width , batch_index)
after_postprocess = time.time()
print("overall postprocess time ", after_postprocess - before_postprocess)


time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)


boxes = list_of_boxes[0]
cls = list_of_boxes[1]
score = list_of_boxes[2]

for i in range(len(boxes)) :
    x1 = boxes[i][0]
    y1 = boxes[i][1]
    x2 = boxes[i][2]
    y2 = boxes[i][3]
    print(x1, y1, x2, y2, cls[i], score[i])
    cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
cv2.imwrite(save_path + 'output.jpg', full_image)
time_single = (time.time() - start_time)/batch_size
print("time in single inference in ms ", time_single)
print ("FPS ", 1/time_single)

exit()

