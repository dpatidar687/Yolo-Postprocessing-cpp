


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
batch_size = 4
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

detector = build.run_yolo_onnx.YoloDetectorv3()


v3_object = build.run_yolo_onnx.Yolov3(number_of_classes,img_size, anchors)

detector.initialize(model_path, height, width, channels, batch_size)

lists = []
for i in range(batch_size) :
    flat_list = v3_object.preprocess(image_path, batch_index)
    lists.append(flat_list)


print(len(lists))



vectors_of_vectors = detector.detect(flat_list)

full_image = cv2.imread(image_path)
input_image_height = full_image.shape[0]
input_image_width = full_image.shape[1]

list_of_boxes = v3_object.postprocess(vectors_of_vectors, confidence , nms_threshold , number_of_classes, input_image_height , input_image_width , batch_index)


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
time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)

exit()

