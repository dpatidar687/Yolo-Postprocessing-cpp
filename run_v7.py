import build.run_yolo_onnx
import time
import os
import cv2
import numpy as np

model_path = "/workspace/yolo_onnx_release/models/yolov7-tiny.onnx"
save_path = '/workspace/yolo_onnx_release/image/'
image_path = "/workspace/yolo_onnx_release/image/Shazim+uddin+pp+image+with+stroke.jpg"


start_time =  time.time()
batch_size = 1

batch_index = 0
channels = 3
height = 640
width = 640
nms_threshold = 0.45
number_of_classes = 80
confidence = 0.1  
img_size = 640  
anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]


v7object = build.run_yolo_onnx.Yolov7(batch_size,img_size, anchors)

v7object.initialize(model_path, height, width, channels, batch_size)

img = cv2.imread(image_path)
flat_list = v7object.preprocess(img, batch_index)

vectors_of_vectors = v7object.detect(flat_list)


full_image = cv2.imread(image_path)
input_image_height = full_image.shape[0]
input_image_width = full_image.shape[1]

list_of_boxes = v7object.postprocess(vectors_of_vectors, confidence , nms_threshold , number_of_classes, input_image_height , input_image_width , batch_index)


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









