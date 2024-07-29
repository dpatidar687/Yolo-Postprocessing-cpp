
import time
import os
import build.Yolo_Infer_CPP

import numpy as np
import cv2
import torch
import ctypes

model_path = "/docker/models/yolo_tiny_25_07.onnx"


img_path = "/docker/image/61vMB3QmbWL._AC_UF894,1000_QL80_.jpg"


start_time =  time.time()
batch_size = 1
nms_threshold = 0.45
number_of_classes = 1
confidence = 0.01
provider='gpu'

anchors = [[81, 82, 135, 169, 344, 319],
        [10, 14, 23, 27, 37, 58]]



v3_object = build.Yolo_Infer_CPP.Yolov3(number_of_classes, anchors, model_path, batch_size, provider)

while True: 
    
    full_image = cv2.imread(img_path)
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

   
    start_batch_time = time.time()
    preprocessed_img_cpp = v3_object.preprocess_batch(batch_list)   
    print("preprocess time in py file ",(time.time() - start_batch_time)*1000)
    
    # print("preprocessed_img_cpp ", hex(id(preprocessed_img_cpp)), hex(id(preprocessed_img_cpp[0])))




    # print(type(preprocessed_img_cpp))
    
    detect_start_time = time.time()   
    inferenced_output = v3_object.detect(preprocessed_img_cpp)
    print("detection time in python file ", (time.time() - detect_start_time )* 1000)
    
        
    post_start_time = time.time()
    list_of_boxes = v3_object.postprocess_batch(inferenced_output, confidence , nms_threshold , full_image.shape[0] , full_image.shape[1])
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
        # cv2.imwrite('/docker/deepak/image/v3_output'+str(k)+'.jpg', full_image)
     
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
    exit()
cap.release()
out.release()
