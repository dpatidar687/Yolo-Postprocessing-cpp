
import time
import os
import build.run_yolo_onnx

import numpy as np
import cv2
import torch
import ctypes

if __name__ == '__main__':

    model_path = "/docker/deepak/models/yolo_tiny_25_07.onnx"
    img_folder_path = './pravar_data/input_images/'

    batch_size = 1
    nms_threshold = 0.6
    number_of_classes = 1
    confidence = 0.1
    provider='cpu'

    anchors = [[81, 82, 135, 169, 344, 319],
            [10, 14, 23, 27, 37, 58]]

    v3_object = build.run_yolo_onnx.Yolov3(number_of_classes, anchors, model_path, batch_size, provider)

    for file in os.listdir(img_folder_path):
        print(file)
        full_image = cv2.imread(os.path.join(img_folder_path, file))
        batch_list = []
        for i in range(batch_size):
            batch_list.append(full_image)

        preprocessed_img_cpp = v3_object.preprocess_batch(batch_list)   
        inferenced_output = v3_object.detect(preprocessed_img_cpp)    
        print(inferenced_output)
        list_of_boxes = v3_object.postprocess_batch(inferenced_output, confidence , nms_threshold , full_image.shape[0] , full_image.shape[1])
        
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
                # print(x1, y1, x2, y2, cls[i], score[i])
                print(int(x1), int(y1), int(x2-x1), int(y2-y1))
            #     cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
            # cv2.imwrite('/docker/deepak/image/v3_output'+str(k)+'.jpg', full_image)
            
