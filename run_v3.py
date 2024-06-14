
import time
import os
import build.run_yolo_onnx

import numpy as np
import cv2
import torch
import ctypes

# model_path = "/docker/deepak/models/yolo_tiny_25_07.onnx"
# model_path = "/docker/deepak/models/person_head_tinyv3.onnx"
save_path = '/docker/deepak/yolo_onnx_release/image/'
video_path = "/docker/deepak/PlatformEdgeCrossing.avi"


model_path = "/docker/models/common.tiny_yolov3/v1/onnx/yolov3-tiny.onnx"
img_path = "/docker/deepak/image/image3.jpg"
img_path = "/docker/deepak/image/person_standing.webp"

# video_path = "/docker/deepak/side _camera_office.mp4"


start_time =  time.time()
batch_size = 1
nms_threshold = 0.45
number_of_classes = 80
confidence = 0.25
provider='cpu'

anchors = [[81, 82, 135, 169, 344, 319],
        [10, 14, 23, 27, 37, 58]]


v3_object = build.run_yolo_onnx.Yolov3(number_of_classes, anchors, model_path, batch_size, provider)

# full_image = cv2.imread(image_path)
loop_start_time = time.time()

video = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
output_path = '/docker/deepak/output_video.mp4'  # Update with your output video file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
out = cv2.VideoWriter(output_path, fourcc, 25.0, (frame_width, frame_height))  # Adjust fps if needed

# batch_list = []





while True: 
    
    ret, full_image = cap.read()
    if ret == False:
        break
    # full_image = cv2.imread(img_path)
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

   
    start_batch_time = time.time()
    preprocessed_img = v3_object.preprocess_batch(batch_list)
    print(hex(id(preprocessed_img)))
    print("preprocess time in py file ",(time.time() - start_batch_time)*1000)

    
    detect_start_time = time.time()   
    inferenced_output = v3_object.detect(preprocessed_img)
    print(hex(id(inferenced_output)))
    print("detection time in python file ", (time.time() - detect_start_time )* 1000)
    
        
    post_start_time = time.time()
    list_of_boxes = v3_object.postprocess_batch(inferenced_output, confidence , nms_threshold , full_image.shape[0] , full_image.shape[1])
    print("post time in py file",(time.time() - post_start_time)*1000)
   
    # for k in range(batch_size):
    #     batch_index = k
    #     full_image = batch_list[k]
        
    #     boxes = list_of_boxes[k][0]
    #     cls = list_of_boxes[k][1]
    #     score = list_of_boxes[k][2]
    #     # print(k, len(boxes))
    #     for i in range(len(boxes)) :
    #         x1 = boxes[i][0]
    #         y1 = boxes[i][1]
    #         x2 = boxes[i][2]
    #         y2 = boxes[i][3]
    #         print(x1, y1, x2, y2, cls[i], score[i])
    #         cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
    #     # cv2.imwrite('/docker/deepak/image/v3_output.jpg', full_image)
    #     # out.write(full_image)
     
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
cap.release()
out.release()


