
import time
import os
import build.run_yolo_onnx

import numpy as np
import cv2
import torch
import ctypes


save_path = '/docker/deepak/yolo_onnx_release/image/'
video_path = "/docker/deepak/PlatformEdgeCrossing.avi"


# model_path = "/docker/deepak/models/modified_best18_06_24.onnx"
# model_path = "/docker/deepak/models/yolov7-tiny.onnx"
# img_path = "/docker/deepak/image/image3.jpg"
model_path = '/docker/models/person_head.tiny_yolov7/v1/onnx/yolov7-tiny.onnx'
img_path = "/docker/deepak/image/person_standing.webp"
# model_path = "/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/piyush.best.416.v3.onnx"


# model_path = "/docker/python_app/best.onnx"

start_time =  time.time()
batch_size = 1
nms_threshold = 0.45
number_of_classes = 2
confidence = 0.3
provider='cpu'

anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]
# anchors = [[10, 13, 16, 30, 33, 23], [116, 90, 156, 198, 373, 326],
#          [30, 61, 62, 45, 59, 119]
#         ]


v7_object = build.run_yolo_onnx.Yolov7(number_of_classes, anchors, model_path, batch_size, provider)

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

height = width = 640
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
    

    ret, full_image = cap.read()
    if ret == False:
        break
    # full_image = cv2.imread(img_path)
   
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

    
    start_batch_time = time.time()
    # preprocessed_img_cpp = v7_object.preprocess_batch(batch_list)   
    preprocessed_img_cpp = preprocess_py(batch_list)  
    print("preprocess time in py file ",(time.time() - start_batch_time)*1000)


    # print(type(preprocessed_img_cpp), preprocessed_img_cpp.shape)
    
    detect_start_time = time.time()   
    inferenced_output = v7_object.detect(preprocessed_img_cpp)
    print("detection time in python file ", (time.time() - detect_start_time )* 1000)
    
        
    post_start_time = time.time()
    list_of_boxes = v7_object.postprocess_batch(inferenced_output, confidence , nms_threshold , full_image.shape[0] , full_image.shape[1])
    print("post time in py file",(time.time() - post_start_time)*1000)
    
    # print(len(list_of_boxes))
    # print(len(batch_list))
    
    # for k in range(batch_size):
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
    #     cv2.imwrite('/docker/deepak/image/v7_output'+str(k)+'.jpg', full_image)
    #     out.write(full_image)
     
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
    # exit()


cap.release()
out.release()
