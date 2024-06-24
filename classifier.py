
import time
import os
import build.run_yolo_onnx

import numpy as np
import cv2
import torch
import ctypes


video_path = "/docker/videos/Sarita_P4L1_04.Jun.2024.mp4"
img_path = "/docker/deepak/image/person_standing.webp"
model_path = "/docker/models/gender_classifier.efficientnetb1/v1/onnx/weightsb1_may4_l1l2sgd-0.75.onnx"


start_time =  time.time()
batch_size = 1
provider='cpu'



classifier = build.run_yolo_onnx.Base_classifier( model_path, batch_size, provider)

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

height = 224
width = 112 
def preprocess_py(batch_list):
    img_array = []
    preprocessed_img = np.empty(())
    for i in range(batch_size):
        img = batch_list[i]
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
    # preprocessed_img_cpp = v3_object.preprocess_batch(batch_list)   
    preprocessed_img_cpp = preprocess_py(batch_list)  
    print("preprocess time in py file ",(time.time() - start_batch_time)*1000)
    
   
    
    detect_start_time = time.time()   
    inferenced_output = classifier.infer(preprocessed_img_cpp)
    print("detection time in python file ", (time.time() - detect_start_time )* 1000)
        
    print(len(inferenced_output))
    print(inferenced_output[0])
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
cap.release()
out.release()
