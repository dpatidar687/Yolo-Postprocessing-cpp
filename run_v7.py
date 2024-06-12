import build.run_yolo_onnx
import time
import os
import cv2
import numpy as np
import time
import os
import build.run_yolo_onnx

import numpy as np
import cv2
import torch


start_time =  time.time()



video_path = "/docker/deepak/PlatformEdgeCrossing.avi"
model_path = "/docker/models/common.tiny_yolov7/v1/onnx/yolov7-tiny.onnx"
img_path = "/docker/deepak/image/Shazim+uddin+pp+image+with+stroke.jpg"
save_path = '/docker/deepak/image/'

start_time =  time.time()
batch_size = 1
nms_threshold = 0.45
number_of_classes = 80
confidence = 0.25
provider='cpu'

anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]

# anchors = [[10, 13, 16, 30, 33, 23],[30, 61, 62, 45, 59, 119],[116, 90, 156, 198, 373, 326]]
# 
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



while True: 
    

    # ret, full_image = cap.read()
    # if ret == False:
    #     break
    full_image = cv2.imread(img_path)
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

    
    start_batch_time = time.time()

    v7_object.preprocess_batch(batch_list)
    preprocessed_img_ptr = v7_object.get_img_ptr()
    print("preprocess time ",(time.time() - start_batch_time)*1000)

    
    # pre = v7_object.get_numpy_array_img()
    # print(len(pre))
        
    detect_start_time = time.time()   
    v7_object.detect(preprocessed_img_ptr)
    feature_map_ptr = v7_object.get_inference_output_ptr()
    # print("feature_map_ptr " , feature_map_ptr)
    
   
    print("detect time ",(time.time() - detect_start_time) *1000)
 
    # infer_output = v7_object.get_numpy_array_inference_output()
    # print(len(infer_output[0]), len(infer_output[1]))
    
    post_start_time = time.time()
    list_of_boxes = v7_object.postprocess_batch(feature_map_ptr, confidence , nms_threshold , full_image.shape[0] , full_image.shape[1])
    print("post time ",(time.time() - post_start_time)*1000)
   
    # for k in range(batch_size):
    #     batch_index = k
    #     full_image = batch_list[k]
        
    #     boxes = list_of_boxes[k][0]
    #     cls = list_of_boxes[k][1]
    #     score = list_of_boxes[k][2]
    #     print(k, len(boxes))
    #     for i in range(len(boxes)) :
    #         x1 = boxes[i][0]
    #         y1 = boxes[i][1]
    #         x2 = boxes[i][2]
    #         y2 = boxes[i][3]
    #         print(x1, y1, x2, y2, cls[i], score[i])
    #         cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
    #     # cv2.imwrite(save_path + 'output.jpg', full_image)
    #     out.write(full_image)
    
     
    end_time_batch = time.time()
    print("overall_time ", (end_time_batch - start_batch_time)*1000)
    print("Batch_FPS ", batch_size/(end_time_batch - start_batch_time))
    print("------------------------------------------------------------------------------------")
    
cap.release()
out.release()


exit()
for j in range(1000):
    
    full_image = cv2.imread(img_path)
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)



    start_batch_time = time.time()   
    v7_object.preprocess_batch(batch_list)
    preprocessed_img_ptr = v7_object.get_img_ptr()

    # print("preprocessed img ptr ",preprocessed_img_ptr)
    print("preprocess time ",(time.time() - start_batch_time)*1000)


    # pre = v7_object.get_numpy_array_img()
    # print(len(pre))
        
    detect_start_time = time.time()   
    v7_object.detect(preprocessed_img_ptr)
    feature_map_ptr = v7_object.get_inference_output_ptr()
    # print("feature_map_ptr " , feature_map_ptr)


    print("detect time ",(time.time() - detect_start_time) *1000)

    infer_output = v7_object.get_numpy_array_inference_output()
    print(len(infer_output[0]), len(infer_output[1]))

    post_start_time = time.time()
    list_of_boxes = v7_object.postprocess_batch(feature_map_ptr, confidence , nms_threshold , number_of_classes, full_image.shape[0] , full_image.shape[1])
    print("post time ",(time.time() - post_start_time)*1000)
    print(len(list_of_boxes))


    for k in range(batch_size):
        # batch_index = k
        # full_image = batch_list[k]
        
        boxes = list_of_boxes[k][0]
        cls = list_of_boxes[k][1]
        score = list_of_boxes[k][2]
        print(k, len(boxes))
        for i in range(len(boxes)) :
            x1 = boxes[i][0]
            y1 = boxes[i][1]
            x2 = boxes[i][2]
            y2 = boxes[i][3]
            print(x1, y1, x2, y2, cls[i], score[i])
            cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
            
            
        cv2.imwrite('/docker/deepak/image/v7_output.jpg', full_image)
    print("entire time ",time.time() - start_time)

        
    end_time_batch = time.time()
    print("overall_time ", (end_time_batch - start_batch_time)*1000)
    print("Batch_FPS ", batch_size/(end_time_batch - start_batch_time))
    print("------------------------------------------------------------------------------------")
    
