


import build.run_yolo_onnx
import time
import os
import numpy as np
import cv2
import torch
obj = build.run_yolo_onnx.YoloDetectorv3()

model_path = "/workspace/yolo_onnx_release/models/yolo_tiny_25_07.onnx"
save_path = '/workspace/yolo_onnx_release/image/'
image_path = "/workspace/yolo_onnx_release/image/61vMB3QmbWL._AC_UF894,1000_QL80_.jpg"



start_time =  time.time()
batch_size = 1

batch_size = 1
batch_index = 0
channels = 3
height = 416
width = 416
nms_threshold = 0.45
number_of_classes = 1
confidence = 0.6


anchors = [[81, 82, 135, 169, 344, 319],
        [10, 14, 23, 27, 37, 58]]



obj.initialize(model_path, height, width, channels, number_of_classes,batch_size, confidence,  nms_threshold, anchors)

def preporcessing(image):
    full_image = cv2.imread(image_path)
    image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = np.transpose(image, (2, 0, 1))
    image = image/255.0
    return image, full_image.shape[0], full_image.shape[1]



# save_path = os.path.dirname(image_path)+'/output/'
# print(save_path)

for i in range(batch_size):  
    img, h, w = preporcessing(image_path)
    flat_list = img.flatten().tolist()
    print(h , 'h, w in python ', w )
    a = obj.detect( flat_list, h, w)
    print(a)
    
    boxes = a[0]
    cls = a[1]
    score = a[2]
    full_image = cv2.imread(image_path)
    for i in range(len(boxes)) :
        # print(boxes[i], cls[i], score[i])
        
        x1 = boxes[i][0]
        y1 = boxes[i][1]
        x2 = boxes[i][2]
        y2 = boxes[i][3]
        cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)

    cv2.imwrite(save_path + 'output.jpg', full_image)
    
 
time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)








    