import build.run_yolo_onnx
import time
import os
import cv2
import numpy as np
obj = build.run_yolo_onnx.YoloDetectorv7()



model_path = "/workspace/yolo_onnx_release/models/best.onnx"
save_path = '/workspace/yolo_onnx_release/image/'
image_path = "/workspace/yolo_onnx_release/image/Shazim+uddin+pp+image+with+stroke.jpg"


start_time =  time.time()
batch_size = 1

batch_index = 0
channels = 3
height = 640
width = 640
nms_threshold = 0.45
number_of_classes = 2
confidence = 0.1    
anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]

# anchors = [[10, 13, 16, 30, 33, 23],  [30, 61, 62, 45, 59, 119],[116, 90, 156, 198, 373, 326]]


obj.initialize(model_path, height, width, channels, number_of_classes,batch_size, confidence,  nms_threshold, anchors)
def preporcessing(image):
    full_image = cv2.imread(image_path)
    # full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(full_image, (height, width))
    image = np.transpose(image, (2, 0, 1))
    image = image/255.0
    return image, full_image.shape[0], full_image.shape[1]



for i in range(batch_size):  
    img, h, w = preporcessing(image_path)
    flat_list = img.flatten().tolist()
    
    a = obj.detect( flat_list, h, w)
    
    
    boxes = a[0]
    cls = a[1]
    score = a[2]
    full_image = cv2.imread(image_path)
    for i in range(len(boxes)) :
        print(boxes[i], cls[i], score[i])
        
        x1 = boxes[i][0]
        y1 = boxes[i][1]
        x2 = boxes[i][2]
        y2 = boxes[i][3]
        cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 2)

    cv2.imwrite(save_path + 'output.jpg', full_image)
    
 
time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)










