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
model = 'yolov7'
anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]
full_image = cv2.imread(image_path)


detector = build.run_yolo_onnx.YoloDetectorv7()
v7_object = build.run_yolo_onnx.Yolov7(number_of_classes, height, anchors)


detector.initialize(model_path, height, width, channels, number_of_classes,batch_size, confidence,  nms_threshold, anchors, model)

detector.initialize(model_path, height, width, channels, number_of_classes,batch_size, confidence,  nms_threshold, anchors, model)

flat_list = v7_object.preprocess(image_path, height, width, channels, 0)


list_of_boxes = detector.detect(flat_list, full_image.shape[0], full_image.shape[1])


boxes = list_of_boxes[0]
cls = list_of_boxes[1]
score = list_of_boxes[2]
full_image = cv2.imread(image_path)

for i in range(len(boxes)) :
    x1 = boxes[i][0]
    y1 = boxes[i][1]
    x2 = boxes[i][2]
    y2 = boxes[i][3]
    print(int(x1), int(y1), int(x2), int(y2), cls[i], score[i])
    cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
cv2.imwrite(save_path + 'output.jpg', full_image)
time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)




exit()
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










