import build.run_yolo_onnx
import time
import os
import cv2
import numpy as np
obj = build.run_yolo_onnx.YoloDetectorv7()



model_path = "/workspace/yolov7/yolov7-tiny.onnx"
save_path = '/workspace/yolo_onnx_release/image/'
image_path = "/workspace/yolo_onnx_release/image/image3.jpg"








start_time =  time.time()
batch_size = 1

batch_size = 1
batch_index = 0
channels = 3
height = 640
width = 640
nms_threshold = 0.45
number_of_classes = 80
confidence = 0.6


obj.initialize(model_path, height, width, channels, number_of_classes,batch_size, confidence,  nms_threshold)

# save_path = os.path.dirname(image_path)+'/output/'
# print(save_path)


for i in range(batch_size):
    a = obj.detect(image_path)

	# print(a)
 
time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)

exit()










import build.run_yolo_onnx
import time
import os
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


obj.initialize(model_path, height, width, channels, number_of_classes,batch_size, confidence,  nms_threshold)

# save_path = os.path.dirname(image_path)+'/output/'
# print(save_path)

for i in range(batch_size):
	a = obj.detect(image_path)

	print(a)
 
time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)






