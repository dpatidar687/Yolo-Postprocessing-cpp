import build.run_yolo_onnx
import time

obj = build.run_yolo_onnx.YoloDetector()

model_path = "/home/manish/Documents/older/alexandria/face_detection.tiny_yolov3/v3/onnx/yolo_tiny_25_07.onnx"
obj.initialize(model_path)


start_time =  time.time()
batch_size = 100

for i in range(batch_size):
	a = obj.detect("/home/manish/tt/2.jpg")

time_single = (time.time() - start_time)/batch_size*1000
print("time in single inference in ms ", time_single)
print ("FPS ", 1000/time_single)
