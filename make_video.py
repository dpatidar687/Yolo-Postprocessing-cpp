
import time
import build.Yolo_Infer_CPP
import numpy as np
import cv2

import build.Yolo_Infer_CPP


model_path = "/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/piyush.best.416.v7.onnx"
letter_box = True
letter_box_color = [114, 114, 114]

provider='cpu'

anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]

img_path = "/docker/image/2_jpg.rf.99922942d0a3d839f3d2cba6fb3716bf.jpg"
full_image = cv2.imread(img_path)
img_path2 = "/docker/image/openvino2.jpg"
full_image2 = cv2.imread(img_path2)

number_of_classes = 7
batch_size = 1
v7_object = build.Yolo_Infer_CPP.Yolov7(number_of_classes, anchors, model_path, batch_size, provider, letter_box, letter_box_color)
conf_thresh = 0.3



spec = '/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/spec.piyush.final.v7.json'
# provider = 'onnx-openvino-cpu'
# provider = 'onnx-cpu'
provider = 'onnx-tensorrt'
# provider = 'onnx-gpu'

model_config = build.Yolo_Infer_CPP.ModelConfig(spec, provider,
                                     batch_size, conf_thresh )


yolo_base = build.Yolo_Infer_CPP.Yolobase(model_config)  


video_path = "/docker/videos/Gandhinagar_Testing_Jun7/10_sec.mp4"
video = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
output_path = '/docker/python_app/tasks/output/onnx_cpu.mp4'  # Update with your output video file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
out_video = cv2.VideoWriter(output_path, fourcc, 25.0, (frame_width, frame_height))  # Adjust fps if needed

count = 0
while True: 
    batch_list = []
    ret, full_image = cap.read()
    if ret == False:
        break
    # full_image = cv2.imread(img_path)
    
    batch_list.append(full_image)
      
    
    preprocessed_img_cpp = yolo_base.preprocess_batch(batch_list) 
    
    
    
    inferenced_output = yolo_base.infer_cpp(preprocessed_img_cpp)
    
    
    # print(inferenced_output)
    out = []
    for key in reversed(inferenced_output):
        print(key , inferenced_output[key].shape)
        out.append(inferenced_output[key])
    
    list_of_boxes = v7_object.postprocess_batch(out, conf_thresh , 0.45 , full_image.shape[0] , full_image.shape[1])


    
      
    for k in range(batch_size):
        full = batch_list[k].copy()
        
        boxes = list_of_boxes[k][0]
        cls = list_of_boxes[k][1]
        score = list_of_boxes[k][2]
        # print(k, len(boxes))
        count+=len(boxes)
        for i in range(len(boxes)) :
            x1 = boxes[i][0]
            y1 = boxes[i][1]
            x2 = boxes[i][2]
            y2 = boxes[i][3]
            print(x1, y1, x2, y2, cls[i], score[i])
            
            cv2.rectangle(full, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
        # cv2.imwrite('/docker/image/openvino'+str(k)+'.jpg', full)
    out_video.write(full)
    # print("overall_time in py file", (time.time() - start_batch_time)*1000)
    # print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
    
    print(count)

cap.release()
out_video.release()