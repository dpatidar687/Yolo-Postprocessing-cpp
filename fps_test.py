
import time
import build.run_yolo_onnx
import numpy as np
import cv2




model_path = "/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/piyush.best.416.v7.onnx"
letter_box = True
letter_box_color = [114, 114, 114]

provider='gpu'

anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]

img_path = "/docker/image/2_jpg.rf.99922942d0a3d839f3d2cba6fb3716bf.jpg"
full_image = cv2.imread(img_path)
img_path2 = "/docker/image/openvino2.jpg"
full_image2 = cv2.imread(img_path2)


number_of_classes = 7
batch_size = 4
v7_object = build.run_yolo_onnx.Yolov7(number_of_classes, anchors, model_path, batch_size, provider, letter_box, letter_box_color)
conf_thresh = 0.3



spec = '/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/spec.piyush.final.v7.json'
provider = 'onnx-openvino-cpu'
# provider = 'onnx-tensorrt'
# provider = 'onnx-cpu'
# 

model_config = build.run_yolo_onnx.ModelConfig(spec, provider,
                                     batch_size, conf_thresh )


yolo_base = build.run_yolo_onnx.Yolobase(model_config)  

while True: 
    batch_list = []
    # for i in range(batch_size):
    #     batch_list.append(full_image)
    batch_list.append(full_image)
    batch_list.append(full_image2)
    batch_list.append(full_image)
    batch_list.append(full_image2)
    
    
    start_batch_time= time.time()    
    
    preprocessed_img_cpp = yolo_base.preprocess_batch(batch_list) 
    
    print("pre : ", preprocessed_img_cpp[0], preprocessed_img_cpp[1], preprocessed_img_cpp[2])
    inferenced_output = yolo_base.detect_ov(preprocessed_img_cpp)
    # inferenced_output_v7 = v7_object.detect(preprocessed_img_cpp)
    
    # print("len of inferenced_output",len(inferenced_output))
    
    print("inf  : ", inferenced_output[0][0], inferenced_output[0][1], inferenced_output[0][2])
    # print(inferenced_output_v7[0][0], inferenced_output_v7[0][1], inferenced_output_v7[0][2])
    
    # list_of_boxes = v7_object.postprocess_batch(inferenced_output, conf_thresh , 0.45 , full_image.shape[0] , full_image.shape[1])

    
    # for k in range(batch_size):
    #     full = batch_list[k].copy()
        
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
    #     #     cv2.rectangle(full, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
    #     # cv2.imwrite('/docker/image/openvino'+str(k)+'.jpg', full)
    # print("overall_time in py file", (time.time() - start_batch_time)*1000)
    # print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    # print("------------------------------------------------------------------------------------")

    # exit()

