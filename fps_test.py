
import time
import build.run_yolo_onnx
import numpy as np
import cv2




# ModelConfig::ModelConfig(const std::string &json_config_path, const std::string &provider,
# 	    const int64_t &batch_size, const float &confidence_threshold, const bool draw_blobs_on_frames, 
#         const std::string &infer_blob, std::map<std::string, std::vector<float>> preprocesses)


model_path = "/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/piyush.best.416.v7.onnx"
letter_box = False
letter_box_color = [0,0,0]

provider='cpu'

anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]

img_path = "/docker/deepak/image/2_jpg.rf.99922942d0a3d839f3d2cba6fb3716bf.jpg"
full_image = cv2.imread(img_path)
number_of_classes = 7
batch_size = 2
v7_object = build.run_yolo_onnx.Yolov7(number_of_classes, anchors, model_path, batch_size, provider, letter_box, letter_box_color)
conf_thresh = 0.3



spec = '/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/spec.piyush.final.v7.json'
provider = 'onnx-openvino-cpu'

'''

model_config = build.run_yolo_onnx.ModelConfig(spec, provider,
                                     batch_size, conf_thresh, False,'no', {}  )

yolo_base = build.run_yolo_onnx.Yolobase(model_config)  

while True: 
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

    
    start_batch_time= time.time()    
    
    preprocessed_img_cpp = yolo_base.preprocess_batch(batch_list) 
    
    inferenced_output = yolo_base.detect_ov(preprocessed_img_cpp)

    print("len of inferenced_output",len(inferenced_output))
   
    list_of_boxes = v7_object.postprocess_batch(inferenced_output, conf_thresh , 0.45 , full_image.shape[0] , full_image.shape[1])

    
    for k in range(batch_size):
        full_image = batch_list[k]
        
        boxes = list_of_boxes[k][0]
        cls = list_of_boxes[k][1]
        score = list_of_boxes[k][2]
        # print(k, len(boxes))
        for i in range(len(boxes)) :
            x1 = boxes[i][0]
            y1 = boxes[i][1]
            x2 = boxes[i][2]
            y2 = boxes[i][3]
            print(x1, y1, x2, y2, cls[i], score[i])
            cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
        cv2.imwrite('/docker/deepak/image/v7_outputb'+str(k)+'.jpg', full_image)
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
    exit()

'''

model_path = "/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/piyush.best.416.v7.onnx"
letter_box = False
letter_box_color = [0,0,0]
provider='cpu'

anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]

img_path = "/docker/deepak/image/2_jpg.rf.99922942d0a3d839f3d2cba6fb3716bf.jpg"
full_image = cv2.imread(img_path)
number_of_classes = 7
batch_size = 2
v7_object = build.run_yolo_onnx.Yolov7(number_of_classes, anchors, model_path, batch_size, provider, letter_box, letter_box_color)
conf_thresh = 0.3



while True: 
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

    
    start_batch_time= time.time()    
    
    preprocessed_img_cpp = v7_object.preprocess_batch(batch_list) 
    
    inferenced_output = v7_object.detect(preprocessed_img_cpp)

    print("len of inferenced_output",len(inferenced_output))
    
    
    list_of_boxes = v7_object.postprocess_batch(inferenced_output, conf_thresh , 0.45 , full_image.shape[0] , full_image.shape[1])
    5106261372566223
# 339.48004150390625 366.2054138183594 390.4420471191406 398.57769775390625 6 0.4777664244174957
    # print(len(list_of_boxes))
    # print(len(batch_list))
    
    for k in range(batch_size):
        full_image = batch_list[k]
        
        boxes = list_of_boxes[k][0]
        cls = list_of_boxes[k][1]
        score = list_of_boxes[k][2]
        # print(k, len(boxes))
        for i in range(len(boxes)) :
            x1 = boxes[i][0]
            y1 = boxes[i][1]
            x2 = boxes[i][2]
            y2 = boxes[i][3]
            print(x1, y1, x2, y2, cls[i], score[i])
        #     cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
        # cv2.imwrite('/docker/deepak/image/v7_outputaaa'+str(k)+'.jpg', full_image)
        # out.write(full_image)
     
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
    exit()




# 256.38555908203125 262.025390625 458.022216796875 636.450927734375 1 0.5106261372566223
# 256.38555908203125 262.025390625 458.022216796875 636.450927734375 1 0.5106261372566223

# 256.3775939941406 262.0298156738281 458.0533142089844 636.4268188476562 1 0.5103762149810791
# 256.3775939941406 262.0298156738281 458.0533142089844 636.4268188476562 1 0.5103762149810791




# 256.3775939941406 262.0298156738281 458.0533142089844 636.4268188476562 1 0.5103762149810791
# 339.4759826660156 366.20404052734375 390.4422607421875 398.58709716796875 6 0.47748276591300964


# 256.38555908203125 262.025390625 458.022216796875 636.450927734375 1 0.5106261372566223
# 339.48004150390625 366.2054138183594 390.4420471191406 398.57769775390625 6 0.4777664244174957

# 254.91073608398438 258.75164794921875 450.87884521484375 639.0 1 0.6499114036560059
# 336.71038818359375 364.77490234375 390.8795166015625 401.9630432128906 6 0.46018850803375244
# 385.8459777832031 104.02576446533203 429.4844970703125 128.59446716308594 6 0.35598239302635193

# 254.91073608398438 258.75164794921875 450.87884521484375 639.0 1 0.6499114036560059
# 336.71038818359375 364.77490234375 390.8795166015625 401.9630432128906 6 0.46018850803375244
# 385.8459777832031 104.02576446533203 429.4844970703125 128.59446716308594 6 0.35598239302635193
