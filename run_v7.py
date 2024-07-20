
import time
import build.run_yolo_onnx
import numpy as np
import cv2




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
batch_size = 1
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


# 254.91073608398438 258.75164794921875 450.87884521484375 639.0 1 0.6499114036560059
# 336.71038818359375 364.77490234375 390.8795166015625 401.9630432128906 6 0.46018850803375244
# 385.8459777832031 104.02576446533203 429.4844970703125 128.59446716308594 6 0.35598239302635193


# 254.92803955078125 258.74554443359375 450.8409423828125 639.0 1 0.6499994993209839
# 336.7172546386719 364.777587890625 390.8779296875 401.95953369140625 6 0.4605385363101959
# 385.85101318359375 104.03095245361328 429.47296142578125 128.58763122558594 6 0.3558728098869324