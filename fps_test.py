
import time
import build.Yolo_Infer_CPP
import numpy as np
import cv2

import build.Yolo_Infer_CPP


# model_path = "/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/piyush.best.416.v7.onnx"
# model_path = '/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/piyush.best.416.v3.onnx'
model_path = '/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/v7_c_12.onnx'
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
batch_size = 1
v7_object = build.Yolo_Infer_CPP.Yolov7(number_of_classes, anchors, model_path, batch_size, provider, letter_box, letter_box_color)
conf_thresh = 0.3



spec = '/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/spec.piyush.final.v7.json'
# provider = 'onnx-openvino-cpu'
# provider = 'onnx-cpu'
# provider = 'onnx-tensorrt'
provider = 'onnx-gpu'

model_config = build.Yolo_Infer_CPP.ModelConfig(spec, provider,
                                     batch_size, conf_thresh )


yolo_base = build.Yolo_Infer_CPP.Yolobase(model_config)  

while True: 
    batch_list = []
    # for i in range(batch_size):
    batch_list.append(full_image)
    # batch_list.append(full_image)
    # batch_list.append(full_image2)
    # batch_list.append(full_image)
    # batch_list.append(full_image2)
    
    
    start_batch_time= time.time()    
    
    preprocessed_img_cpp = yolo_base.preprocess_batch(batch_list) 
    
    
    
    inferenced_output = yolo_base.infer_cpp(preprocessed_img_cpp)
    # # print(inferenced_output)
    out = []
    for key in (inferenced_output):
        print( key , ":" , inferenced_output[key][0], inferenced_output[key][1], inferenced_output[key][2], inferenced_output[key][3], inferenced_output[key][4])
        out.append(inferenced_output[key])
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-+----------------------------------------------------------------")
    
    # inferenced_output_v7 = v7_object.detect(preprocessed_img_cpp)
    
    # print(len(inferenced_output_v7))
    
    
    list_of_boxes = v7_object.postprocess_batch(out, conf_thresh , 0.45 , full_image.shape[0] , full_image.shape[1])

    
    for k in range(batch_size):
        full = batch_list[k].copy()
        
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
        #     cv2.rectangle(full, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
        # cv2.imwrite('/docker/image/openvino'+str(k)+'.jpg', full)
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")

    # exit()

# # /model.77/m.2/Conv_output_0 [ 0.52747965 -0.42160907 -0.43888327 -0.7595067   0.06159086]
# # /model.77/m.1/Conv_output_1 [ 0.37329528 -0.5819995  -0.5670101  -0.34987402 -0.48597363]
# # /model.77/m.0/Conv_output_2 [ 1.085072   -0.43592775 -1.6875155  -0.25875378 -0.3825836 ]


 # list1 = []
    # list2 = []
    # for i in out:
    #     for j in i :
    #         list1.append(j)
    # for i in reversed(inferenced_output_v7):
    #     for j in i :
    #         list2.append(j)
            
    # print(len(list1))
    # print(len(list2))
    
    # count = 0
    # for i in range(len(list1)):
    #     if(list1[i] == list2[i]):
    #         print(list1[i], list2[i])
    #         count+=1
            
    # print(count)
    # print(list1[0], list2[0])
    # print(list1[1], list2[1])
    # print(list1[2], list2[2])
        
    # array1 = np.array(list1)
    # array2 = np.array(list2)

    # # Compute dot product
    # dot_product = np.dot(array1, array2)

    # # Compute magnitudes (norms) of the arrays
    # magnitude1 = np.linalg.norm(array1)
    # magnitude2 = np.linalg.norm(array2)
    
    # if magnitude1 == 0 or magnitude2 == 0:
    #     cosine_similarity = 0  # or handle it as appropriate for your application
    # else:
    #     cosine_similarity = dot_product / (magnitude1 * magnitude2)
    # # Calculate cosine similarity (dot product divided by magnitudes)
    # cosine_similarity = dot_product / (magnitude1 * magnitude2)

    # print(f"Dot Product: {dot_product}")
    # print(f"Magnitude of Array 1: {magnitude1}")
    # print(f"Magnitude of Array 2: {magnitude2}")
    # print(f"Cosine Similarity: {cosine_similarity}")
        
        
    # print("len of inferenced_output",len(inferenced_output))
    
    # print("inf  : ", inferenced_output[0][0], inferenced_output[0][1], inferenced_output[0][2])