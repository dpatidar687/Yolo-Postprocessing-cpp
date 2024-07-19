
import time
import build.run_yolo_onnx
import numpy as np
import cv2

mc = build.run_yolo_onnx.ModelConfig('/docker/deepak/Yolo-Postprocessing-cpp/spec.json', 'onnx-openvino-cpu',
                                     1, 0.3, False,'no', {}  )

yc = build.run_yolo_onnx.Yolobase(
 mc   
)


save_path = '/docker/deepak/yolo_onnx_release/image/'
img_path = "/docker/deepak/image/2_jpg.rf.99922942d0a3d839f3d2cba6fb3716bf.jpg"


start_time =  time.time()
batch_size = 1
number_of_classes = 7
model_path = "/docker/models/anpr_plate_vehicle_detector.tiny_yolov7/v1/onnx/v7_c_12.onnx"
letter_box = True
letter_box_color = [114, 114, 114]

provider='cpu'

anchors = [[116, 90, 156, 198, 373, 326],
         [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23]]




v7_object = build.run_yolo_onnx.Yolov7(number_of_classes, anchors, model_path, batch_size, provider, letter_box, letter_box_color)

full_image = cv2.imread(img_path)


height = width = 416
def preprocess_py(batch_list):
    img_array = []
    preprocessed_img = np.empty(())
    for i in batch_list:
        img =i
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (height, width), interpolation = cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img = img/255.0
        img_array.append(img)
        
    preprocessed_img = np.array(img_array)
    preprocessed_img = preprocessed_img.reshape(batch_size*height*width*3)
    return preprocessed_img



while True: 
    batch_list = []
    for i in range(batch_size):
        batch_list.append(full_image)

    
    start_batch_time = time.time()
    preprocessed_img_cpp = v7_object.preprocess_batch(batch_list) 
    pre = yc.preprocess_batch(batch_list) 
    print("preprocess time in py file ",(time.time() - start_batch_time)*1000)

    

    detect_start_time = time.time()   
    inferenced_output = v7_object.detect(pre)
    io = yc.detect_ov(preprocessed_img_cpp)

    
    print(len(inferenced_output), len(io))
    for i in range(3):
        print(len(inferenced_output[i]), len(io[i]))
        
    print(inferenced_output)
    print(io)

    print("detection time in python file ", (time.time() - detect_start_time )* 1000)
    
        
    post_start_time = time.time()
    list_of_boxes = v7_object.postprocess_batch(inferenced_output, 0.5 , 0.45 , full_image.shape[0] , full_image.shape[1])
    print("post time in py file",(time.time() - post_start_time)*1000)
    
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
            cv2.rectangle(full_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0,0), 3)
        cv2.imwrite('/docker/deepak/image/v7_output'+str(k)+'.jpg', full_image)
        # out.write(full_image)
     
    print("overall_time in py file", (time.time() - start_batch_time)*1000)
    print("Batch_FPS in py file ", batch_size/(time.time() - start_batch_time))
    print("------------------------------------------------------------------------------------")
    exit()


cap.release()
out.release()




# 256.38555908203125 262.025390625 458.022216796875 636.450927734375 1 0.5106261372566223
# 256.38555908203125 262.025390625 458.022216796875 636.450927734375 1 0.5106261372566223

# 256.3775939941406 262.0298156738281 458.0533142089844 636.4268188476562 1 0.5103762149810791
# 256.3775939941406 262.0298156738281 458.0533142089844 636.4268188476562 1 0.5103762149810791
