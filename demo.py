import cv2
import torch
import time

from yolo_detectAPI import DetectAPI


if __name__ == '__main__':
    
    camera_index = 0
    capture = cv2.VideoCapture(camera_index)

    yolo_api = DetectAPI(
        weights='weights/best.pt', 
        device='0', 
        conf_thres=0.4, iou_thres=0.1,
        half=False)
    
    with torch.no_grad():
        while True:
            ret, frame = capture.read()
            if not ret:
                continue
            start_time = time.time()
            
            result, names = yolo_api.detect([frame])
            drew_image = result[0][0]  # 每一帧图片的处理结果图片
            # 每一帧图像的识别结果（可包含多个物体）
            for cls, (x1, y1, x2, y2), conf in result[0][1]:
                print(f"name: {names[cls]}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}")  # 识别物体种类、左上角x坐标、左上角y轴坐标、右下角x轴坐标、右下角y轴坐标，置信度
                '''
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
                cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
                '''
            # print()  # 将每一帧的结果输出分开
            cv2.imshow("drew_image", drew_image)

            if cv2.waitKey(1) == ord('q'):
                break
            print(f"Infer FPS: {1/(time.time() - start_time)}")