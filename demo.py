import cv2
import torch
import time

from yolo_detectAPI import DetectAPI

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    a = DetectAPI(weights='weights/best.pt', device='0', conf_thres=0.4, iou_thres=0.1)
    with torch.no_grad():
        while True:
            rec, img = cap.read()
            start_time = time.time()
            result, names = a.detect([img])
            img = result[0][0]  # 每一帧图片的处理结果图片
            # 每一帧图像的识别结果（可包含多个物体）
            for cls, (x1, y1, x2, y2), conf in result[0][1]:
                # print(names[cls], x1, y1, x2, y2, conf)  # 识别物体种类、左上角x坐标、左上角y轴坐标、右下角x轴坐标、右下角y轴坐标，置信度
                '''
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
                cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))'''
            # print()  # 将每一帧的结果输出分开
            cv2.imshow("video", img)

            if cv2.waitKey(1) == ord('q'):
                break
            print(f"FPS: {1/(time.time() - start_time)}")