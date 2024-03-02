# -*- coding:utf-8 -*-
import os
import random
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync

"""
使用面向对象编程中的类来封装，需要去除掉原始 detect.py 中的结果保存方法，重写
保存方法将结果保存到一个 csv 文件中并打上视频的对应帧率

"""


class YoloOpt:
    def __init__(self, weights='weights/last.pt',
                 imgsz=(640, 640), conf_thres=0.4,
                 iou_thres=0.1, device='0', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False,
                 project='/detect/result', name='result_exp',
                 save_csv=True):
        self.weights = weights  # 权重文件地址
        self.source = None  # 待识别的图像
        if imgsz is None:
            self.imgsz = (640, 640)
        self.imgsz = imgsz  # 输入图片的大小，默认 (640,640)
        self.conf_thres = conf_thres  # object置信度阈值 默认0.25  用在nms中
        self.iou_thres = iou_thres  # 做nms的iou阈值 默认0.45   用在nms中
        self.device = device  # 执行代码的设备，由于项目只能用 CPU，这里只封装了 CPU 的方法
        self.view_img = view_img  # 是否展示预测之后的图片或视频 默认False
        self.classes = classes  # 只保留一部分的类别，默认是全部保留
        self.agnostic_nms = agnostic_nms  # 进行NMS去除不同类别之间的框, 默认False
        self.augment = augment  # augmented inference TTA测试时增强/多尺度预测，可以提分
        self.update = update  # 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        self.exist_ok = exist_ok  # 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        self.project = project  # 保存测试日志的参数，本程序没有用到
        self.name = name  # 每次实验的名称，本程序也没有用到
        self.save_csv = save_csv  # 是否保存成 csv 文件，本程序目前也没有用到


class DetectAPI:
    def __init__(self, weights, imgsz=(640,640), device=None, conf_thres=None, iou_thres=None):
        """
        Init Detect API
        Args:
            weights: model
            imgsz: default 640
            conf_thres: 用于物体的识别率，object置信度阈值 默认0.25，大于此准确率才会显示识别结果
            iou_thres: 用于去重，做nms的iou阈值 默认0.45，数值越小去重程度越高
        """
        self.opt = YoloOpt(weights=weights, imgsz=imgsz)
        if conf_thres is not None:
            self.opt.conf_thres = conf_thres
        if iou_thres is not None:
            self.opt.iou_thres = iou_thres
        weights = self.opt.weights
        imgsz = self.opt.imgsz

        # Initialize 初始化
        # 获取设备 CPU/CUDA
        if device is None:
            self.device = select_device(self.opt.device)
        else:
            self.device = select_device(device)        
        # 不使用半精度
        self.half = self.device.type != 'cpu'  # # FP16 supported on limited backends with CUDA
        self.half = False

        # Load model 加载模型
        self.model = DetectMultiBackend(weights, self.device, dnn=False, fp16=self.half)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # 不使用半精度
        if self.half:
            self.model.half()  # switch to FP16

        # read names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        self.visualize = False
    

    def detect(self, source):
        # 输入 detect([img])
        if type(source) != list:
            raise TypeError('source must a list and contain picture read by cv2')

        # DataLoader 加载数据
        # 直接从 source 加载数据
        dataset = LoadImages(source)
        # 源程序通过路径加载数据，现在 source 就是加载好的数据，因此 LoadImages 就要重写
        bs = 1  # set batch size

        # 保存的路径
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        result = []
        #if self.device.type != 'cpu':
        #    self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
        #        next(self.model.parameters())))  # run once
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *(640,640)))  # warmup
        dt, seen = (Profile(), Profile(), Profile()), 0

        for im, im0s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                self.visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = self.model(im, augment=self.opt.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                               self.opt.agnostic_nms, max_det=2)

            # Process predictions
            # 处理每一张图片
            det = pred[0]  # API 一次只处理一张图片，因此不需要 for 循环
            im0 = im0s.copy()  # copy 一个原图片的副本图片
            result_txt = []  # 储存检测结果，每新检测出一个物品，长度就加一。
            # 每一个元素是列表形式，储存着 类别，坐标，置信度
            # 设置图片上绘制框的粗细，类别名称
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 映射预测信息到原图
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                 #
                for *xyxy, conf, cls in reversed(det):
                    line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # abel format
                    result_txt.append(line)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=self.colors[int(cls)])
            result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
        return result, self.names


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    a = DetectAPI(weights='weights/last.pt', device='0')
    with torch.no_grad():
        while True:
            rec, img = cap.read()
            result, names = a.detect([img])
            img = result[0][0]  # 每一帧图片的处理结果图片
            # 每一帧图像的识别结果（可包含多个物体）
            for cls, (x1, y1, x2, y2), conf in result[0][1]:
                print(names[cls], x1, y1, x2, y2, conf)  # 识别物体种类、左上角x坐标、左上角y轴坐标、右下角x轴坐标、右下角y轴坐标，置信度
                '''
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
                cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))'''
            print()  # 将每一帧的结果输出分开
            cv2.imshow("vedio", img)

            if cv2.waitKey(1) == ord('q'):
                break
