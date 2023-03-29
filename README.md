


# Introduction

YoloV5 作为 YoloV4 之后的改进型，在算法上做出了优化，检测的性能得到了一定的提升。其特点之一就是权重文件非常的小，可以在一些配置更低的移动设备上运行，且提高速度的同时准确度更高。本次使用的是最新推出的 YoloV5 Version7 版本。

GitHub 地址：[YOLOv5 🚀 是世界上最受欢迎的视觉 AI，代表 Ultralytics 对未来视觉 AI 方法的开源研究，结合在数千小时的研究和开发中积累的经验教训和最佳实践。](https://github.com/ultralytics/yolov5/releases/tag/v7.0)



---

# Section 1 起因

本人目前的一个项目需要使用到手势识别，得益于 YoloV5 的优秀的识别速度与准确率，因此识别部分的模型均使用 YoloV5 Version7 版本进行训练。训练之后需要使用这个模型，原始的 `detect.py` 程序使用 `argparse` 对参数进行封装，这为初期验证模型提供了一定的便利，我们可以通过 `Pycharm` 或者 `Terminal` 来快速地执行程序，然后在 `run/detect` 路径下快速地查看到结果。但是在实际的应用中，识别程序往往是作为整个系统的一个组件来运行的，现有的 `detect.py` 无法满足使用需求，因此需要将其封装成一个可供多个程序调用的 `API` 接口。通过这个接口可以获得 `种类、坐标、置信度` 这三个信息。通过这些信息来控制系统软件做出对应的操作。



---

# Section 2 魔改的思路

这部分的代码与思路参照了 [爆改YOLOV7的detect.py制作成API接口供其他python程序调用（超低延时）](https://blog.csdn.net/weixin_51331359/article/details/126012620) 这篇文章的思路。由于 YoloV5 和 YoloV7 的程序有些许不一样，因此做了一些修改。

大体的思路是去除掉 `argparse` 部分，通过类将参数封装进去，去除掉识别这个核心功能之外的其它功能。

未打包程序见博客 [魔改并封装 YoloV5 Version7 的 detect.py 成 API接口以供 python 程序使用](https://blog.csdn.net/qq_17790209/article/details/129061528)



# Section 3 如何安装到 Python 环境

从 `whl` 文件夹或者从`Release`下载 `yolo_detectAPI-5.7-py3-none-any.whl` ，在下载目录内进入 Terminal 并切换至你要安装的 Python 环境。输入下面的命令安装 Python 库。这里需要注意，Python 环境需要 3.8 及以上版本才能使用。

```shell
pip install .\yolo_detectAPI-5.7-py3-none-any.whl
```

这个库使用 CPU 执行程序，如果需要使用 GPU 执行程序请 clone 源码自行打包修改程序。

自行打包需要进入到 clone 之后的项目的根目录，打开终端输入下面的命令，然后在 `dist` 文件夹内就可找到你需要的 `whl` 文件。

```python
python setup.py sdist bdist_wheel
```



# Section 4 如何在项目中使用

使用下面的代码就可以引用这个库。其中的 `cv2`,`torch` 在没有特定版本需求的情况下不需要单独安装，安装本API库的时候程序会自动安装这些依赖的库。

```python
import cv2
import yolo_detectAPI
import torch

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    a = yolo_detectAPI.DetectAPI(weights='last.pt', conf_thres=0.5, iou_thres=0.5)  # 你要使用的模型的路径
    with torch.no_grad():
        while True:
            rec, img = cap.read()
            result, names = a.detect([img])
            img = result[0][0]  # 每一帧图片的处理结果图片
            # 每一帧图像的识别结果（可包含多个物体）
            for cls, (x1, y1, x2, y2), conf in result[0][1]:
                print(names[cls], x1, y1, x2, y2, conf)  # 识别物体种类、左上角x坐标、左上角y轴坐标、右下角x轴坐标、右下角y轴坐标，置信度
                
            cv2.imshow("video", img)

            if cv2.waitKey(1) == ord('q'):
                break
```



---

# Section 5 其他

其它问题欢迎进企鹅群交流：913211989 ( 小猫不要摸鱼 )

进群令牌：fGithub

不可商用，开源，论文引用请标注如下内容

```
[1] Da Kuang.YoloV5 Version7 Detect API for Python3[EB/OL]. https://github.com/Ender-William/YoloDetectAPI, 2023-02-17/引用日期{YYYY-MM-DD}.
```



---

# Reference
本程序的修改参考了以下的资料，在此为前人做出的努力与贡献表示感谢！

https://github.com/ultralytics/yolov5/releases/tag/v7.0
https://blog.csdn.net/weixin_51331359/article/details/126012620
https://blog.csdn.net/CharmsLUO/article/details/123422822

# Update Version 5.7.1 2023-03-29
添加了 `conf_thres` 和 `iou_thres` 的设置方法，在初始化识别方法时可以添加。
```python
yolo_detectAPI.DetectAPI(weights='last.pt', conf_thres=0.5, iou_thres=0.5)
```
`iou_thres` 过大容易出现一个目标多个检测框；

`iou_thres` 过小容易出现检测结果少的问题。