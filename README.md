
# Push-Streaming

Hi, this repository documents the process of pushing streams on some ultra-lightweight nets. The general steps are that opencv calls the **board**（like Raspberry Pi）'s camera, transmits the detected live video to an ultra-lightweight network like **yolo-fastest, nanodet**, **ghostnet**, and then talks about pushing the processed video frames to the web using the **flask** lightweight framework, which basically guarantees **real-time** performance.

<img src="https://github.com/pengtougu/Push-Streaming/blob/master/result/step.png" width="700" height="500" alt="step"/><br/>

# Requirements

Please install the following packages first
-   Linux & MacOS & window
- python>= 3.6.0
- opencv-python>= 4.2.X
- flask>= 1.0.0
## inference
- YOLOv3-Fastest： [https://github.com/dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
    Models：[Yolo-Fastest-1.1-xl](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)
    
Equipment | Computing backend | System | Framework | Run time
 :-----:|:-----:|:-----:|:----------:|:----:|
Raspberrypi 3B| 4xCortex-A53 | Linux(arm64) | dnn | 89ms
Intel | Core i5-4210 | window10（x64） | dnn | 67ms


- YOLOv4-Tiny： [https://github.com/AlexeyAB/darknet](https://github.com/dog-qiuqiu/Yolo-Fastest)
    Models：[yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

Equipment | Computing backend | System | Framework | Run time
 :-----:|:-----:|:-----:|:----------:|:----:|
Raspberrypi 3B| 4xCortex-A53 | Linux(arm64) | dnn | 97ms
Intel | Core i5-4210 | window10（x64） | dnn | 71ms


- YOLOv5s-onnx： [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

Equipment | Computing backend | System | Framework | Run time
 :-----:|:-----:|:-----:|:----------:|:----:|
Raspberrypi 3B| 4xCortex-A53 | Linux(arm64) | dnn | 973ms
Intel | Core i5-4210 | window10（x64） | dnn | 197ms

   
- Nanodet： [https://github.com/RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)

   updating. . . 
- Efficientnet-lite： [https://github.com/rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)

   updating. . . 

## Demo

First of all, I have tested this demo in window, mac and linux environments and it works in all of them.

**Run v3_fastest.py**
-  	Inference images use ```python yolov3_fastest.py --image dog.jpg```
-   Inference video use ```python yolov3_fastest.py --video test.mp4```
-   Inference webcam use ```python yolov3_fastest.py --fourcc 0```

**Run v4_tiny.py**
-  	Inference images use ```python v4_tiny.py --image person.jpg```
-   Inference video use ```python v4_tiny.py --video test.mp4```
-   Inference webcam use ```python v4_tiny.py --fourcc 0```

**Run app.py**    -（Push-Streaming online）

-  	Inference with v3-fastest ```python app.py --model v3_fastest```
-   Inference with v4-tiny ```python app.py --model v4_tiny```
- Inference with v4-tiny ```python app.py --model v5_dnn```

⚡  **Please note! Be sure to be on the same LAN！**
##  Demo Effects
**Run v3_fastest.py**

-  	image→video→capture→push stream

<img src="https://github.com/pengtougu/Push-Streaming/blob/master/result/v3_merge.png" width="700" height="600" alt="stream"/><br/>

**Run v4_tiny.py**

-  	image→video→capture→push stream

**Run v5_dnn.py**

-  	image(473 ms / Inference Image / Core i5-4210)→video→capture(213 ms / Inference Image / Core i5-4210)→push stream
很有意思的是，用onnx＋dnn的方式调用v5s的模型，推理图片要比摄像头处理帧多花一倍的时间，这里摸鱼调了一个下午，还是找不出问题所在，希望看到的大佬可以帮看看代码，点破问题所在，感谢！
##  Thanks

-   [https://github.com/dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
-   [https://github.com/hpc203/Yolo-Fastest-opencv-dnn](https://github.com/hpc203/Yolo-Fastest-opencv-dnn)
-  [https://github.com/miguelgrinberg/flask-video-streaming](https://github.com/miguelgrinberg/flask-video-streaming)
- [https://github.com/hpc203/yolov5-dnn-cpp-python](https://github.com/hpc203/yolov5-dnn-cpp-python)


##  other
-   中文操作教程：[https://blog.csdn.net/weixin_45829462/article/details/115806322](https://blog.csdn.net/weixin_45829462/article/details/115806322)
