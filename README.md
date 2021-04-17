# Push-Streaming

Hi, this repository documents the process of pushing streams on some ultra-lightweight nets. The general steps are that opencv calls the **board**（like Raspberry Pi）'s camera, transmits the detected live video to an ultra-lightweight network like **yolo-fastest, nanodet**, **ghostnet**, and then talks about pushing the processed video frames to the web using the **flask** lightweight framework, which basically guarantees **real-time** performance.


# Requirements

Please install the following packages first
-   Linux & MacOS & window
- python>= 3.6.0
- opencv-python>= 4.2.X
- flask>= 1.0.0
## inference
- Yolo-Fastest： [https://github.com/dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
    Models：[Yolo-Fastest-1.1-xl](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)

Equipment | Computing backend | System | Framework | Run time
 :-----:|:-----:|:-----:|:----------:|:----:|
Raspberrypi 3B| 4xCortex-A53 | Linux(arm64) | dnn | 89ms
Intel | Core i5-4210 | window10（x64） | dnn | 67ms


- Nanodet： [https://github.com/RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)


   updating. . . 



## Demo

First of all, I have tested this demo in window, mac and linux environments and it works in all of them.



-   Inference images

```python yolov3_faster.py --image dog.jpg```

-   Inference video

```python yolov3_faster.py --video test.mp4```

-   Inference webcam

```python yolov3_faster.py --fourcc 0```
-   Push-Streaming

```python app.py```

⚡  **Please note! Be sure to be on the same LAN！**
##  Demo Effects

-   Demo images
[![image](https://github.com/pengtougu/Push-Streaming/blob/master/result/dog.jpg)](https://github.com/pengtougu/Push-Streaming/blob/master/result/dog.jpg)
-   Demo video
[![image](https://github.com/pengtougu/Push-Streaming/blob/master/result/video_cut.jpg)](https://github.com/pengtougu/Push-Streaming/blob/master/result/video_cut.jpg)
-   Demo camera
[![image](https://github.com/pengtougu/Push-Streaming/blob/master/result/capture.jpg)](https://github.com/pengtougu/Push-Streaming/blob/master/result/capture.jpg)
-   Demo Push-Streaming
[![image](https://github.com/pengtougu/Push-Streaming/blob/master/result/stream.jpg)](https://github.com/pengtougu/Push-Streaming/blob/master/result/stream.jpg)

##  Thanks

-   [https://github.com/dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
-   [https://github.com/hpc203/Yolo-Fastest-opencv-dnn](https://github.com/hpc203/Yolo-Fastest-opencv-dnn)
-  [https://github.com/miguelgrinberg/flask-video-streaming](https://github.com/miguelgrinberg/flask-video-streaming)

