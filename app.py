from flask import Flask, render_template, Response
import cv2
from yolov3_faster import *

# Initialize the parameters
confThreshold = 0.25  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 320  # Width of network's input image
inpHeight = 320  # Height of network's input image

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "Yolo-Fastest-coco/yolo-fastest-xl.cfg"
modelWeights = "Yolo-Fastest-coco/yolo-fastest-xl.weights"
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]
np.random.seed(42)
colors = []
colors = np.random.randint(0, 255, size=(len(classes), 3),
                                   dtype="uint8")

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)
        # self.video.set(3, 960)  # set video width
        # self.video.set(4, 720)  # set video height

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return image


app = Flask(__name__)


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


def gen(camera):
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # Process inputs

    while True:
        frame = camera.get_frame()
        # print(frame)
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (inpWidth, inpHeight), [0, 0, 0], swapRB=False, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg, 所以前端要接收的应该是图片enimcodo后的base64
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port = 5000)