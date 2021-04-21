from flask import Flask, render_template, Response
from v3_fastest import *
from v4_tiny import *

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


def v3_fastest(camera):
    global model
    while True:
        frame = camera.get_frame()
        # print(frame)
        v3_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg, 所以前端要接收的应该是图片enimcodo后的base64
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def v4_tiny(camera):
    global model
    while True:
        frame = camera.get_frame()
        v4_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    if model == 'v3_fastest':
        return Response(v3_fastest(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    if model == 'v4_tiny':
        return Response(v4_tiny(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection using YOLO-Fastest in OPENCV')
    parser.add_argument('--model', type=str, default='', help='v3_fastest or v4_tiny or Nano')
    args = parser.parse_args()
    model = args.model
    app.run(host='0.0.0.0', debug=True, port=5000)
