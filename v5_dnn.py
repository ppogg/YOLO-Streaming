import cv2
import numpy as np

model = 'model/yolov5s-320.onnx'
names_file = 'data/coco.names'
with open(names_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

class yolov5():
    def __init__(self, confThreshold = 0.2, nmsThreshold = 0.2, objThreshold = 0.4):
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]
        num_classes = len(classes)
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)

        self.net = cv2.dnn.readNet(model)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.Height = 320
        self.Width = 320

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def v5_inference(self, frame):
        import time
        begin = time.time()
        outs = self.detect(frame)
        frameHeight, frameWidth = frame.shape[0], frame.shape[1]
        ratioh, ratiow = frameHeight / self.Height, frameWidth / self.Width
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold and detection[4] > self.objThreshold:
                    center_x = int(detection[0] * ratiow)
                    center_y = int(detection[1] * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        time = (time.time() - begin) * 1000
        print('Inference time: %.2f ms' % time)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        color = [int(c) for c in COLORS[classId]]
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=2)
        label = '%.2f' % conf
        label = '%s:%s' % (classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        # cv2.putText(frame, Time, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return frame

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.Height, self.Width), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = outs[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # outs[i] = outs[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            outs[i] = outs[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            if self.grid[i].shape[2:4] != outs[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-outs[i]))  ### sigmoid
            ###其实只需要对x,y,w,h做sigmoid变换的， 不过全做sigmoid变换对结果影响不大，因为sigmoid是单调递增函数，那么就不影响类别置信度的排序关系，因此不影响后面的NMS
            ###不过设断点查看类别置信度，都是负数，看来有必要做sigmoid变换把概率值强行拉回到0到1的区间内
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, self.no))
        z = np.concatenate(z, axis=1)
        return z

if __name__ == "__main__":
    yolonet = yolov5()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 960)  # set video width
    cap.set(4, 780)  # set video height
    while True:
        ret, frame = cap.read()
        yolonet.v5_inference(frame)
        cv2.imshow('fourcc', frame)
        k = cv2.waitKey(20)
        # q键退出
        if (k & 0xff == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
