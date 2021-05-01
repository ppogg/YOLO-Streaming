import cv2
import argparse
import numpy as np

model = 'data/yolov4-tiny.weights'
config = r'data/yolov4-tiny.cfg'
labelsPath = r"data/coco.names"

v4tiny_Net = cv2.dnn.readNetFromDarknet(config, model)
ln = v4tiny_Net.getLayerNames()
ln = [ln[i[0] - 1] for i in v4tiny_Net.getUnconnectedOutLayers()]

LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = []
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
conf = 0.5
threshold = 0.5

def postprocess(frame, networkOutput):
    (H, W) = frame.shape[:2]
    boxes = []
    confidences = []
    classIDs = []
    for output in networkOutput:
        """
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                   dtype="uint8")
        # 每过一次循环随机换一次颜色，如果不想换颜色可以直接放到循环外面
        # 放里面只适用于只有一类目标的时候，如果有多类目标，建议放到外面
        """

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            # end = time.time()
            # seconds = end - begin
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)


# def draw_time(frame, t):
#     fps = 1.0 / t * 10000000
#     label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
#     cv2.putText(frame, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#     cv2.putText(frame, "FPS: %.2f" % fps, (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # cv2.putText(frame, "TIME: " + str(localtime), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def v4_inference(frame):
    v4tiny_Net.setInput(cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False))
    networkOutput = v4tiny_Net.forward(ln)
    postprocess(frame, networkOutput)
    # localtime = time.asctime(time.localtime(time.time()))
    t, _ = v4tiny_Net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    # draw_time(frame, t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detectiom using YOLOv4 in OPENCV')
    parser.add_argument('--image', type=str, default='', help='Path to image file.')
    parser.add_argument('--fourcc', type=int, default=1, help='Open the VideoCapture')
    parser.add_argument('--video', type=str, default='', help='Open the video')
    args = parser.parse_args()

    if args.fourcc == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1280)  # set video width
        cap.set(4, 960)  # set video height
        while True:
            # begin = time.time()
            ret, frame = cap.read()
            v4_inference(frame)
            cv2.imshow('capture', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    elif args.image:
        frame = cv2.imread(args.image)
        v4_inference(frame)
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cap = cv2.VideoCapture(args.video)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('result.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        font = cv2.FONT_HERSHEY_SIMPLEX

        while (cap.isOpened()):
            ret, frame = cap.read()
            v4_inference(frame)
            out.write(frame)
            cv2.imshow('video', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()